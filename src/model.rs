use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Module, Tensor};
use candle_transformers::models::stable_diffusion;
use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;

pub struct AnimeModel {
    device: Device,
    dtype: DType,
    sd_config: stable_diffusion::StableDiffusionConfig,
    vae: stable_diffusion::vae::AutoEncoderKL,
    unet: stable_diffusion::unet_2d::UNet2DConditionModel,
    tokenizer: Tokenizer,
    clip_weights_path: std::path::PathBuf,
    n_steps: usize,
    _use_f16: bool,
}

impl AnimeModel {
    pub fn new(model_dir: &str, n_steps: usize, use_f16: bool, height: usize, width: usize) -> Result<Self> {
        let device = if candle_core::utils::metal_is_available() {
            println!("Using Metal GPU");
            Device::new_metal(0)?
        } else {
            println!("Metal not available, using CPU");
            Device::Cpu
        };
        let dtype = if use_f16 { DType::F16 } else { DType::F32 };

        let sd_config = stable_diffusion::StableDiffusionConfig::v1_5(None, Some(height), Some(width));

        println!("Loading VAE...");
        let vae_path = std::path::PathBuf::from(model_dir).join(if use_f16 {
            "vae/diffusion_pytorch_model.fp16.safetensors"
        } else {
            "vae/diffusion_pytorch_model.safetensors"
        });
        let vae = sd_config.build_vae(vae_path, &device, dtype)?;

        println!("Loading UNet...");
        let unet_path = std::path::PathBuf::from(model_dir).join(if use_f16 {
            "unet/diffusion_pytorch_model.fp16.safetensors"
        } else {
            "unet/diffusion_pytorch_model.safetensors"
        });
        let unet = sd_config.build_unet(unet_path, &device, 4, false, dtype)?;

        println!("Loading tokenizer...");
        let tokenizer_path = std::path::PathBuf::from(model_dir).join("tokenizer/tokenizer.json");
        let tokenizer = if tokenizer_path.exists() {
            Tokenizer::from_file(tokenizer_path).map_err(|e| anyhow::anyhow!(e))?
        } else {
            let api = Api::new()?;
            let repo = api.model("openai/clip-vit-base-patch32".to_string());
            let tok_path = repo.get("tokenizer.json")?;
            Tokenizer::from_file(tok_path).map_err(|e| anyhow::anyhow!(e))?
        };

        let clip_weights_path = std::path::PathBuf::from(model_dir).join(if use_f16 {
            "text_encoder/model.fp16.safetensors"
        } else {
            "text_encoder/model.safetensors"
        });

        Ok(Self {
            device,
            dtype,
            sd_config,
            vae,
            unet,
            tokenizer,
            clip_weights_path,
            n_steps,
            _use_f16: use_f16,
        })
    }

    pub fn run(
        &mut self,
        prompt: &str,
        img_tensor: Option<&[f32]>,
        width: usize,
        height: usize,
        strength: f64,
        progress: Option<&dyn Fn(usize, usize, f32)>,
    ) -> Result<Vec<f32>> {
        let guide_scale = 7.5;
        let vae_scale = 0.18215;

        if let Some(cb) = progress {
            cb(0, self.n_steps, 0.0);
        }

        let text_embeddings = self.text_embeddings(prompt, guide_scale > 1.0)?;

        let mut scheduler = self.sd_config.build_scheduler(self.n_steps)?;

        let init_latent_dist = match img_tensor {
            Some(img) => {
                let img = Tensor::from_vec(img.to_vec(), (1, 3, height, width), &self.device)?
                    .to_dtype(self.dtype)?;
                Some(self.vae.encode(&img)?)
            }
            None => None,
        };

        let t_start = if img_tensor.is_some() {
            self.n_steps - (self.n_steps as f64 * strength) as usize
        } else {
            0
        };

        let latents = match &init_latent_dist {
            Some(init_latent_dist) => {
                let latents = (init_latent_dist.sample()? * vae_scale)?.to_device(&self.device)?;
                if t_start < scheduler.timesteps().len() {
                    let noise = latents.randn_like(0f64, 1f64)?;
                    scheduler.add_noise(&latents, noise, scheduler.timesteps()[t_start])?
                } else {
                    latents
                }
            }
            None => {
                let latents = Tensor::randn(
                    0f32,
                    1f32,
                    (1, 4, height / 8, width / 8),
                    &self.device,
                )?;
                (latents * scheduler.init_noise_sigma())?
            }
        };
        let mut latents = latents.to_dtype(self.dtype)?;

        let timesteps = scheduler.timesteps().to_vec();
        let active_steps = timesteps.len() - t_start;
        for (step_idx, &timestep) in timesteps.iter().enumerate() {
            if step_idx < t_start {
                continue;
            }

            if let Some(cb) = progress {
                let done = step_idx - t_start;
                let elapsed = 0.0;
                cb(done, active_steps, elapsed);
            }

            let start = std::time::Instant::now();

            let latent_model_input = if guide_scale > 1.0 {
                Tensor::cat(&[&latents, &latents], 0)?
            } else {
                latents.clone()
            };
            let latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)?;

            let noise_pred = self.unet.forward(&latent_model_input, timestep as f64, &text_embeddings)?;

            let noise_pred = if guide_scale > 1.0 {
                let noise_pred = noise_pred.chunk(2, 0)?;
                let (uncond, text) = (&noise_pred[0], &noise_pred[1]);
                (uncond + ((text - uncond)? * guide_scale)?)?
            } else {
                noise_pred
            };

            latents = scheduler.step(&noise_pred, timestep, &latents)?;

            let dt = start.elapsed().as_secs_f32();
            if let Some(cb) = progress {
                let done = step_idx - t_start + 1;
                cb(done, active_steps, dt);
            }
        }

        let images = self.vae.decode(&(latents / vae_scale)?)?;
        let images = ((images / 2.)? + 0.5)?.to_device(&Device::Cpu)?;
        let images = (images.clamp(0f32, 1.)? * 255.)?.to_dtype(DType::U8)?;

        let image = images.i(0)?.flatten_all()?;
        let raw: Vec<u8> = image.to_vec1()?;
        Ok(raw.iter().map(|&v| v as f32 / 255.0).collect())
    }

    fn text_embeddings(&self, prompt: &str, use_guide_scale: bool) -> Result<Tensor> {
        let pad_id = match &self.sd_config.clip.pad_with {
            Some(padding) => *self.tokenizer.get_vocab(true).get(padding.as_str()).unwrap(),
            None => *self.tokenizer.get_vocab(true).get("!").unwrap(),
        };

        let mut tokens = self.tokenizer.encode(prompt, true).map_err(|e| anyhow::anyhow!(e))?
            .get_ids().to_vec();
        while tokens.len() < self.sd_config.clip.max_position_embeddings {
            tokens.push(pad_id);
        }
        let tokens = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;

        let text_model = stable_diffusion::build_clip_transformer(
            &self.sd_config.clip,
            &self.clip_weights_path,
            &self.device,
            DType::F32,
        )?;
        let text_embeddings = text_model.forward(&tokens)?;

        let text_embeddings = if use_guide_scale {
            let uncond_tokens = self.tokenizer.encode("", true).map_err(|e| anyhow::anyhow!(e))?
                .get_ids().to_vec();
            let mut uncond_tokens = uncond_tokens;
            while uncond_tokens.len() < self.sd_config.clip.max_position_embeddings {
                uncond_tokens.push(pad_id);
            }
            let uncond_tokens = Tensor::new(uncond_tokens.as_slice(), &self.device)?.unsqueeze(0)?;
            let uncond_embeddings = text_model.forward(&uncond_tokens)?;
            Tensor::cat(&[uncond_embeddings, text_embeddings], 0)?.to_dtype(self.dtype)?
        } else {
            text_embeddings.to_dtype(self.dtype)?
        };
        Ok(text_embeddings)
    }
}
