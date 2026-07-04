mod image_utils;
mod model;

use anyhow::Result;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "anime-style", about = "Anime-style image generator using local diffusion model")]
struct Cli {
    #[arg(short, long)]
    prompt: String,

    #[arg(short = 'i', long)]
    image: Option<String>,

    #[arg(short, long, default_value = "output.png")]
    output: String,

    #[arg(short, long, default_value_t = 512)]
    size: u32,

    #[arg(short = 't', long, default_value_t = 20)]
    steps: u32,

    #[arg(long, default_value = "./models/stable-diffusion-v1-5")]
    model_dir: String,

    #[arg(long)]
    f16: bool,

    #[arg(long, default_value_t = 0.75)]
    strength: f64,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    println!("Loading model from: {}", cli.model_dir);
    let mut anime_model = model::AnimeModel::new(
        &cli.model_dir,
        cli.steps as usize,
        cli.f16,
        cli.size as usize,
        cli.size as usize,
    )?;

    let img_tensor = match &cli.image {
        Some(path) => {
            println!("Loading input image: {path}");
            let img = image_utils::load(path)?;
            let resized = image_utils::resize(&img, cli.size);
            Some(image_utils::rgb_to_tensor(&resized)?)
        }
        None => None,
    };

    let w = cli.size as usize;
    let h = cli.size as usize;

    println!("Running inference ({} steps)...", cli.steps);
    let result = anime_model.run(&cli.prompt, img_tensor.as_deref(), w, h, cli.strength, None)?;

    let result_img = image_utils::tensor_to_rgb(&result, w, h);
    result_img.save(&cli.output)?;
    println!("Saved to: {}", cli.output);

    Ok(())
}
