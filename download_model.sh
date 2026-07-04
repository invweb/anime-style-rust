#!/bin/bash
set -e

MODEL_DIR="./models/stable-diffusion-v1-5"
mkdir -p "$MODEL_DIR"

echo "Downloading Stable Diffusion v1.5 model files..."

# UNet fp16 (~1.7GB)
echo "Downloading UNet..."
mkdir -p "$MODEL_DIR/unet"
curl -L "https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/unet/diffusion_pytorch_model.fp16.safetensors" \
  -o "$MODEL_DIR/unet/diffusion_pytorch_model.fp16.safetensors"

# VAE fp16 (~335MB)
echo "Downloading VAE..."
mkdir -p "$MODEL_DIR/vae"
curl -L "https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/vae/diffusion_pytorch_model.fp16.safetensors" \
  -o "$MODEL_DIR/vae/diffusion_pytorch_model.fp16.safetensors"

# Text Encoder fp16 (~492MB)
echo "Downloading Text Encoder..."
mkdir -p "$MODEL_DIR/text_encoder"
curl -L "https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/text_encoder/model.fp16.safetensors" \
  -o "$MODEL_DIR/text_encoder/model.fp16.safetensors"

# Tokenizer (~1MB)
echo "Downloading Tokenizer..."
mkdir -p "$MODEL_DIR/tokenizer"
curl -L "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/tokenizer.json" \
  -o "$MODEL_DIR/tokenizer/tokenizer.json"

echo ""
echo "Done! Model files downloaded to $MODEL_DIR"
echo "Total size: ~2.5GB"
echo ""
echo "Usage:"
echo "  Text-to-image:"
echo "    cargo run --bin anime-cli --release -- --prompt 'anime girl, cherry blossoms' --model-dir $MODEL_DIR --f16"
echo ""
echo "  Image-to-image:"
echo "    cargo run --bin anime-cli --release -- --prompt 'anime style' --image photo.jpg --model-dir $MODEL_DIR --f16"
