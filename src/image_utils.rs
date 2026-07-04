use anyhow::{Context, Result};
use image::{DynamicImage, RgbImage};

pub fn load(path: &str) -> Result<DynamicImage> {
    image::open(path).with_context(|| format!("Failed to open image: {path}"))
}

pub fn resize(img: &DynamicImage, size: u32) -> RgbImage {
    img.resize_exact(size, size, image::imageops::FilterType::Lanczos3)
        .to_rgb8()
}

pub fn rgb_to_tensor(img: &RgbImage) -> Result<Vec<f32>> {
    let (w, h) = img.dimensions();
    let pixels = img.as_raw();
    let mut tensor = Vec::with_capacity(3 * (w * h) as usize);
    // CHW format: channel first
    for c in 0..3 {
        for y in 0..h {
            for x in 0..w {
                let idx = ((y * w + x) * 3 + c) as usize;
                tensor.push(pixels[idx] as f32 / 255.0);
            }
        }
    }
    Ok(tensor)
}

pub fn tensor_to_rgb(tensor: &[f32], width: usize, height: usize) -> RgbImage {
    let mut img = RgbImage::new(width as u32, height as u32);
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let r = (tensor[idx].clamp(0.0, 1.0) * 255.0) as u8;
            let g = (tensor[width * height + idx].clamp(0.0, 1.0) * 255.0) as u8;
            let b = (tensor[2 * width * height + idx].clamp(0.0, 1.0) * 255.0) as u8;
            img.put_pixel(x as u32, y as u32, image::Rgb([r, g, b]));
        }
    }
    img
}
