use clap::Parser;
use image::{ImageReader, RgbImage};
use minifb::{Key, Window, WindowOptions};
use rand::Rng;
use std::path::{PathBuf, Path};

#[derive(Parser)]
struct Args {
    target: PathBuf,

    #[clap(short, long, default_value = "4096")]
    iterations: usize,
}

fn main() {
    let args = Args::parse();

    let target = ImageReader::open(&args.target)
        .expect("couldn't load given image")
        .decode()
        .expect("couldn't decode given image")
        .into_rgb8();

    let target = Image::from(target);
    let width = target.width;
    let height = target.height;

    let mut approx = Image::from(RgbImage::new(width, height));

    let mut canvas = vec![0; (width * height) as usize];

    let mut window = Window::new(
        "brushez",
        width as usize,
        height as usize,
        WindowOptions::default(),
    )
        .unwrap();

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let mut improvements_this_frame = 0;

        for _ in 0..args.iterations {
            if tick(&target, &mut approx) {
                improvements_this_frame += 1;
                // Update display more frequently for smoother brush strokes
                if improvements_this_frame % 10 == 0 {
                    approx.encode(&mut canvas);
                    window
                        .update_with_buffer(&canvas, width as usize, height as usize)
                        .unwrap();
                }
            }
        }

        // Final update for this frame
        approx.encode(&mut canvas);
        window
            .update_with_buffer(&canvas, width as usize, height as usize)
            .unwrap();
    }

    // Save the final image when window closes
    if !window.is_open() || window.is_key_down(Key::Escape) {
        // Create the output filename
        let input_path = Path::new(&args.target);
        let input_stem = input_path.file_stem().unwrap().to_str().unwrap();
        let output_filename = format!("generated_images/{}_brushez.jpg", input_stem);

        // Convert the current state to an image
        let mut output_image = RgbImage::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let [r, g, b] = approx.color_at([x, y]);
                output_image.put_pixel(x, y, image::Rgb([r, g, b]));
            }
        }

        // Ensure the directory exists
        std::fs::create_dir_all("generated_images").expect("Failed to create output directory");

        // Save the image
        output_image.save(&output_filename).expect("Failed to save output image");
        println!("Saved final image to: {}", output_filename);
    }
}

fn calculate_weighted_stroke_color(target: &Image, stroke_points: &[[isize; 2]]) -> [u8; 3] {
    // Calculate average color along the stroke for better approximation
    let mut valid_points = 0;
    let mut total_r = 0u32;
    let mut total_g = 0u32;
    let mut total_b = 0u32;

    for [x, y] in stroke_points.iter() {
        if *x >= 0 && *y >= 0 && *x < target.width as isize && *y < target.height as isize {
            let [r, g, b] = target.color_at([*x as u32, *y as u32]);
            total_r += r as u32;
            total_g += g as u32;
            total_b += b as u32;
            valid_points += 1;
        }
    }

    if valid_points == 0 {
        // If no valid points, sample from the center of the image as fallback
        let center_x = (target.width / 2) as u32;
        let center_y = (target.height / 2) as u32;
        return target.color_at([center_x, center_y]);
    }

    [
        (total_r / valid_points) as u8,
        (total_g / valid_points) as u8,
        (total_b / valid_points) as u8,
    ]
}

fn tick(target: &Image, approx: &mut Image) -> bool {
    let mut rng = rand::thread_rng();
    
    // Random start point
    let start_x = rng.gen_range(0..target.width) as isize;
    let start_y = rng.gen_range(0..target.height) as isize;
    
    // Random angle (0 to 2π)
    let angle = rng.gen_range(0.0..2.0 * std::f32::consts::PI);
    
    // Random length (reasonable stroke length - smaller than circles for more control)
    let max_length = (target.width.min(target.height) / 6) as isize; // Reduced from /3 to /6
    let length = rng.gen_range(3..=max_length.max(3) as usize) as isize; // Smaller minimum length
    
    // Random brush width (smaller range for more precision)
    let width = rng.gen_range(1..=4) as isize; // Reduced from 1..=8 to 1..=4
    
    // Calculate end point
    let end_x = start_x + (length as f32 * angle.cos()) as isize;
    let end_y = start_y + (length as f32 * angle.sin()) as isize;

    // Generate brush stroke points
    let stroke_points = generate_brush_stroke_points(start_x, start_y, angle, length, width);

    // Calculate weighted average color
    let color = calculate_weighted_stroke_color(target, &stroke_points);

    // Generate all points that would be affected by the brush stroke
    let changes = stroke_points
        .into_iter()
        .filter(|&[x, y]| {
            x >= 0 &&
                y >= 0 &&
                x < target.width as isize &&
                y < target.height as isize
        })
        .map(|[x, y]| ([x as u32, y as u32], color));

    // Check if drawing this brush stroke would improve the approximation
    let loss_delta = Image::loss_delta(target, approx, changes.clone());

    if loss_delta >= 0.0 {
        return false;
    }

    // Apply the changes if the brush stroke improves the approximation
    approx.apply(changes);
    true
}

fn generate_line_points(x0: isize, y0: isize, x1: isize, y1: isize) -> Vec<[isize; 2]> {
    let mut points = Vec::new();
    
    let dx = (x1 - x0).abs();
    let dy = (y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx - dy;
    
    let mut x = x0;
    let mut y = y0;
    
    loop {
        points.push([x, y]);
        
        if x == x1 && y == y1 {
            break;
        }
        
        let e2 = 2 * err;
        if e2 > -dy {
            err -= dy;
            x += sx;
        }
        if e2 < dx {
            err += dx;
            y += sy;
        }
    }
    
    points
}

fn generate_brush_stroke_points(start_x: isize, start_y: isize, angle: f32, length: isize, width: isize) -> Vec<[isize; 2]> {
    let mut points = Vec::new();
    
    // Calculate end point from start + angle + length
    let end_x = start_x + (length as f32 * angle.cos()) as isize;
    let end_y = start_y + (length as f32 * angle.sin()) as isize;
    
    // Generate points along the stroke centerline (using Bresenham's line algorithm)
    let centerline_points = generate_line_points(start_x, start_y, end_x, end_y);
    
    // For each centerline point, add points perpendicular to create brush width
    let perp_angle = angle + std::f32::consts::PI / 2.0;
    
    for [cx, cy] in centerline_points {
        for w in -(width/2)..=(width/2) {
            let px = cx + (w as f32 * perp_angle.cos()) as isize;
            let py = cy + (w as f32 * perp_angle.sin()) as isize;
            points.push([px, py]);
        }
    }
    
    points
}

type Point = [u32; 2];
type Color = [u8; 3];

struct Image {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
}

impl Image {
    fn loss_delta(
        target: &Self,
        approx: &Self,
        changes: impl IntoIterator<Item = (Point, Color)>,
    ) -> f32 {
        changes
            .into_iter()
            .map(|(pos, new_col)| {
                let target_color = target.color_at(pos);
                let approx_color = approx.color_at(pos);

                let loss_without_changes = Self::pixel_loss(target_color, approx_color);
                let loss_with_changes = Self::pixel_loss(target_color, new_col);

                loss_with_changes - loss_without_changes
            })
            .sum()
    }

    fn pixel_loss(a: Color, b: Color) -> f32 {
        a.into_iter()
            .zip(b)
            .map(|(a, b)| (a as f32 - b as f32).powi(2))
            .sum()
    }

    fn apply(&mut self, changes: impl IntoIterator<Item = (Point, Color)>) {
        for (pos, col) in changes {
            *self.color_at_mut(pos) = col;
        }
    }

    fn encode(&self, buf: &mut [u32]) {
        let mut buf = buf.iter_mut();

        for y in 0..self.height {
            for x in 0..self.width {
                let [r, g, b] = self.color_at([x, y]);
                *buf.next().unwrap() = u32::from_be_bytes([0, r, g, b]);
            }
        }
    }

    fn color_at(&self, point: Point) -> Color {
        let offset = (point[1] * self.width + point[0]) as usize * 3;
        let color = &self.pixels[offset..][..3];
        color.try_into().unwrap()
    }

    fn color_at_mut(&mut self, [x, y]: [u32; 2]) -> &mut Color {
        let offset = (y * self.width + x) as usize * 3;
        let color = &mut self.pixels[offset..][..3];
        color.try_into().unwrap()
    }
}

impl From<RgbImage> for Image {
    fn from(img: RgbImage) -> Self {
        let width = img.width();
        let height = img.height();
        let pixels = img.pixels().flat_map(|pixel| pixel.0).collect();

        Self {
            width,
            height,
            pixels,
        }
    }
}