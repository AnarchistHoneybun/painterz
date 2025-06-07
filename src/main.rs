use clap::Parser;
use image::{ImageReader, RgbImage};
use minifb::{Key, Window, WindowOptions};
use rand::Rng;
use std::path::{Path, PathBuf};

#[derive(Parser)]
struct Args {
    target: PathBuf,

    #[clap(short, long, default_value = "4096")]
    iterations: usize,
}

// ============================================================================
// HIERARCHICAL PAINTING SYSTEM
// ============================================================================

#[derive(Clone, Debug)]
struct StageParameters {
    min_brush_size: isize,
    max_brush_size: isize,
    stroke_count: usize,
    color_accuracy: f32,    // How closely to match target colors (0.0 = loose, 1.0 = exact)
    edge_emphasis: f32,     // How much to follow edges vs random placement (0.0 = random, 1.0 = edge-only)
    update_frequency: usize, // How often to update display during this stage
}

struct ImagePyramid {
    levels: Vec<Image>,
    stage_params: Vec<StageParameters>,
}

impl ImagePyramid {
    fn new(original: &Image) -> Self {
        let levels = vec![
            // Stage 0: Composition - very blurred, coarse details
            downsample_and_blur(original, 0.25, 4.0),
            
            // Stage 1: Form definition - medium blur, basic shapes
            downsample_and_blur(original, 0.5, 2.0),
            
            // Stage 2: Surface details - slight blur, most details visible
            apply_blur(original, 0.8),
            
            // Stage 3: Finishing touches - original resolution, edge-enhanced
            enhance_edges(original, 1.2),
        ];
        
        let stage_params = vec![
            StageParameters {
                min_brush_size: 150,
                max_brush_size: 300,
                stroke_count: 800,
                color_accuracy: 0.7,
                edge_emphasis: 0.2,
                update_frequency: 5,
            },
            StageParameters {
                min_brush_size: 50,
                max_brush_size: 120,
                stroke_count: 1600,
                color_accuracy: 0.85,
                edge_emphasis: 0.5,
                update_frequency: 10,
            },
            StageParameters {
                min_brush_size: 20,
                max_brush_size: 40,
                stroke_count: 3200,
                color_accuracy: 0.95,
                edge_emphasis: 0.8,
                update_frequency: 20,
            },
            StageParameters {
                min_brush_size: 5,
                max_brush_size: 10,
                stroke_count: 6400,
                color_accuracy: 0.98,
                edge_emphasis: 0.9,
                update_frequency: 15,
            },
        ];
        
        Self { levels, stage_params }
    }
}

fn downsample_and_blur(image: &Image, scale: f32, blur_radius: f32) -> Image {
    // First downsample
    let new_width = (image.width as f32 * scale) as u32;
    let new_height = (image.height as f32 * scale) as u32;
    let downsampled = downsample_image(image, new_width, new_height);
    
    // Then blur
    let blurred = apply_blur(&downsampled, blur_radius);
    
    // Scale back up to original size for consistent processing
    upsample_image(&blurred, image.width, image.height)
}

fn downsample_image(image: &Image, new_width: u32, new_height: u32) -> Image {
    let mut pixels = vec![0u8; (new_width * new_height * 3) as usize];
    
    let x_ratio = image.width as f32 / new_width as f32;
    let y_ratio = image.height as f32 / new_height as f32;
    
    for y in 0..new_height {
        for x in 0..new_width {
            // Sample with basic averaging for anti-aliasing
            let src_x_base = (x as f32 * x_ratio) as u32;
            let src_y_base = (y as f32 * y_ratio) as u32;
            
            let mut total_r = 0u32;
            let mut total_g = 0u32;
            let mut total_b = 0u32;
            let mut count = 0u32;
            
            // Average a small area for better downsampling
            for dy in 0..2 {
                for dx in 0..2 {
                    let src_x = (src_x_base + dx).min(image.width - 1);
                    let src_y = (src_y_base + dy).min(image.height - 1);
                    
                    let [r, g, b] = image.color_at([src_x, src_y]);
                    total_r += r as u32;
                    total_g += g as u32;
                    total_b += b as u32;
                    count += 1;
                }
            }
            
            let dst_idx = ((y * new_width + x) * 3) as usize;
            pixels[dst_idx] = (total_r / count) as u8;
            pixels[dst_idx + 1] = (total_g / count) as u8;
            pixels[dst_idx + 2] = (total_b / count) as u8;
        }
    }
    
    Image {
        width: new_width,
        height: new_height,
        pixels,
    }
}

fn upsample_image(image: &Image, new_width: u32, new_height: u32) -> Image {
    let mut pixels = vec![0u8; (new_width * new_height * 3) as usize];
    
    let x_ratio = image.width as f32 / new_width as f32;
    let y_ratio = image.height as f32 / new_height as f32;
    
    for y in 0..new_height {
        for x in 0..new_width {
            let src_x = ((x as f32 * x_ratio) as u32).min(image.width - 1);
            let src_y = ((y as f32 * y_ratio) as u32).min(image.height - 1);
            
            let color = image.color_at([src_x, src_y]);
            let dst_idx = ((y * new_width + x) * 3) as usize;
            pixels[dst_idx] = color[0];
            pixels[dst_idx + 1] = color[1];
            pixels[dst_idx + 2] = color[2];
        }
    }
    
    Image {
        width: new_width,
        height: new_height,
        pixels,
    }
}

fn apply_blur(image: &Image, radius: f32) -> Image {
    if radius < 0.1 {
        return image.clone();
    }
    
    let mut blurred_pixels = vec![0u8; image.pixels.len()];
    let kernel_size = (radius * 2.0) as usize + 1;
    let half_kernel = kernel_size / 2;
    
    // Simple box blur for performance
    for y in 0..image.height {
        for x in 0..image.width {
            let mut total_r = 0u32;
            let mut total_g = 0u32;
            let mut total_b = 0u32;
            let mut count = 0u32;
            
            // Average surrounding pixels
            for ky in 0..kernel_size {
                for kx in 0..kernel_size {
                    let sample_x = (x as isize + kx as isize - half_kernel as isize)
                        .max(0)
                        .min(image.width as isize - 1) as u32;
                    let sample_y = (y as isize + ky as isize - half_kernel as isize)
                        .max(0)
                        .min(image.height as isize - 1) as u32;
                    
                    let [r, g, b] = image.color_at([sample_x, sample_y]);
                    total_r += r as u32;
                    total_g += g as u32;
                    total_b += b as u32;
                    count += 1;
                }
            }
            
            let dst_idx = ((y * image.width + x) * 3) as usize;
            blurred_pixels[dst_idx] = (total_r / count) as u8;
            blurred_pixels[dst_idx + 1] = (total_g / count) as u8;
            blurred_pixels[dst_idx + 2] = (total_b / count) as u8;
        }
    }
    
    Image {
        width: image.width,
        height: image.height,
        pixels: blurred_pixels,
    }
}

fn enhance_edges(image: &Image, strength: f32) -> Image {
    let mut enhanced_pixels = image.pixels.clone();
    
    // Simple edge enhancement using unsharp mask
    for y in 1..(image.height - 1) {
        for x in 1..(image.width - 1) {
            let center = image.color_at([x, y]);
            
            // Calculate average of surrounding pixels
            let mut avg_r = 0f32;
            let mut avg_g = 0f32;
            let mut avg_b = 0f32;
            let mut count = 0f32;
            
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    if dx == 0 && dy == 0 { continue; }
                    let [r, g, b] = image.color_at([(x as i32 + dx) as u32, (y as i32 + dy) as u32]);
                    avg_r += r as f32;
                    avg_g += g as f32;
                    avg_b += b as f32;
                    count += 1.0;
                }
            }
            
            avg_r /= count;
            avg_g /= count;
            avg_b /= count;
            
            // Enhance difference from average
            let enhanced_r = (center[0] as f32 + (center[0] as f32 - avg_r) * strength).clamp(0.0, 255.0) as u8;
            let enhanced_g = (center[1] as f32 + (center[1] as f32 - avg_g) * strength).clamp(0.0, 255.0) as u8;
            let enhanced_b = (center[2] as f32 + (center[2] as f32 - avg_b) * strength).clamp(0.0, 255.0) as u8;
            
            let idx = ((y * image.width + x) * 3) as usize;
            enhanced_pixels[idx] = enhanced_r;
            enhanced_pixels[idx + 1] = enhanced_g;
            enhanced_pixels[idx + 2] = enhanced_b;
        }
    }
    
    Image {
        width: image.width,
        height: image.height,
        pixels: enhanced_pixels,
    }
}

// ============================================================================
// PAINT MIXING SYSTEM (unchanged from previous version)
// ============================================================================

#[derive(Clone, Copy, Debug)]
struct PaintLayer {
    color: [u8; 3],
    thickness: f32,    // Paint thickness (0.0 to 1.0)
    wetness: f32,      // How wet the paint is (0.0 to 1.0)
    age: u32,          // How long since applied (in iterations)
}

struct PaintSurface {
    width: u32,
    height: u32,
    layers: Vec<Vec<PaintLayer>>, // Stack of paint layers per pixel
    max_layers: usize,
}

impl PaintSurface {
    fn new(width: u32, height: u32) -> Self {
        let pixel_count = (width * height) as usize;
        let mut layers = Vec::with_capacity(pixel_count);
        
        // Initialize with white canvas
        for _ in 0..pixel_count {
            layers.push(vec![PaintLayer {
                color: [240, 240, 240], // Slightly off-white to reduce harsh contrast
                thickness: 1.0,
                wetness: 0.0,  // Canvas starts dry
                age: 1000,     // Very old (dry)
            }]);
        }
        
        Self {
            width,
            height,
            layers,
            max_layers: 5, // Limit paint buildup
        }
    }
    
    fn apply_stroke(&mut self, stroke: &BrushStroke, new_color: [u8; 3], paint_amount: f32) {
        for &[x, y] in &stroke.points {
            if x >= 0 && y >= 0 && x < self.width as isize && y < self.height as isize {
                self.apply_paint_to_pixel(x as u32, y as u32, new_color, paint_amount);
            }
        }
        
        // Age all paint slightly (but not every stroke to maintain performance)
        if rand::random::<f32>() < 0.1 { // Only 10% of the time
            self.age_paint();
        }
    }
    
    fn apply_paint_to_pixel(&mut self, x: u32, y: u32, new_color: [u8; 3], paint_amount: f32) {
        let idx = (y * self.width + x) as usize;
        let pixel_layers = &mut self.layers[idx];
        
        // Check if we're painting on wet paint (more conservative threshold)
        let top_layer = pixel_layers.last().unwrap();
        let is_wet_on_wet = top_layer.wetness > 0.5 && top_layer.age < 8; // Higher wetness threshold, shorter age
        
        if is_wet_on_wet {
            // Mix with existing wet paint
            let mixed_color = mix_colors(
                top_layer.color,
                top_layer.thickness,
                new_color,
                paint_amount,
                top_layer.wetness,
            );
            
            // Update the top layer with mixed result
            if let Some(last_layer) = pixel_layers.last_mut() {
                last_layer.color = mixed_color;
                last_layer.thickness = (last_layer.thickness + paint_amount * 0.5).min(1.0);
                last_layer.wetness = (last_layer.wetness + paint_amount * 0.8).min(1.0);
                last_layer.age = 0; // Reset age since we just painted
            }
        } else {
            // Apply as new layer
            let new_layer = PaintLayer {
                color: new_color,
                thickness: paint_amount,
                wetness: paint_amount * 0.9, // New paint starts wet
                age: 0,
            };
            
            pixel_layers.push(new_layer);
            
            // Limit layer buildup
            if pixel_layers.len() > self.max_layers {
                pixel_layers.remove(0);
            }
        }
    }
    
    fn age_paint(&mut self) {
        for pixel_layers in &mut self.layers {
            for layer in pixel_layers.iter_mut() {
                layer.age += 1;
                
                // Paint dries over time
                if layer.age > 5 {
                    layer.wetness *= 0.95; // Gradually dry out
                }
                
                // Very old paint is completely dry
                if layer.age > 20 {
                    layer.wetness = 0.0;
                }
            }
        }
    }
    
    fn get_visible_color(&self, x: u32, y: u32) -> [u8; 3] {
        let idx = (y * self.width + x) as usize;
        let pixel_layers = &self.layers[idx];
        
        // Composite layers from bottom to top
        let mut result_color = [240u8, 240u8, 240u8]; // Match the off-white canvas start
        
        for layer in pixel_layers {
            if layer.thickness > 0.05 { // Slightly higher threshold for visibility
                result_color = self.composite_over(result_color, layer.color, layer.thickness);
            }
        }
        
        result_color
    }
    
    fn composite_over(&self, base: [u8; 3], overlay: [u8; 3], thickness: f32) -> [u8; 3] {
        // Convert thickness to opacity with high minimum opacity for strong colors
        let opacity = (thickness * 0.3 + 0.7).clamp(0.7, 1.0); // Minimum 70% opacity, up to 100%
        let inv_opacity = 1.0 - opacity;
        
        [
            (base[0] as f32 * inv_opacity + overlay[0] as f32 * opacity) as u8,
            (base[1] as f32 * inv_opacity + overlay[1] as f32 * opacity) as u8,
            (base[2] as f32 * inv_opacity + overlay[2] as f32 * opacity) as u8,
        ]
    }
    
    fn to_image(&self) -> Image {
        let mut pixels = vec![0u8; (self.width * self.height * 3) as usize];
        
        for y in 0..self.height {
            for x in 0..self.width {
                let color = self.get_visible_color(x, y);
                let idx = ((y * self.width + x) * 3) as usize;
                pixels[idx] = color[0];
                pixels[idx + 1] = color[1];
                pixels[idx + 2] = color[2];
            }
        }
        
        Image {
            width: self.width,
            height: self.height,
            pixels,
        }
    }
    
    // Method to calculate loss delta for stroke evaluation
    fn loss_delta(
        &self,
        target: &Image,
        changes: impl IntoIterator<Item = ([u32; 2], [u8; 3], f32)>, // position, color, paint_amount
    ) -> f32 {
        changes
            .into_iter()
            .map(|(pos, new_col, paint_amount)| {
                let target_color = target.color_at(pos);
                let current_color = self.get_visible_color(pos[0], pos[1]);
                
                // Simulate what the color would be after applying this paint
                let simulated_color = if paint_amount > 0.01 {
                    // Check if there would be wet-on-wet mixing
                    let idx = (pos[1] * self.width + pos[0]) as usize;
                    let top_layer = self.layers[idx].last().unwrap();
                    let is_wet_on_wet = top_layer.wetness > 0.5 && top_layer.age < 8; // Match the updated threshold
                    
                    if is_wet_on_wet {
                        mix_colors(
                            current_color,
                            top_layer.thickness,
                            new_col,
                            paint_amount,
                            top_layer.wetness,
                        )
                    } else {
                        // Convert thickness to opacity with high minimum for consistency
                        let opacity = (paint_amount * 0.3 + 0.7).clamp(0.7, 1.0);
                        let inv_opacity = 1.0 - opacity;
                        [
                            (current_color[0] as f32 * inv_opacity + new_col[0] as f32 * opacity) as u8,
                            (current_color[1] as f32 * inv_opacity + new_col[1] as f32 * opacity) as u8,
                            (current_color[2] as f32 * inv_opacity + new_col[2] as f32 * opacity) as u8,
                        ]
                    }
                } else {
                    current_color
                };

                let loss_without_changes = Self::pixel_loss(target_color, current_color);
                let loss_with_changes = Self::pixel_loss(target_color, simulated_color);

                loss_with_changes - loss_without_changes
            })
            .sum()
    }
    
    fn pixel_loss(a: [u8; 3], b: [u8; 3]) -> f32 {
        a.into_iter()
            .zip(b)
            .map(|(a, b)| (a as f32 - b as f32).powi(2))
            .sum()
    }
    
    fn encode(&self, buf: &mut [u32]) {
        let mut buf = buf.iter_mut();

        for y in 0..self.height {
            for x in 0..self.width {
                let [r, g, b] = self.get_visible_color(x, y);
                *buf.next().unwrap() = u32::from_be_bytes([0, r, g, b]);
            }
        }
    }
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

    // Create hierarchical image pyramid
    println!("Creating image pyramid for hierarchical painting...");
    let pyramid = ImagePyramid::new(&target);

    // Initialize paint surface
    let mut paint_surface = PaintSurface::new(width, height);

    let mut canvas = vec![0; (width * height) as usize];
    let mut total_stroke_count = 0;

    let mut window = Window::new(
        "brushez - hierarchical painting",
        width as usize,
        height as usize,
        WindowOptions::default(),
    )
    .unwrap();

    // Paint through each stage of the hierarchy
    for (stage_idx, (target_level, stage_params)) in pyramid.levels.iter().zip(&pyramid.stage_params).enumerate() {
        println!("Stage {}: {} strokes with brushes {}-{}px", 
                 stage_idx, stage_params.stroke_count, 
                 stage_params.min_brush_size, stage_params.max_brush_size);
        
        let mut stage_improvements = 0;
        let mut stroke_attempts = 0;
        
        for _stroke_num in 0..stage_params.stroke_count {
            if window.is_key_down(Key::Escape) {
                break;
            }
            
            if tick_hierarchical(target_level, &mut paint_surface, stage_params, total_stroke_count) {
                total_stroke_count += 1;
                stage_improvements += 1;
                
                // Update display based on stage frequency
                if stage_improvements % stage_params.update_frequency == 0 {
                    paint_surface.encode(&mut canvas);
                    window
                        .update_with_buffer(&canvas, width as usize, height as usize)
                        .unwrap();
                }
            }
            stroke_attempts += 1;
        }
        
        println!("Stage {} completed: {}/{} successful strokes", 
                 stage_idx, stage_improvements, stroke_attempts);
        
        // Final update for this stage
        paint_surface.encode(&mut canvas);
        window
            .update_with_buffer(&canvas, width as usize, height as usize)
            .unwrap();
            
        if window.is_key_down(Key::Escape) {
            break;
        }
    }

    // Keep window open until escape is pressed
    while window.is_open() && !window.is_key_down(Key::Escape) {
        window
            .update_with_buffer(&canvas, width as usize, height as usize)
            .unwrap();
    }

    // Save the final image when window closes
    if !window.is_open() || window.is_key_down(Key::Escape) {
        // Create the output filename
        let input_path = Path::new(&args.target);
        let input_stem = input_path.file_stem().unwrap().to_str().unwrap();
        let output_filename = format!("generated_images/{}_hierarchical.jpg", input_stem);

        // Convert the current state to an image
        let final_image = paint_surface.to_image();
        let mut output_image = RgbImage::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let [r, g, b] = final_image.color_at([x, y]);
                output_image.put_pixel(x, y, image::Rgb([r, g, b]));
            }
        }

        // Ensure the directory exists
        std::fs::create_dir_all("generated_images").expect("Failed to create output directory");

        // Save the image
        output_image
            .save(&output_filename)
            .expect("Failed to save output image");
        println!("Saved final hierarchical painting to: {}", output_filename);
    }
}

fn tick_hierarchical(
    target: &Image, 
    paint_surface: &mut PaintSurface, 
    stage_params: &StageParameters,
    stroke_count: usize
) -> bool {
    let mut rng = rand::thread_rng();

    // Replace this in the tick_hierarchical function:
let length = rng.gen_range(
    (stage_params.min_brush_size / 2).max(3) as i32..=stage_params.max_brush_size as i32
) as isize;

let width = rng.gen_range(
    (stage_params.min_brush_size / 3).max(1) as i32..=(stage_params.max_brush_size / 2).max(2) as i32
) as isize;

    // Stroke placement based on stage emphasis
    let (start_x, start_y) = if rng.random::<f32>() < stage_params.edge_emphasis {
        // Edge-aware placement (simplified for now - could use edge detection)
        select_high_contrast_point(target, &mut rng)
    } else {
        // Random placement
        (
            rng.gen_range(0..target.width) as isize,
            rng.gen_range(0..target.height) as isize,
        )
    };

    // Random angle (could be made edge-aware in future)
    let angle = rng.gen_range(0.0..2.0 * std::f32::consts::PI);

    // Generate interval spline brush stroke
    let stroke = generate_interval_spline_stroke(start_x, start_y, angle, length, width, &mut rng);
    
    // Calculate color with stage-appropriate accuracy
    let target_color = calculate_stage_appropriate_color(target, &stroke, stage_params);

    // Determine paint amount based on brush size and stage
    let base_paint_amount = (width as f32 / stage_params.max_brush_size as f32) * 0.8 + 0.6;
    let paint_amount = base_paint_amount.clamp(0.7, 1.0);

    // Generate all points that would be affected by the brush stroke
    let changes = stroke.points
        .iter()
        .filter(|&&[x, y]| {
            x >= 0 && y >= 0 && x < target.width as isize && y < target.height as isize
        })
        .map(|&[x, y]| ([x as u32, y as u32], target_color, paint_amount));

    // Check if drawing this brush stroke would improve the approximation
    let loss_delta = paint_surface.loss_delta(target, changes.clone());

    if loss_delta >= 0.0 {
        return false;
    }

    // Apply the changes if the brush stroke improves the approximation
    paint_surface.apply_stroke(&stroke, target_color, paint_amount);
    true
}

fn select_high_contrast_point(target: &Image, rng: &mut impl Rng) -> (isize, isize) {
    // Simple high-contrast point selection - find areas with color variation
    let mut best_x = target.width / 2;
    let mut best_y = target.height / 2;
    let mut best_contrast = 0.0;
    
    // Sample a few random points and pick the one with highest local contrast
    for _ in 0..20 {
        let x = rng.gen_range(1..(target.width - 1));
        let y = rng.gen_range(1..(target.height - 1));
        
        let center = target.color_at([x, y]);
        let mut total_diff = 0.0;
        
        // Check 4-connected neighbors
        for &(dx, dy) in &[(0, 1), (0, -1), (1, 0), (-1, 0)] {
            let neighbor = target.color_at([(x as i32 + dx) as u32, (y as i32 + dy) as u32]);
            let diff = color_difference(center, neighbor);
            total_diff += diff;
        }
        
        if total_diff > best_contrast {
            best_contrast = total_diff;
            best_x = x;
            best_y = y;
        }
    }
    
    (best_x as isize, best_y as isize)
}

fn color_difference(c1: [u8; 3], c2: [u8; 3]) -> f32 {
    let dr = c1[0] as f32 - c2[0] as f32;
    let dg = c1[1] as f32 - c2[1] as f32;
    let db = c1[2] as f32 - c2[2] as f32;
    (dr * dr + dg * dg + db * db).sqrt()
}

fn calculate_stage_appropriate_color(
    target: &Image, 
    stroke: &BrushStroke, 
    stage_params: &StageParameters
) -> [u8; 3] {
    if stage_params.color_accuracy > 0.9 {
        // High accuracy - use precise center-biased sampling
        calculate_weighted_stroke_color_with_center_bias(target, stroke)
    } else {
        // Lower accuracy - use simpler sampling for more impressionistic effect
        calculate_simple_average_color(target, &stroke.points)
    }
}

// Standalone paint mixing functions
fn mix_colors(
    base_color: [u8; 3],
    base_thickness: f32,
    new_color: [u8; 3],
    new_thickness: f32,
    wetness_factor: f32,
) -> [u8; 3] {
    // Color mixing based on paint properties - less aggressive mixing
    let mixing_strength = wetness_factor * 0.4; // Reduced from 0.7 to preserve color intensity
    
    // Favor the new color more heavily to maintain vibrant colors
    let base_weight = base_thickness * mixing_strength;
    let new_weight = new_thickness; // New paint dominates more
    let total_weight = base_weight + new_weight;
    
    if total_weight < 0.001 {
        return new_color; // Default to new color if weights are negligible
    }
    
    // Convert to subtractive mixing space (simplified)
    let new_ratio = new_weight / total_weight;
    let mixed_r = subtractive_mix(base_color[0], new_color[0], new_ratio);
    let mixed_g = subtractive_mix(base_color[1], new_color[1], new_ratio);
    let mixed_b = subtractive_mix(base_color[2], new_color[2], new_ratio);
    
    [mixed_r, mixed_g, mixed_b]
}

fn subtractive_mix(base: u8, new: u8, mix_ratio: f32) -> u8 {
    // Simple subtractive mixing approximation
    let base_f = 1.0 - (base as f32 / 255.0); // Convert to absorption
    let new_f = 1.0 - (new as f32 / 255.0);
    
    let mixed_absorption = base_f * (1.0 - mix_ratio) + new_f * mix_ratio;
    let mixed_reflection = 1.0 - mixed_absorption;
    
    (mixed_reflection * 255.0).clamp(0.0, 255.0) as u8
}

// ============================================================================
// COMPLETE INTERVAL SPLINE IMPLEMENTATION (Following the Paper)
// ============================================================================

#[derive(Clone, Copy, Debug)]
struct Point2D {
    x: f32,
    y: f32,
}

impl Point2D {
    fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}

// Control shape as described in the paper - stores interval information
#[derive(Clone, Debug)]
struct ControlShape {
    center: Point2D,
    width_interval: [f32; 2],  // [min_width, max_width] - the interval part
    height_interval: [f32; 2], // [min_height, max_height] - NOW USED for elliptical shapes
    angle: f32,                // Orientation of the control shape
    pressure: f32,             // Brush pressure at this point - NOW USED for density
}

impl ControlShape {
    fn new(center: Point2D, width: f32, height: f32, angle: f32, pressure: f32) -> Self {
        // Create intervals with variation for natural brush effects
        let width_var = width * 0.15; // 15% variation
        let height_var = height * 0.1; // 10% variation for height
        
        Self {
            center,
            width_interval: [width - width_var, width + width_var],
            height_interval: [height - height_var, height + height_var],
            angle,
            pressure,
        }
    }
    
    // Get actual width/height considering intervals and pressure
    fn effective_width(&self) -> f32 {
        let base_width = (self.width_interval[0] + self.width_interval[1]) / 2.0;
        base_width * self.pressure.sqrt() // Pressure affects effective width
    }
    
    fn effective_height(&self) -> f32 {
        let base_height = (self.height_interval[0] + self.height_interval[1]) / 2.0;
        base_height * self.pressure.sqrt() // Pressure affects effective height
    }
}

// Interval spline curve representation
struct IntervalSplineCurve {
    control_shapes: Vec<ControlShape>, // NOW USED for interpolation
    upper_boundary: Vec<Point2D>,
    lower_boundary: Vec<Point2D>,
    centerline: Vec<Point2D>,
}

// Final stroke representation
struct BrushStroke {
    points: Vec<[isize; 2]>,
    centerline: Vec<Point2D>,
    control_shapes: Vec<ControlShape>, // Store for color sampling
}

fn generate_interval_spline_stroke(
    start_x: isize,
    start_y: isize,
    angle: f32,
    length: isize,
    base_width: isize,
    rng: &mut impl Rng,
) -> BrushStroke {
    // Step 1: Generate control shapes along the stroke path
    let control_shapes = generate_control_shapes(
        start_x as f32, 
        start_y as f32, 
        angle, 
        length as f32, 
        base_width as f32, 
        rng
    );
    
    // Step 2: Create interval spline curve from control shapes
    let interval_curve = create_interval_spline_curve(&control_shapes);
    
    // Step 3: Rasterize the interval curve to get final stroke points
    let mut stroke = rasterize_interval_spline(&interval_curve);
    
    // Step 4: Apply dry brush effects sparingly (15% chance)
    if rng.random::<f32>() < 0.15 {
        apply_dry_brush_effect(&mut stroke, rng);
    }
    
    stroke
}

fn generate_control_shapes(
    start_x: f32,
    start_y: f32,
    base_angle: f32,
    length: f32,
    base_width: f32,
    rng: &mut impl Rng,
) -> Vec<ControlShape> {
    let num_shapes = 5; // Number of control shapes along the stroke
    let mut shapes = Vec::new();
    
    for i in 0..num_shapes {
        let t = i as f32 / (num_shapes - 1) as f32; // 0.0 to 1.0 along stroke
        
        // Base position along the intended stroke path
        let base_x = start_x + t * length * base_angle.cos();
        let base_y = start_y + t * length * base_angle.sin();
        
        // Add natural curvature (more in middle, less at ends)
        let curve_influence = (t * (1.0 - t) * 4.0).min(1.0);
        let perp_angle = base_angle + std::f32::consts::PI / 2.0;
        let curve_strength = rng.gen_range(-0.3..=0.3) * length * curve_influence;
        
        let center_x = base_x + curve_strength * perp_angle.cos();
        let center_y = base_y + curve_strength * perp_angle.sin();
        
        // Natural width tapering (as described in paper)
        let taper_factor = calculate_natural_taper(t);
        let width_variation = 1.0 + rng.gen_range(-0.2..=0.2);
        let width = base_width * taper_factor * width_variation;
        
        // Height is typically smaller than width for realistic brush shapes
        let height = width * rng.gen_range(0.4..=0.8);
        
        // Pressure simulation (starts high, ends low, with variation)
        let pressure = if t < 0.1 {
            1.0 // High pressure at start
        } else if t > 0.8 {
            0.2 + (1.0 - t) * 0.8 // Tapers off at end
        } else {
            0.7 + rng.gen_range(-0.3..=0.3) // Middle variation
        }.clamp(0.1, 1.0);
        
        // Slight angle variation for organic feel
        let shape_angle = base_angle + rng.gen_range(-0.2..=0.2);
        
        shapes.push(ControlShape::new(
            Point2D::new(center_x, center_y),
            width,
            height,
            shape_angle,
            pressure,
        ));
    }
    
    shapes
}

fn calculate_natural_taper(t: f32) -> f32 {
    // Natural brush taper: starts thick, stays thick in middle, tapers at end
    if t < 0.15 {
        // Gentle start taper
        0.7 + 0.3 * (t / 0.15).powi(2)
    } else if t > 0.75 {
        // Strong end taper (exponential for natural brush lift)
        let end_t = (t - 0.75) / 0.25;
        (1.0 - end_t).powi(3) * 0.9 + 0.1
    } else {
        // Full width in middle with slight variation
        1.0
    }
}

fn create_interval_spline_curve(control_shapes: &[ControlShape]) -> IntervalSplineCurve {
    if control_shapes.len() < 2 {
        return IntervalSplineCurve {
            control_shapes: control_shapes.to_vec(),
            upper_boundary: Vec::new(),
            lower_boundary: Vec::new(),
            centerline: Vec::new(),
        };
    }
    
    let num_samples = 100; // High resolution for smooth curves
    let mut centerline = Vec::new();
    let mut upper_boundary = Vec::new();
    let mut lower_boundary = Vec::new();
    
    // Generate interpolated curves using Catmull-Rom splines
    for i in 0..num_samples {
        let t = i as f32 / (num_samples - 1) as f32;
        
        // Interpolate center position
        let center = interpolate_center_catmull_rom(control_shapes, t);
        centerline.push(center);
        
        // Interpolate width, height, and angle at this position using control shapes
        let width = interpolate_width_from_shapes(control_shapes, t);
        let height = interpolate_height_from_shapes(control_shapes, t);
        let angle = interpolate_angle_from_shapes(control_shapes, t);
        let pressure = interpolate_pressure_from_shapes(control_shapes, t);
        
        // Apply pressure to effective dimensions
        let effective_width = width * pressure.sqrt();
        let effective_height = height * pressure.sqrt();
        
        // Calculate perpendicular direction for elliptical cross-section
        let perp_angle = angle + std::f32::consts::PI / 2.0;
        let half_width = effective_width / 2.0;
        let half_height = effective_height / 2.0;
        
        // Create elliptical upper and lower boundary points
        // Use average of width and height for simplified boundary (could be more complex)
        let avg_radius = (half_width + half_height) / 2.0;
        
        let upper = Point2D::new(
            center.x + avg_radius * perp_angle.cos(),
            center.y + avg_radius * perp_angle.sin(),
        );
        let lower = Point2D::new(
            center.x - avg_radius * perp_angle.cos(),
            center.y - avg_radius * perp_angle.sin(),
        );
        
        upper_boundary.push(upper);
        lower_boundary.push(lower);
    }
    
    IntervalSplineCurve {
        control_shapes: control_shapes.to_vec(),
        upper_boundary,
        lower_boundary,
        centerline,
    }
}

fn interpolate_center_catmull_rom(shapes: &[ControlShape], t: f32) -> Point2D {
    if shapes.len() < 2 {
        return shapes[0].center;
    }
    
    let segment_t = t * (shapes.len() - 1) as f32;
    let segment_index = segment_t.floor() as usize;
    let local_t = segment_t - segment_index as f32;
    
    if segment_index >= shapes.len() - 1 {
        return shapes[shapes.len() - 1].center;
    }
    
    // Get 4 control points for Catmull-Rom
    let p0 = if segment_index == 0 { shapes[0].center } else { shapes[segment_index - 1].center };
    let p1 = shapes[segment_index].center;
    let p2 = shapes[segment_index + 1].center;
    let p3 = if segment_index + 2 < shapes.len() { shapes[segment_index + 2].center } else { shapes[shapes.len() - 1].center };
    
    // Catmull-Rom interpolation formula
    let t2 = local_t * local_t;
    let t3 = t2 * local_t;
    
    let x = 0.5 * (
        2.0 * p1.x +
        (-p0.x + p2.x) * local_t +
        (2.0 * p0.x - 5.0 * p1.x + 4.0 * p2.x - p3.x) * t2 +
        (-p0.x + 3.0 * p1.x - 3.0 * p2.x + p3.x) * t3
    );
    
    let y = 0.5 * (
        2.0 * p1.y +
        (-p0.y + p2.y) * local_t +
        (2.0 * p0.y - 5.0 * p1.y + 4.0 * p2.y - p3.y) * t2 +
        (-p0.y + 3.0 * p1.y - 3.0 * p2.y + p3.y) * t3
    );
    
    Point2D::new(x, y)
}

fn interpolate_width_from_shapes(shapes: &[ControlShape], t: f32) -> f32 {
    if shapes.len() < 2 {
        return shapes[0].effective_width();
    }
    
    let segment_t = t * (shapes.len() - 1) as f32;
    let segment_index = segment_t.floor() as usize;
    let local_t = segment_t - segment_index as f32;
    
    if segment_index >= shapes.len() - 1 {
        return shapes[shapes.len() - 1].effective_width();
    }
    
    let width1 = shapes[segment_index].effective_width();
    let width2 = shapes[segment_index + 1].effective_width();
    
    width1 * (1.0 - local_t) + width2 * local_t
}

fn interpolate_height_from_shapes(shapes: &[ControlShape], t: f32) -> f32 {
    if shapes.len() < 2 {
        return shapes[0].effective_height();
    }
    
    let segment_t = t * (shapes.len() - 1) as f32;
    let segment_index = segment_t.floor() as usize;
    let local_t = segment_t - segment_index as f32;
    
    if segment_index >= shapes.len() - 1 {
        return shapes[shapes.len() - 1].effective_height();
    }
    
    let height1 = shapes[segment_index].effective_height();
    let height2 = shapes[segment_index + 1].effective_height();
    
    height1 * (1.0 - local_t) + height2 * local_t
}

fn interpolate_angle_from_shapes(shapes: &[ControlShape], t: f32) -> f32 {
    if shapes.len() < 2 {
        return shapes[0].angle;
    }
    
    let segment_t = t * (shapes.len() - 1) as f32;
    let segment_index = segment_t.floor() as usize;
    let local_t = segment_t - segment_index as f32;
    
    if segment_index >= shapes.len() - 1 {
        return shapes[shapes.len() - 1].angle;
    }
    
    let angle1 = shapes[segment_index].angle;
    let angle2 = shapes[segment_index + 1].angle;
    
    // Handle angle wrapping
    let mut diff = angle2 - angle1;
    if diff > std::f32::consts::PI {
        diff -= 2.0 * std::f32::consts::PI;
    } else if diff < -std::f32::consts::PI {
        diff += 2.0 * std::f32::consts::PI;
    }
    
    angle1 + diff * local_t
}

fn interpolate_pressure_from_shapes(shapes: &[ControlShape], t: f32) -> f32 {
    if shapes.len() < 2 {
        return shapes[0].pressure;
    }
    
    let segment_t = t * (shapes.len() - 1) as f32;
    let segment_index = segment_t.floor() as usize;
    let local_t = segment_t - segment_index as f32;
    
    if segment_index >= shapes.len() - 1 {
        return shapes[shapes.len() - 1].pressure;
    }
    
    let pressure1 = shapes[segment_index].pressure;
    let pressure2 = shapes[segment_index + 1].pressure;
    
    pressure1 * (1.0 - local_t) + pressure2 * local_t
}

fn rasterize_interval_spline(curve: &IntervalSplineCurve) -> BrushStroke {
    let mut points = Vec::new();
    
    if curve.upper_boundary.is_empty() || curve.lower_boundary.is_empty() {
        return BrushStroke { 
            points, 
            centerline: curve.centerline.clone(),
            control_shapes: curve.control_shapes.clone(),
        };
    }
    
    // Find bounding box
    let mut min_x = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    
    for point in curve.upper_boundary.iter().chain(curve.lower_boundary.iter()) {
        min_x = min_x.min(point.x);
        max_x = max_x.max(point.x);
        min_y = min_y.min(point.y);
        max_y = max_y.max(point.y);
    }
    
    // Sample every integer coordinate in bounding box and test if inside stroke
    for y in (min_y.floor() as isize)..=(max_y.ceil() as isize) {
        for x in (min_x.floor() as isize)..=(max_x.ceil() as isize) {
            if is_point_inside_stroke(&Point2D::new(x as f32, y as f32), curve) {
                points.push([x, y]);
            }
        }
    }
    
    BrushStroke { 
        points, 
        centerline: curve.centerline.clone(),
        control_shapes: curve.control_shapes.clone(),
    }
}

fn is_point_inside_stroke(point: &Point2D, curve: &IntervalSplineCurve) -> bool {
    // Use ray casting to determine if point is inside the stroke polygon
    // formed by upper_boundary + reversed lower_boundary
    
    let mut polygon = curve.upper_boundary.clone();
    let mut lower_reversed = curve.lower_boundary.clone();
    lower_reversed.reverse();
    polygon.extend(lower_reversed);
    
    if polygon.len() < 3 {
        return false;
    }
    
    let mut inside = false;
    let mut j = polygon.len() - 1;
    
    for i in 0..polygon.len() {
        if ((polygon[i].y > point.y) != (polygon[j].y > point.y)) &&
           (point.x < (polygon[j].x - polygon[i].x) * (point.y - polygon[i].y) / (polygon[j].y - polygon[i].y) + polygon[i].x) {
            inside = !inside;
        }
        j = i;
    }
    
    inside
}

// ============================================================================
// DRY BRUSH EFFECTS (Applied Sparingly)
// ============================================================================

fn apply_dry_brush_effect(stroke: &mut BrushStroke, rng: &mut impl Rng) {
    let dryness = rng.gen_range(0.3..=0.7); // Random dryness level
    
    // Create gaps in the stroke (broken texture effect)
    stroke.points.retain(|_| {
        // Keep more points in center, fewer at edges
        rng.random::<f32>() > dryness * 0.4 // Remove up to 28% of points when very dry
    });
    
    // Add texture variation to remaining points
    for point in &mut stroke.points {
        if rng.random::<f32>() < dryness * 0.3 {
            // Small random displacement for scratchy texture
            point[0] += rng.gen_range(-1i32..=1i32) as isize;
            point[1] += rng.gen_range(-1i32..=1i32) as isize;
        }
    }
    
    // Add some bristle marks along the centerline (very sparingly)
    if rng.random::<f32>() < 0.4 { // 40% chance for bristle marks
        add_sparse_bristle_marks(stroke, rng);
    }
}

fn add_sparse_bristle_marks(stroke: &mut BrushStroke, rng: &mut impl Rng) {
    let mut bristle_points = Vec::new();
    let bristle_spacing = 8; // Every 8th point along centerline
    
    for (i, center) in stroke.centerline.iter().enumerate() {
        if i % bristle_spacing == 0 && rng.random::<f32>() < 0.6 { // 60% chance per eligible point
            // Get approximate width at this position
            let t = i as f32 / (stroke.centerline.len() - 1) as f32;
            let width = if !stroke.control_shapes.is_empty() {
                interpolate_width_from_shapes(&stroke.control_shapes, t)
            } else {
                4.0 // Fallback width
            };
            
            // Add 2-3 bristle marks perpendicular to stroke direction
            let num_bristles = rng.gen_range(2..=3);
            for bristle_idx in 0..num_bristles {
                let offset_ratio = (bristle_idx as f32 / (num_bristles - 1) as f32 - 0.5) * 0.8;
                
                // Calculate perpendicular direction
                let perp_angle = if i < stroke.centerline.len() - 1 {
                    let next = stroke.centerline[i + 1];
                    let dx = next.x - center.x;
                    let dy = next.y - center.y;
                    dy.atan2(dx) + std::f32::consts::PI / 2.0
                } else if i > 0 {
                    let prev = stroke.centerline[i - 1];
                    let dx = center.x - prev.x;
                    let dy = center.y - prev.y;
                    dy.atan2(dx) + std::f32::consts::PI / 2.0
                } else {
                    0.0
                };
                
                let bristle_offset = offset_ratio * width * 0.3;
                let bristle_x = center.x + bristle_offset * perp_angle.cos();
                let bristle_y = center.y + bristle_offset * perp_angle.sin();
                
                // Add small line of bristle points
                for line_step in 0..3 {
                    let step_offset = (line_step as f32 - 1.0) * 0.5;
                    let final_x = bristle_x + step_offset * perp_angle.cos();
                    let final_y = bristle_y + step_offset * perp_angle.sin();
                    
                    bristle_points.push([final_x as isize, final_y as isize]);
                }
            }
        }
    }
    
    stroke.points.extend(bristle_points);
}

fn calculate_weighted_stroke_color_with_center_bias(target: &Image, stroke: &BrushStroke) -> [u8; 3] {
    if stroke.centerline.is_empty() {
        // Fallback to simple average if no centerline
        return calculate_simple_average_color(target, &stroke.points);
    }
    
    let mut total_r = 0.0;
    let mut total_g = 0.0;
    let mut total_b = 0.0;
    let mut total_weight = 0.0;
    
    // Sample along centerline with higher weights toward center
    let centerline_samples = 20; // Sample 20 points along centerline
    for i in 0..centerline_samples {
        let t = i as f32 / (centerline_samples - 1) as f32;
        let centerline_idx = (t * (stroke.centerline.len() - 1) as f32) as usize;
        let center_point = stroke.centerline[centerline_idx];
        
        let x = center_point.x as u32;
        let y = center_point.y as u32;
        
        if x < target.width && y < target.height {
            let [r, g, b] = target.color_at([x, y]);
            
            // Weight based on distance from stroke center (higher weight in middle)
            let center_distance = (t - 0.5).abs(); // 0.0 at center, 0.5 at ends
            let weight = 2.0 - center_distance * 2.0; // 2.0 at center, 1.0 at ends
            
            total_r += r as f32 * weight;
            total_g += g as f32 * weight;
            total_b += b as f32 * weight;
            total_weight += weight;
        }
    }
    
    if total_weight > 0.0 {
        [
            (total_r / total_weight) as u8,
            (total_g / total_weight) as u8,
            (total_b / total_weight) as u8,
        ]
    } else {
        calculate_simple_average_color(target, &stroke.points)
    }
}

fn calculate_simple_average_color(target: &Image, stroke_points: &[[isize; 2]]) -> [u8; 3] {
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

// ============================================================================
// IMAGE PROCESSING AND DISPLAY CODE (adapted for hierarchical approach)
// ============================================================================

type Point = [u32; 2];
type Color = [u8; 3];

// derive clone
#[derive(Clone)]
struct Image {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
}

impl Image {
    fn color_at(&self, point: Point) -> Color {
        let offset = (point[1] * self.width + point[0]) as usize * 3;
        let color = &self.pixels[offset..][..3];
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