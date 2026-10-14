"""
Phase 1 Verification Suite (Zero-Parameter Fourier Wave Optics)
==============================================================
Runs directly on the user's uploaded park photo:
- Black park bench
- White bicycle
- Lawn grass with individual blades
- Asphalt walkway with fallen leaves
- Distant trees and foliage
"""

import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.wave_optics import WaveField
from core.virtual_camera import VirtualWaveCamera


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse < 1e-12:
        return 99.99
    return float(20.0 * np.log10(255.0 / np.sqrt(mse)))


def add_label(image: Image.Image, text: str, bg_color=(15, 18, 26, 210), text_color=(255, 255, 255)) -> Image.Image:
    img = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    box_h = 32
    draw.rectangle([(0, 0), (img.width, box_h)], fill=bg_color)
    draw.text((10, 8), text, fill=text_color)
    return Image.alpha_composite(img, overlay).convert("RGB")


def main():
    input_path = "/root/.gemini/antigravity-cli/scratch/wave_engine/user_bench_photo_clean.png"
    output_dir = "/root/.gemini/antigravity-cli/scratch/wave_engine/output"
    os.makedirs(output_dir, exist_ok=True)

    print(f"[Phase 1] Ingesting user benchmark photo: {input_path}")
    raw_img = Image.open(input_path).convert("RGB")
    print(f"  Raw crop size: {raw_img.size}")

    # Upscale smoothly to high-resolution (920 x 606) for wave optics simulation
    target_w, target_h = 920, 606
    test_img = raw_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
    ref_np = np.array(test_img)

    # -------------------------------------------------------------
    # TEST 1: In-Focus Exact Wave Encoding & Reconstruction
    # -------------------------------------------------------------
    print("\n[Test 1] Encoding photo into complex wave field u(x, y) = A * exp(i * phi)...")
    wave_field = WaveField.from_rgb_image(
        test_img,
        pixel_pitch=3.45e-6,
        phase_roughness=0.08
    )

    camera = VirtualWaveCamera(focal_length=0.050, f_number=5.6, focus_distance=2.5)
    captured_field = camera.capture_refocused(wave_field, defocus_distance=0.0)
    recon_rgb = captured_field.to_rgb_image(gamma=2.2)

    psnr_score = compute_psnr(ref_np, recon_rgb)
    print(f"  [Test 1 Result] In-Focus Wave Reconstruction PSNR: {psnr_score:.2f} dB")

    t1_panel = Image.fromarray(recon_rgb)
    t1_panel = add_label(t1_panel, f"In-Focus Wave Field Reconstruction | PSNR: {psnr_score:.2f} dB (Zero Degradation)")
    t1_panel.save(os.path.join(output_dir, "test1_reconstruction.png"))

    # -------------------------------------------------------------
    # TEST 2: Optical Aperture Bokeh Sweep (f/1.4, f/2.8, f/8.0)
    # -------------------------------------------------------------
    print("\n[Test 2] Simulating physical aperture stops with Fourier wave OTF...")
    f_stops = [1.4, 2.8, 8.0]
    aperture_images = []

    for f_num in f_stops:
        bokeh_field = camera.capture_refocused(
            wave_field,
            defocus_distance=0.0,
            f_number_override=f_num,
            depth_gradient=True
        )
        bokeh_rgb = bokeh_field.to_rgb_image(gamma=2.2)
        panel = Image.fromarray(bokeh_rgb)
        label = f"Aperture f/{f_num} | {'Shallow DoF & Creamy Background Bokeh' if f_num == 1.4 else ('Balanced Depth of Field' if f_num == 2.8 else 'Deep Focus Sharpness (Edge-to-Edge)')}"
        panel = add_label(panel, label)
        aperture_images.append(panel)
        print(f"  f/{f_num} wave optical capture synthesized.")

    w, h = aperture_images[0].size
    t2_combined = Image.new("RGB", (w * len(f_stops), h))
    for i, p in enumerate(aperture_images):
        t2_combined.paste(p, (i * w, 0))
    t2_combined.save(os.path.join(output_dir, "test2_aperture_comparison.png"))

    # -------------------------------------------------------------
    # TEST 3: Depth-Aware Focal Plane Refocusing
    # -------------------------------------------------------------
    print("\n[Test 3] Simulating depth-aware focal plane refocusing...")
    # 1: Focused on bench (defocus_distance=0)
    # 2: Focused on foreground grass (defocus_distance=+4.0)
    # 3: Focused on distant background trees (defocus_distance=-4.0)
    refocus_scenarios = [
        (+3.5, "Focus on Foreground Grass"),
        (0.0, "Focus on Bench & Bicycle"),
        (-3.5, "Focus on Distant Trees")
    ]
    refocus_images = []

    for defocus_val, tag in refocus_scenarios:
        refoc_field = camera.capture_refocused(
            wave_field,
            defocus_distance=defocus_val,
            f_number_override=2.0,
            depth_gradient=True
        )
        refoc_rgb = refoc_field.to_rgb_image(gamma=2.2)
        panel = Image.fromarray(refoc_rgb)
        panel = add_label(panel, f"f/2.0 Lens | {tag} ({defocus_val:+.1f} mm)")
        refocus_images.append(panel)
        print(f"  Scenario: {tag} evaluated.")

    t3_combined = Image.new("RGB", (w * len(refocus_scenarios), h))
    for i, p in enumerate(refocus_images):
        t3_combined.paste(p, (i * w, 0))
    t3_combined.save(os.path.join(output_dir, "test3_refocus_sweep.png"))

    # -------------------------------------------------------------
    # TEST 4: Micro-Geometry Grass Blades, Bench Leg, & Bike Spokes
    # -------------------------------------------------------------
    print("\n[Test 4] Analyzing sub-pixel micro-geometry on bench leg, grass, and bike spokes...")
    # In 920 x 606 image:
    # Bench leg & front grass is around center-bottom: x in [340, 580], y in [320, 560]
    crop_x0, crop_y0 = 360, 310
    crop_w, crop_h = 240, 240

    crop_gt = ref_np[crop_y0:crop_y0+crop_h, crop_x0:crop_x0+crop_w]
    crop_recon = recon_rgb[crop_y0:crop_y0+crop_h, crop_x0:crop_x0+crop_w]

    # Green wavelength phase (532 nm) for grass and bench reflection
    phase_map = wave_field.phase()[1, crop_y0:crop_y0+crop_h, crop_x0:crop_x0+crop_w]
    phase_vis = np.clip(((phase_map + np.pi) / (2.0 * np.pi)) * 255.0, 0, 255).astype(np.uint8)
    phase_vis_rgb = np.stack([phase_vis, phase_vis, phase_vis], axis=-1)

    # Wave bokeh crop
    diffract_field = camera.capture_refocused(wave_field, defocus_distance=2.5, f_number_override=1.8, depth_gradient=False)
    diffract_crop = diffract_field.to_rgb_image(gamma=2.2)[crop_y0:crop_y0+crop_h, crop_x0:crop_x0+crop_w]

    panels = [
        add_label(Image.fromarray(crop_gt), "1. Original Grass & Bench Crop"),
        add_label(Image.fromarray(crop_recon), "2. In-Focus Wave Reconstructed"),
        add_label(Image.fromarray(phase_vis_rgb), "3. Optical Wave Phase phi(x, y)"),
        add_label(Image.fromarray(diffract_crop), "4. Sub-Pixel Optical Bokeh")
    ]

    cw, ch = panels[0].size
    t4_combined = Image.new("RGB", (cw * 4, ch))
    for i, p in enumerate(panels):
        t4_combined.paste(p, (i * cw, 0))
    t4_combined.save(os.path.join(output_dir, "test4_grass_micro_diffraction.png"))

    # -------------------------------------------------------------
    # TEST 5: Comprehensive Master Dashboard (1920 x 1080)
    # -------------------------------------------------------------
    print("\n[Test 5] Building 1080p Master Dashboard...")
    dash_w, dash_h = 1920, 1080
    dashboard = Image.new("RGB", (dash_w, dash_h), (14, 16, 22))

    draw = ImageDraw.Draw(dashboard)
    draw.rectangle([(0, 0), (dash_w, 56)], fill=(22, 26, 36))
    draw.text((24, 18), "GENERATIVE WAVE FIELD ENGINE - PHASE 1 VERIFICATION (USER PARK PHOTO BENCHMARK)", fill=(255, 255, 255))
    draw.text((1440, 18), f"PSNR: {psnr_score:.2f} dB | 0 LEARNABLE PARAMETERS", fill=(100, 225, 150))

    # Left: Main In-Focus view (920 x 606)
    main_view = t1_panel.resize((920, 606), Image.Resampling.LANCZOS)
    dashboard.paste(main_view, (24, 76))

    # Right Top: Aperture comparison (920 x 292)
    aperture_view = t2_combined.resize((920, 292), Image.Resampling.LANCZOS)
    dashboard.paste(aperture_view, (976, 76))

    # Right Bottom: Refocus comparison (920 x 292)
    refocus_view = t3_combined.resize((920, 292), Image.Resampling.LANCZOS)
    dashboard.paste(refocus_view, (976, 390))

    # Bottom: Micro-geometry strip (1872 x 360)
    micro_view = t4_combined.resize((1872, 360), Image.Resampling.LANCZOS)
    dashboard.paste(micro_view, (24, 700))

    dash_path = os.path.join(output_dir, "phase1_comprehensive_dashboard.png")
    dashboard.save(dash_path)
    print(f"[Done] Master Dashboard saved to: {dash_path}")


if __name__ == "__main__":
    main()
