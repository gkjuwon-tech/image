import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, EulerAncestralDiscreteScheduler
from diffusers import StableDiffusionImg2ImgPipeline
from transformers import pipeline

print("====================================================================")
print("MULTI-PROMPT MASTER GALLERY SYNTHESIS WITH OPTICAL WAVE ADAPTER")
print("====================================================================")

# 1. Fourier Spatial-Frequency Phase Adapter Definition
class OpticalWaveAdapter(nn.Module):
    def __init__(self, in_channels=1280, hidden_channels=192):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(16, hidden_channels)
        self.act = nn.SiLU()
        self.phase_angle = nn.Parameter(torch.randn(hidden_channels, 1, 1) * 0.05)
        self.conv_out = nn.Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1)
        self.scale = nn.Parameter(torch.tensor(0.15))
        
    def forward(self, x):
        h = self.act(self.norm1(self.conv_in(x)))
        H_freq = torch.fft.rfft2(h.float(), norm="ortho")
        phase_factor = torch.complex(torch.cos(self.phase_angle.float()), torch.sin(self.phase_angle.float()))
        H_mod = H_freq * phase_factor
        h_spatial = torch.fft.irfft2(H_mod, s=h.shape[-2:], norm="ortho").to(h.dtype)
        out = self.conv_out(h_spatial)
        return self.scale * out

wave_adapter = OpticalWaveAdapter(in_channels=1280, hidden_channels=192).to("cuda", dtype=torch.float16)
ckpt_path = "/workspace/output/optical_wave_adapter_trained.pt"
assert os.path.exists(ckpt_path), f"Checkpoint {ckpt_path} missing!"
state_dict = torch.load(ckpt_path, map_location="cuda", weights_only=True)
state_dict = {k: v.to(torch.float16) for k, v in state_dict.items()}
wave_adapter.load_state_dict(state_dict)
wave_adapter.eval()
print(f"Loaded trained Fourier adapter with learned scale: {wave_adapter.scale.item():.4f}")

# 2. Pipeline Initialization
print("Loading ControlNet and Realistic Vision Pipeline...")
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_canny",
    torch_dtype=torch.float16
).to("cuda")

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "SG161222/Realistic_Vision_V5.1_noVAE",
    controlnet=controlnet,
    torch_dtype=torch.float16,
    safety_checker=None
).to("cuda")

pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.enable_attention_slicing()

# Register hook on UNet mid_block
def mid_block_hook(module, input, output):
    res = wave_adapter(output)
    return output + res

hook_handle = pipe.unet.mid_block.register_forward_hook(mid_block_hook)
print("SUCCESS: Trained OpticalWaveAdapter hooked into UNet mid_block!")

# 1024 Refinement pipeline
refine_pipe = StableDiffusionImg2ImgPipeline(
    vae=pipe.vae,
    text_encoder=pipe.text_encoder,
    tokenizer=pipe.tokenizer,
    unet=pipe.unet,
    scheduler=pipe.scheduler,
    safety_checker=None,
    feature_extractor=None
).to("cuda")

# Depth Estimator
print("Loading Intel/dpt-hybrid-midas depth estimator...")
depth_estimator = pipeline("depth-estimation", model="Intel/dpt-hybrid-midas", device=0)

# 3. ASM Wave Optics Engine
class RayleighSommerfeldASMEngine:
    def __init__(self, wavelengths=(650e-9, 532e-9, 450e-9), pixel_pitch=3.45e-6, f_number=2.8, focal_length=0.050):
        self.wavelengths = wavelengths
        self.pixel_pitch = pixel_pitch
        self.f_number = f_number
        self.focal_length = focal_length
        self.aperture_radius = focal_length / (2.0 * f_number)

    def propagate_channel(self, channel_tensor, z_defocus, lam):
        H, W = channel_tensor.shape
        device = channel_tensor.device
        fx = torch.fft.fftfreq(W, d=self.pixel_pitch, device=device)
        fy = torch.fft.fftfreq(H, d=self.pixel_pitch, device=device)
        FY, FX = torch.meshgrid(fy, fx, indexing="ij")
        rho_sq = FX**2 + FY**2
        
        f_nyquist = 1.0 / (2.0 * self.pixel_pitch)
        cutoff_freq = min(self.aperture_radius / (lam * self.focal_length), f_nyquist * 0.35)
        pupil_mask = torch.exp(-rho_sq / (2.0 * (cutoff_freq * 0.65)**2))
        
        field_in = torch.complex(torch.sqrt(torch.clamp(channel_tensor, min=1e-8)), torch.zeros_like(channel_tensor))
        arg = 1.0 - (lam**2) * rho_sq
        propagating_mask = (arg > 0).float()
        kz = (2.0 * math.pi / lam) * torch.sqrt(torch.clamp(arg, min=0.0))
        H_transfer = torch.complex(torch.cos(kz * z_defocus), torch.sin(kz * z_defocus)) * propagating_mask
        
        U_freq = torch.fft.fft2(field_in)
        U_prop = torch.fft.ifft2(U_freq * H_transfer * pupil_mask)
        return torch.real(U_prop * torch.conj(U_prop))

    def render(self, img_pil, depth_field):
        img_np = np.array(img_pil).astype(np.float32) / 255.0
        img_t = torch.from_numpy(img_np).permute(2, 0, 1).to("cuda")
        depth_t = torch.from_numpy(depth_field).to("cuda")
        
        out_channels = []
        for c, lam in enumerate(self.wavelengths):
            ch = img_t[c]
            ch_bokeh = self.propagate_channel(ch, z_defocus=0.002, lam=lam)
            bokeh_weight = torch.clamp((depth_t - 0.20) / 0.80, 0.0, 0.70)
            blended = (1.0 - bokeh_weight) * ch + bokeh_weight * ch_bokeh
            out_channels.append(blended)
            
        final_t = torch.stack(out_channels, dim=0)
        final_np = final_t.permute(1, 2, 0).cpu().numpy()
        final_np = (np.clip(final_np, 0.0, 1.0) * 255.0).astype(np.uint8)
        return Image.fromarray(final_np)

asm_engine = RayleighSommerfeldASMEngine()

# 4. Load Conditioning Sketch
sketch_path = "/workspace/output/park_control_sketch.png"
sketch_img = Image.open(sketch_path).convert("RGB")
sketch_768 = sketch_img.resize((768, 768), Image.BILINEAR)

# 5. Define Diverse Prompts
prompts_dict = {
    "autumn_sunset": {
        "title": "Autumn Golden Hour Sunset",
        "prompt": (
            "A magnificent landscape photograph of an autumn park at golden hour, a grand maple tree "
            "on the left with fiery crimson and amber-gold foliage gracefully arching overhead, "
            "a single solitary handcrafted dark wood park bench resting on sun-dappled lawn covered with crisp fallen leaves in the foreground, "
            "a tranquil reflective lake on the right mirroring radiant sunset clouds and warm violet-orange skies, "
            "crisp leaf textures, wood grain, sharp focus on wooden bench, circular lens pupil optical bokeh in background, authentic 8k DSLR photography"
        ),
        "negative_prompt": "painting, drawing, illustration, cartoon, anime, 3d render, CGI, plastic, oversaturated, blurry, multiple benches, extra limbs, bad anatomy, artificial, amateur, watermark, text",
        "seed": 8888
    },
    "rainy_twilight": {
        "title": "Rainy Twilight with Reflections",
        "prompt": (
            "A cinematic masterwork photograph of a quiet park at twilight after gentle evening rain, "
            "an ancient weeping willow tree on the left dripping with crystalline rain droplets, "
            "a single solitary dark wooden park bench on the foreground lawn glistening with damp reflections, "
            "a serene reflective lake on the right mirroring the deep blue twilight sky and distant ambient lanterns, "
            "puddle specular reflections, sharp focus on wooden bench and water droplets, physical wave optics circular bokeh, authentic 8k DSLR photography"
        ),
        "negative_prompt": "painting, drawing, illustration, cartoon, anime, 3d render, CGI, plastic, oversaturated, blurry, miniature bench, multiple benches, artificial, amateur, watermark, text",
        "seed": 5555
    },
    "spring_cherry_blossom": {
        "title": "Spring Cherry Blossom Bloom",
        "prompt": (
            "An enchanting masterwork landscape photograph of a sunlit Japanese garden park in full spring bloom, "
            "a grand weeping cherry blossom sakura tree on the left with lush delicate pink blossoms gracefully arching overhead, "
            "a single solitary handcrafted dark wood park bench resting on vibrant emerald green lawn scattered with pink petals, "
            "a crystal-clear reflective koi pond on the right mirroring pink blossom canopy and azure sky, "
            "radiant morning sun rays, crisp petal textures, sharp focus on wooden bench, circular lens pupil bokeh, authentic 8k DSLR photography"
        ),
        "negative_prompt": "painting, drawing, illustration, cartoon, anime, 3d render, CGI, plastic, oversaturated, blurry, multiple benches, extra limbs, bad anatomy, artificial, amateur, watermark, text",
        "seed": 9999
    }
}

# 6. Execute Generation Loop
for slug, info in prompts_dict.items():
    print(f"\n========================================================")
    print(f"Generating Scene: {info['title']} ({slug})")
    print(f"========================================================")
    
    gen = torch.Generator(device="cuda").manual_seed(info["seed"])
    
    # Base synthesis at 768x768
    with torch.inference_mode():
        base_img = pipe(
            prompt=info["prompt"],
            negative_prompt=info["negative_prompt"],
            image=sketch_768,
            num_inference_steps=50,
            guidance_scale=8.0,
            controlnet_conditioning_scale=0.82,
            generator=gen,
            width=768,
            height=768
        ).images[0]
        
        # 1024 native texture refinement
        base_1024 = base_img.resize((1024, 1024), Image.LANCZOS)
        refined_1024 = refine_pipe(
            prompt=info["prompt"],
            negative_prompt=info["negative_prompt"],
            image=base_1024,
            strength=0.22,
            guidance_scale=7.5,
            num_inference_steps=30,
            generator=gen
        ).images[0]

    # Monocular depth map
    depth_res = depth_estimator(refined_1024)
    depth_raw = np.array(depth_res["depth"]).astype(np.float32)
    depth_norm = 1.0 - ((depth_raw - depth_raw.min()) / (depth_raw.max() - depth_raw.min() + 1e-6))
    depth_map = cv2.resize(depth_norm, (1024, 1024), interpolation=cv2.INTER_CUBIC)
    
    # ASM Wave Optics rendering
    final_img = asm_engine.render(refined_1024, depth_map)
    
    out_img_path = f"/workspace/output/virtual_park_{slug}.png"
    out_depth_path = f"/workspace/output/virtual_park_{slug}_depth.png"
    final_img.save(out_img_path, quality=99)
    cv2.imwrite(out_depth_path, (depth_map * 255.0).astype(np.uint8))
    
    print(f"Saved: {out_img_path} ({final_img.size[0]}x{final_img.size[1]}, {os.path.getsize(out_img_path)/(1024*1024):.2f} MB)")

print("\nALL DIVERSE PROMPT SCENES GENERATED SUCCESSFULLY!")
