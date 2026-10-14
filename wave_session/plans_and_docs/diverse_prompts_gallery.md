# Diverse Prompts Optical Wave Generative Gallery

> **Created**: 2026-10-14 07:07 KST  
> **Last updated**: 2026-10-14 07:07 KST  
> **Modified by**: Antigravity, Pair Programming Assistant  
> **Version**: 1.0.0

---

## 1. Multi-Prompt Scenic Gallery

Using our trained **Fourier OpticalWaveAdapter** ($4.426\text{ M}$ parameters, actively hooked into UNet `mid_block`) paired with ControlNet geometry conditioning and the **Rayleigh-Sommerfeld Angular Spectrum Method (ASM)** physical wave optics solver, we synthesized four distinct scenic interpretations of the virtual park environment.

````carousel
![1. Sunlit Morning Park (Master)](/root/.gemini/antigravity-cli/brain/d00fed9b-b89d-457b-a2e4-57b2a9f5450b/virtual_park_sota_master.png)
<!-- slide -->
![2. Autumn Golden Hour Sunset](/root/.gemini/antigravity-cli/brain/d00fed9b-b89d-457b-a2e4-57b2a9f5450b/virtual_park_autumn_sunset.png)
<!-- slide -->
![3. Rainy Twilight with Water Reflections](/root/.gemini/antigravity-cli/brain/d00fed9b-b89d-457b-a2e4-57b2a9f5450b/virtual_park_rainy_twilight.png)
<!-- slide -->
![4. Spring Cherry Blossom Japanese Garden](/root/.gemini/antigravity-cli/brain/d00fed9b-b89d-457b-a2e4-57b2a9f5450b/virtual_park_spring_cherry_blossom.png)
````

---

## 2. Comparative Scene Breakdown

| Scene Variant | Prompt Key Elements | Depth & Optical Characteristics | Synthesis Speed |
| :--- | :--- | :--- | :--- |
| **1. Sunlit Morning** | Weeping willow, morning sunbeams, handcrafted bench, calm lake | Strong crepuscular light rays, specular lake highlights, crisp lawn blades | 26.8s (50 steps) |
| **2. Autumn Sunset** | Fiery crimson & amber maple canopy, fallen leaves, sunset clouds | Golden-hour warm color cast, fallen leaf ground textures, distant bridge | 24.3s (50 steps) |
| **3. Rainy Twilight** | Gentle evening rain, crystalline water droplets, blue twilight sky | Low-light specular puddle reflections, dark willow silhouette, soft atmospheric haze | 23.9s (50 steps) |
| **4. Spring Sakura** | Delicate pink cherry blossoms, traditional arched footbridge, koi pond | High dynamic range pastel tones, petal scattered lawn, dual bridge reflections | 24.7s (50 steps) |

---

## 3. Physical Optics & Invariant Compliance

1. **Optical Wave Consistency**:
   - Each frame was processed through the 3-channel Rayleigh-Sommerfeld ASM engine across $\lambda_R = 650\text{ nm}, \lambda_G = 532\text{ nm}, \lambda_B = 450\text{ nm}$ with circular lens pupil aperture stop ($f/2.8, 50\text{ mm}$).
   - Depth fields were predicted using `Intel/dpt-hybrid-midas`, maintaining in-focus foreground structures while applying smooth circular aperture diffraction bokeh to background waters and skies.

2. **Parameter Ceiling**:
   - All variations utilized the identical trained checkpoint `optical_wave_adapter_trained.pt` (**4,425,729 parameters**, strictly $< 20.0\text{ M}$ ceiling).

3. **Compute Budget & Latency**:
   - Total batch execution time for 3 novel scenes: **72.9 seconds** on NVIDIA GeForce RTX 4090.
   - Remaining Vast.ai balance: **\$7.61** (well within budget limit).
