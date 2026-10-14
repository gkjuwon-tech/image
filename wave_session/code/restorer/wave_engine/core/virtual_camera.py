"""
Virtual Wave Camera & Physical Lens Simulator (Phase 1)
======================================================
Implements rigorous Fourier optics image formation:
- Generalized Pupil Function P(u, v) * exp(i * 2pi * W_20 * r^2)
- Diffraction-limited Incoherent Point Spread Function (IPSF) & Airy Disk
- True optical bokeh with Fresnel diffraction fringes
- Variable aperture (f/1.4, f/2.8, f/8.0) and polygonal/circular iris
- Zero-aliasing padded convolution eliminating boundary artifacts
"""

import numpy as np
from scipy import fft
from typing import Tuple, Optional
from .wave_optics import WaveField


class VirtualWaveCamera:
    """
    Fourier Wave Optics Camera Simulator.
    """
    def __init__(
        self,
        focal_length: float = 0.050,
        f_number: float = 2.8,
        num_blades: int = 9,
        focus_distance: float = 2.5
    ):
        self.focal_length = float(focal_length)
        self.f_number = float(f_number)
        self.num_blades = int(num_blades)
        self.focus_distance = float(focus_distance)

    def compute_wave_psf(
        self,
        kernel_size: int,
        f_number: float,
        defocus_mm: float,
        wavelength: float,
        pixel_pitch: float
    ) -> np.ndarray:
        """
        Computes the physical wave-optical Point Spread Function (PSF) |h(x, y)|^2
        via Fourier transformation of the generalized pupil function.
        """
        # Normalized pupil coordinates [-1.0, 1.0]
        y_coords = np.linspace(-1.0, 1.0, kernel_size)
        x_coords = np.linspace(-1.0, 1.0, kernel_size)
        U, V = np.meshgrid(x_coords, y_coords)
        R = np.sqrt(U**2 + V**2)

        # Aperture transmission P(u, v)
        if self.num_blades <= 0:
            pupil = np.clip((1.0 - R) / 0.05 + 0.5, 0.0, 1.0)
        else:
            theta = np.arctan2(V, U)
            blade_angle = 2.0 * np.pi / self.num_blades
            poly_r = np.cos(blade_angle / 2.0) / np.cos(np.mod(theta, blade_angle) - blade_angle / 2.0)
            pupil = np.clip((poly_r - R) / 0.05 + 0.5, 0.0, 1.0)

        # Defocus wave aberration: W_20 * r^2 in waves
        # Defocus aberration coefficient: W_20 = (Delta_z) / (8 * N^2 * lambda)
        w20 = (defocus_mm * 1e-3) / (8.0 * (f_number**2) * wavelength + 1e-18)
        # Cap w20 to prevent extreme phase wrapping in discrete grid
        w20 = np.clip(w20, -12.0, 12.0)
        
        aberration_phase = 2.0 * np.pi * w20 * (R**2)
        generalized_pupil = pupil * np.exp(1j * aberration_phase)

        # Coherent impulse response h(x, y) = IFFT2(P)
        h = fft.fftshift(fft.ifft2(fft.ifftshift(generalized_pupil)))
        
        # Incoherent PSF = |h|^2
        psf = np.real(h * np.conj(h))
        
        # Normalize sum to 1.0 (energy conservation)
        psf_sum = np.sum(psf)
        if psf_sum > 0:
            psf /= psf_sum
        return psf

    def capture_refocused(
        self,
        wave_field: WaveField,
        defocus_distance: float = 0.0,
        f_number_override: Optional[float] = None,
        depth_gradient: bool = False
    ) -> WaveField:
        """
        Simulates optical capture with wave-optical depth of field.
        
        Args:
            wave_field: Input WaveField object.
            defocus_distance: Overall defocus shift in millimeters.
            f_number_override: Optional f-stop (e.g. 1.4, 2.8, 8.0).
            depth_gradient: If True, simulates natural depth variation across the vertical frame.
        """
        f_num = self.lens_f_number if hasattr(self, "lens_f_number") else (
            f_number_override if f_number_override is not None else self.f_number
        )

        h, w = wave_field.shape
        pitch = wave_field.pixel_pitch
        raw_intensity = wave_field.intensity()  # (3, H, W)
        output_intensity = np.zeros_like(raw_intensity)

        # In-focus fast path: identity transmission
        if abs(defocus_distance) < 1e-6 and not depth_gradient:
            return WaveField(
                complex_field=wave_field.complex_field.copy(),
                pixel_pitch=pitch,
                wavelengths=wave_field.wavelengths
            )

        # Kernel dimension (odd number)
        kernel_size = 65

        for c in range(wave_field.channels):
            wl = wave_field.wavelengths[c]
            
            if not depth_gradient:
                psf = self.compute_wave_psf(
                    kernel_size=kernel_size,
                    f_number=f_num,
                    defocus_mm=defocus_distance,
                    wavelength=wl,
                    pixel_pitch=pitch
                )
                
                # Symmetrical reflection padding to prevent boundary reflection artifacts
                pad_h = kernel_size // 2
                pad_w = kernel_size // 2
                channel_img = raw_intensity[c]
                padded = np.pad(channel_img, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
                
                # 2D FFT convolution via OTF
                psf_padded = np.zeros_like(padded)
                sy = (padded.shape[0] - kernel_size) // 2
                sx = (padded.shape[1] - kernel_size) // 2
                psf_padded[sy:sy+kernel_size, sx:sx+kernel_size] = psf
                
                # Frequency domain multiplication
                otf = fft.fft2(fft.ifftshift(psf_padded))
                img_freq = fft.fft2(padded)
                conv_res = np.real(fft.ifft2(img_freq * otf))
                
                # Unpad
                output_intensity[c] = conv_res[pad_h:pad_h+h, pad_w:pad_w+w]
            else:
                # Continuous depth plane sweep across image height
                # In park photo: bottom is close (y=H), middle is bench (y=0.5H), top is trees (y=0)
                y_norm = np.linspace(-1.0, 1.0, h)[:, np.newaxis]  # top=-1, bottom=+1
                # Defocus varies smoothly with depth
                # When focused on bench (defocus_distance=0), bench is sharp, top is blurred, bottom mildly blurred
                depth_defocus = defocus_distance + y_norm * 4.0
                
                # Multi-plane depth slicing (3 layers for high quality)
                accum = np.zeros((h, w), dtype=np.float64)
                weight_sum = np.zeros((h, w), dtype=np.float64)
                
                sample_defocuses = [defocus_distance - 4.0, defocus_distance, defocus_distance + 4.0]
                for s_defocus in sample_defocuses:
                    psf = self.compute_wave_psf(
                        kernel_size=kernel_size,
                        f_number=f_num,
                        defocus_mm=s_defocus,
                        wavelength=wl,
                        pixel_pitch=pitch
                    )
                    pad_h = kernel_size // 2
                    pad_w = kernel_size // 2
                    padded = np.pad(raw_intensity[c], ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
                    psf_padded = np.zeros_like(padded)
                    sy = (padded.shape[0] - kernel_size) // 2
                    sx = (padded.shape[1] - kernel_size) // 2
                    psf_padded[sy:sy+kernel_size, sx:sx+kernel_size] = psf
                    
                    otf = fft.fft2(fft.ifftshift(psf_padded))
                    conv = np.real(fft.ifft2(fft.fft2(padded) * otf))[pad_h:pad_h+h, pad_w:pad_w+w]
                    
                    # Gaussian weighting by depth proximity
                    diff = depth_defocus - s_defocus
                    w_mask = np.exp(-0.5 * (diff / 2.5)**2)
                    accum += conv * w_mask
                    weight_sum += w_mask
                    
                output_intensity[c] = accum / np.maximum(weight_sum, 1e-12)

        # Synthesize complex field from output intensity preserving wave phase
        output_intensity = np.clip(output_intensity, 0.0, 1.0)
        output_amplitude = np.sqrt(output_intensity)
        output_complex = output_amplitude * np.exp(1j * wave_field.phase())

        return WaveField(
            complex_field=output_complex,
            pixel_pitch=pitch,
            wavelengths=wave_field.wavelengths
        )
