"""Frequency Domain Utilities for Deepfake & Synthetic Image Forensics.

Extracts 2D Fast Fourier Transform (FFT) log-magnitude spectra and applies
spectral and noise-residual analysis to detect generative manipulation artifacts.
"""

import io
import base64
import numpy as np
import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms


# Standard ImageNet normalization for Spatial ConvNeXt Backbone
RGB_NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)


def letterbox_image(
    img: Image.Image,
    target_size: tuple[int, int] = (224, 224),
    pad_color: tuple[int, int, int] = (0, 0, 0)
) -> Image.Image:
    """Resizes a PIL Image preserving aspect ratio with centered padding."""
    w, h = img.size
    target_w, target_h = target_size
    if w == target_w and h == target_h:
        return img

    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resample_mode = getattr(Image, 'Resampling', Image).LANCZOS
    resized = img.resize((new_w, new_h), resample_mode)

    new_img = Image.new("RGB", target_size, pad_color)
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    new_img.paste(resized, (paste_x, paste_y))
    return new_img


def letterbox_gray(
    gray_arr: np.ndarray,
    target_size: tuple[int, int] = (224, 224)
) -> np.ndarray:
    """Resizes a 2D grayscale array preserving aspect ratio with zero padding."""
    h, w = gray_arr.shape[:2]
    target_h, target_w = target_size
    if h == target_h and w == target_w:
        return gray_arr.astype(np.float32)

    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(gray_arr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros(target_size, dtype=np.float32)
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    canvas[paste_y:paste_y + new_h, paste_x:paste_x + new_w] = resized
    return canvas


def load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """Decodes raw image bytes into a BGR numpy ndarray."""
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        # Fallback to PIL in case of unconventional headers/formats
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return img


def extract_fft_spectrum(
    image_bytes: bytes,
    target_size: tuple[int, int] = (224, 224),
    cutoff_radius: int = 15
) -> tuple[torch.Tensor, str]:
    """Extracts 2D-FFT log-magnitude spectrum with letterboxing and high-pass circular filter.

    Args:
        image_bytes: Raw bytes of the input image.
        target_size: (Height, Width) to resize image for frequency analysis.
        cutoff_radius: Radius in pixels around center DC frequency to filter out for UI view.

    Returns:
        spectrum_tensor: PyTorch tensor with shape (1, 1, H, W), normalized in [0, 1].
        spectrum_base64: Base64-encoded PNG data string formatted for UI rendering.
    """
    # 1. Read raw image bytes and convert to grayscale with aspect-ratio letterboxing
    img_bgr = load_image_from_bytes(image_bytes)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_resized = letterbox_gray(gray, target_size=target_size)

    # 2. Subtract mean and apply 2D Hann window to eliminate artificial edge leakage
    norm_gray = gray_resized - np.mean(gray_resized)
    hann_2d = np.outer(np.hanning(target_size[0]), np.hanning(target_size[1])).astype(np.float32)
    windowed = norm_gray * hann_2d

    # 3. Compute 2D-FFT and shift low frequencies to center
    f_transform = np.fft.fft2(windowed)
    f_shift = np.fft.fftshift(f_transform)

    # 4. Compute log-magnitude spectrum
    magnitude_spectrum = 20.0 * np.log(np.abs(f_shift) + 1e-8)

    # 5. Min-max normalization for neural network input [0, 1] matching training pipeline
    spec_min = np.min(magnitude_spectrum)
    spec_max = np.max(magnitude_spectrum)
    if spec_max - spec_min > 1e-8:
        normalized_spectrum = (magnitude_spectrum - spec_min) / (spec_max - spec_min)
    else:
        normalized_spectrum = np.zeros_like(magnitude_spectrum, dtype=np.float32)

    # Convert to PyTorch 1-channel tensor: shape (1, 1, H, W)
    spectrum_tensor = torch.from_numpy(normalized_spectrum).float().unsqueeze(0).unsqueeze(0)

    # 6. Apply high-pass circular filter mask for high-contrast UI display
    h, w = magnitude_spectrum.shape
    center_y, center_x = h // 2, w // 2
    y_grid, x_grid = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2)
    high_pass_mask = (dist_from_center > cutoff_radius).astype(np.float32)
    vis_spectrum = magnitude_spectrum * high_pass_mask

    v_min, v_max = np.min(vis_spectrum), np.max(vis_spectrum)
    if v_max - v_min > 1e-8:
        vis_norm = ((vis_spectrum - v_min) / (v_max - v_min) * 255.0).astype(np.uint8)
    else:
        vis_norm = (normalized_spectrum * 255.0).astype(np.uint8)

    vis_colored = cv2.applyColorMap(vis_norm, cv2.COLORMAP_INFERNO)

    success, buffer = cv2.imencode('.png', vis_colored)
    if not success:
        raise RuntimeError("Failed to encode FFT spectrum visualization as PNG.")

    spectrum_b64 = base64.b64encode(buffer).decode('utf-8')
    spectrum_data_url = f"data:image/png;base64,{spectrum_b64}"

    return spectrum_tensor, spectrum_data_url


def preprocess_rgb_image(image_bytes: bytes) -> torch.Tensor:
    """Preprocesses raw image bytes for the RGB Spatial Backbone with aspect-ratio preservation.

    Returns:
        PyTorch tensor of shape (1, 3, 224, 224).
    """
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    letterboxed = letterbox_image(pil_img, target_size=(224, 224))
    to_tensor = transforms.ToTensor()
    tensor = RGB_NORMALIZE(to_tensor(letterboxed)).unsqueeze(0)
    return tensor


def compute_frequency_forensic_metrics(img_bgr: np.ndarray) -> dict:
    """Computes dynamic, rigorous frequency-domain and noise-residual forensic metrics.

    Analyzes 2D-FFT spectral peakiness, periodic harmonic grid spikes in noise residue,
    power-law spectral decay linearity, and diffusion latent smoothing to
    distinguish generative diffusion/GAN images from authentic camera sensors.

    Args:
        img_bgr: Decoded image array in BGR format.

    Returns:
        Dictionary containing calibrated ai_probability, confidence, verdict,
        artifact_flags, and diagnostic measurements.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape

    # 1. Standardized 224x224 analysis frame
    gray_224 = cv2.resize(gray, (224, 224), interpolation=cv2.INTER_AREA)

    # 2. Windowed 2D Fast Fourier Transform & Centered Log-Magnitude Spectrum
    hann_2d = np.outer(np.hanning(224), np.hanning(224))
    norm_gray = gray_224 - np.mean(gray_224)
    windowed_gray = norm_gray * hann_2d

    f_transform = np.fft.fft2(windowed_gray)
    f_shift = np.fft.fftshift(f_transform)
    mag_spectrum = 20.0 * np.log(np.abs(f_shift) + 1e-8)

    cy, cx = 112, 112
    y_grid, x_grid = np.ogrid[:224, :224]
    dist_map = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)

    # 3. High-Frequency Annulus Statistics (radius 18 to 100 px)
    hf_mask = (dist_map >= 18) & (dist_map <= 100)
    hf_vals = mag_spectrum[hf_mask]

    hf_mean = float(np.mean(hf_vals))
    hf_std = float(np.std(hf_vals)) + 1e-6
    hf_max = float(np.max(hf_vals))
    hf_median = float(np.median(hf_vals))

    # Peak Z-Score (Outlier spike intensity vs background noise floor)
    peak_zscore = float((hf_max - hf_mean) / hf_std)

    # Top 1% vs Median (Measures energy concentration in outlier frequency bins)
    sorted_hf = np.sort(hf_vals)
    top_1pct = sorted_hf[-max(1, int(len(sorted_hf) * 0.01)):]
    top_1pct_diff = float(np.mean(top_1pct) - hf_median)

    # 4. Periodic Harmonic Grid Spike Detection via High-Pass Noise Residue Autocorrelation
    # Denoise with 3x3 median filter to extract high-frequency noise/texture residue
    blurred = cv2.medianBlur(np.uint8(np.clip(gray_224, 0, 255)), 3).astype(np.float32)
    residue = gray_224 - blurred
    residue_norm = residue - np.mean(residue)

    # 2D Autocorrelation of noise residue
    res_fft = np.fft.fftshift(np.fft.fft2(residue_norm * hann_2d))
    res_power = np.abs(res_fft) ** 2
    res_autocorr = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(res_power)).real)
    center_energy = res_autocorr[112, 112] + 1e-8
    res_autocorr_norm = res_autocorr / center_energy

    # Test for periodic upsampling peaks at harmonic strides (e.g. 4, 8, 16 pixels)
    grid_spike_score = 0.0
    for dy in [-16, -8, 8, 16]:
        for dx in [-16, -8, 8, 16]:
            if dx != 0 or dy != 0:
                corr_val = abs(float(res_autocorr_norm[112 + dy, 112 + dx]))
                # Natural camera noise autocorrelation drops to near 0 off-center (< 0.06).
                # Generative periodic checkerboards/grids create distinct harmonic peaks.
                if corr_val > 0.06:
                    grid_spike_score += (corr_val - 0.05) * 400.0

    # 5. Radial Power Law Decay Linearity (Tests adherence to natural 1/f^alpha optics)
    r_radii = np.arange(10, 105, 3)
    radial_energies = []
    for r in r_radii:
        r_mask = (dist_map >= r - 1.5) & (dist_map <= r + 1.5)
        if np.any(r_mask):
            radial_energies.append(float(np.mean(mag_spectrum[r_mask])))
        else:
            radial_energies.append(0.0)
    radial_energies = np.array(radial_energies)
    log_r = np.log(r_radii)

    corr_matrix = np.corrcoef(log_r, radial_energies)
    r_squared = float(corr_matrix[0, 1] ** 2) if not np.isnan(corr_matrix[0, 1]) else 0.5
    slope = float(np.polyfit(log_r, radial_energies, 1)[0])

    # 6. Spatial Residual Variance & Diffusion Latent Smoothing Analysis
    laplacian = cv2.Laplacian(gray_224, cv2.CV_32F)
    patch_vars = []
    for py in range(0, 224, 16):
        for px in range(0, 224, 16):
            p = laplacian[py:py + 16, px:px + 16]
            patch_vars.append(float(np.var(p)))
    patch_vars = np.array(patch_vars)
    patch_median = float(np.median(patch_vars))
    # Fraction of patches that are abnormally flat/smoothed
    smooth_patch_ratio = float(np.mean(patch_vars < max(0.8, patch_median * 0.15))) if patch_median > 0.5 else 0.0

    # 7. Calibrated Dynamic Forensic Score
    score = 0.0

    # Grid Spikes (primary cue for GAN / transpose convolution upsampling)
    if grid_spike_score > 20.0:
        score += min(45.0, grid_spike_score * 0.8)
    elif grid_spike_score > 5.0:
        score += (grid_spike_score - 5.0) * 1.5

    # Diffusion Latent Smoothing
    if smooth_patch_ratio > 0.35:
        score += min(35.0, (smooth_patch_ratio - 0.30) * 70.0)
    elif smooth_patch_ratio > 0.18:
        score += (smooth_patch_ratio - 0.18) * 30.0

    # Spectral Peak Z-Score
    if peak_zscore > 4.5:
        score += min(25.0, (peak_zscore - 4.0) * 12.0)
    elif peak_zscore > 3.8:
        score += (peak_zscore - 3.5) * 6.0

    # Spectral Outlier Energy Concentration
    if top_1pct_diff > 90.0:
        score += min(20.0, (top_1pct_diff - 80.0) * 0.4)

    # Natural camera sensor optics bonus: Smooth 1/f roll-off with clean noise floor
    if r_squared > 0.85 and slope < -2.0 and grid_spike_score < 5.0 and smooth_patch_ratio < 0.20:
        score -= 28.0
    elif r_squared > 0.75 and slope < -1.5 and grid_spike_score < 10.0:
        score -= 12.0

    ai_prob = float(np.clip(18.0 + score, 4.0, 98.0))
    is_ai = ai_prob >= 50.0

    # 8. Artifact Flag Cues Detection
    artifact_flags = []
    if is_ai:
        if grid_spike_score > 15.0:
            artifact_flags.append("Upsampling Grid Anomaly")
        if smooth_patch_ratio > 0.30:
            artifact_flags.append("Diffusion Latent Smoothing")
        if peak_zscore > 4.2 or top_1pct_diff > 85.0:
            artifact_flags.append("Frequency Domain Residual Spike")
        if r_squared < 0.70:
            artifact_flags.append("Spectral Decay Distortion")
        if not artifact_flags:
            artifact_flags.append("Generative Spectral Signature")

    return {
        "ai_probability": round(ai_prob, 2),
        "confidence": round(max(ai_prob, 100.0 - ai_prob), 2),
        "verdict": "AI-GENERATED" if is_ai else "AUTHENTIC SENSOR CAPTURE",
        "artifact_flags": artifact_flags,
        "metrics": {
            "peak_zscore": round(peak_zscore, 2),
            "top_1pct_diff": round(top_1pct_diff, 2),
            "grid_spike_score": round(grid_spike_score, 1),
            "smooth_patch_ratio": round(smooth_patch_ratio, 2),
            "r_squared": round(r_squared, 3)
        }
    }
