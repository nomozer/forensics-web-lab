import io
import random
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np

class RealisticDegradationPipeline:
    """
    Applies realistic social media transmission and compression degradations
    without destroying the underlying semantic class label.
    """

    def __init__(
        self,
        jpeg_prob: float = 0.6,
        resize_prob: float = 0.35,
        blur_prob: float = 0.25,
        noise_prob: float = 0.25,
        seed: int = 42,
    ):
        self.jpeg_prob = jpeg_prob
        self.resize_prob = resize_prob
        self.blur_prob = blur_prob
        self.noise_prob = noise_prob
        self.rng = random.Random(seed)

    def apply_jpeg_compression(self, img: Image.Image, quality: int) -> Image.Image:
        buffer = io.BytesIO()
        img.convert("RGB").save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer)

    def apply_resize_degradation(self, img: Image.Image, scale: float) -> Image.Image:
        orig_w, orig_h = img.size
        new_w = max(16, int(orig_w * scale))
        new_h = max(16, int(orig_h * scale))
        down = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
        return down.resize((orig_w, orig_h), Image.Resampling.BICUBIC)

    def apply_noise(self, img: Image.Image, sigma: float = 5.0) -> Image.Image:
        arr = np.array(img, dtype=np.float32)
        noise = np.random.normal(0, sigma, arr.shape)
        noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy)

    def __call__(self, img: Image.Image) -> Image.Image:
        out = img

        # 1. Random resize downsampling
        if self.rng.random() < self.resize_prob:
            scale = self.rng.uniform(0.5, 0.85)
            out = self.apply_resize_degradation(out, scale)

        # 2. Gaussian blur or slight sharpen
        if self.rng.random() < self.blur_prob:
            if self.rng.random() < 0.5:
                out = out.filter(ImageFilter.GaussianBlur(radius=self.rng.uniform(0.5, 1.2)))
            else:
                enhancer = ImageEnhance.Sharpness(out)
                out = enhancer.enhance(self.rng.uniform(1.2, 1.8))

        # 3. Sensor noise
        if self.rng.random() < self.noise_prob:
            out = self.apply_noise(out, sigma=self.rng.uniform(3.0, 8.0))

        # 4. Multi-pass JPEG recompression
        if self.rng.random() < self.jpeg_prob:
            q = int(self.rng.uniform(45, 95))
            out = self.apply_jpeg_compression(out, q)

        return out
