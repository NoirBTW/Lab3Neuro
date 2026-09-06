"""Одинаковая детерминированная предобработка для оценки и приложения."""
import numpy as np
from PIL import Image, ImageOps


def preprocess(image, config):
    image = ImageOps.exif_transpose(image).convert('RGB')
    size = config['image_size']
    resized = ImageOps.fit(image, (size, size), method=Image.Resampling.BICUBIC)
    x = np.asarray(resized, dtype=np.float32) / 255.0
    x = (x - np.array(config['mean'], dtype=np.float32)) / np.array(config['std'], dtype=np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1), dtype=np.float32)
