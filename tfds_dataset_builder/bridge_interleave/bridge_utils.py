from __future__ import annotations

import numpy as np
from PIL import Image

def resize(img_array: Image | np.ndarray, target_size: tuple = (224, 224)) -> np.ndarray:
    if isinstance(img_array, np.ndarray):
        img = Image.fromarray(img_array).convert('RGB')
    else:
        img = img_array
    img = img.resize(target_size) # different aspect ratio
    return np.array(img)