"""SAM v1 runner placeholder.

Citylens currently standardizes on SAM2; this module is intentionally unused.
It is kept to preserve the package layout for potential future migration work.
"""

from __future__ import annotations

import numpy as np


def run_sam1_auto_mask(image_rgb: np.ndarray) -> np.ndarray:
    raise NotImplementedError("SAM v1 is not supported in citylens-core (use segmentation_backend='sam2')")
