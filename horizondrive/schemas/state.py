from typing import Optional

import torch
import numpy as np


class State:
    def __init__(self):
        self.weight_dtype: torch.dtype = torch.float32
        self.using_deepspeed: bool = False
        self.zero_stage: int = 0
        self.rng: Optional[np.random.Generator] = None
        self.torch_rng: Optional[torch.Generator] = None
