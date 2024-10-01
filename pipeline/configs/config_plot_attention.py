import os
from typing import List, Optional, Callable, Tuple, Dict, Literal, Set, Union

from dataclasses import dataclass
from typing import Tuple


@dataclass
class Config:
    model_path: str
    model_alias: str
    save_dir: str
    # for generation_trajectory

    def artifact_path(self) -> str:
        save_dir = self.save_dir
        # return os.path.join(os.path.dirname(os.path.realpath(__file__)), "runs", "activation_pca", self.model_alias)
        return os.path.join(save_dir, "runs", "activation_pca", self.model_alias)
