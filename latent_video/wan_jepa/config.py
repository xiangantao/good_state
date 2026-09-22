from dataclasses import dataclass

from ..classification.config import RunConfig as BaseConfig


@dataclass(frozen=True)
class RunConfig(BaseConfig):
    jepa_model_path: str = "../models/vjepa2/vitl.pt"
    artifact_dir: str = "latent_video/wan_jepa/artifacts"

    def __post_init__(self):
        super().__post_init__()
        if self.clip.resolution != (256, 256) or self.clip.grid != (16, 16):
            raise ValueError("Wan-JEPA requires256 RGB and native16x16 spatial grid")
        if self.num_heads != 16 or self.num_probe_blocks != 4:
            raise ValueError("Validated classifier uses depth4 and16 heads")
        if self.data.crop_size != 256 or self.data.num_segments != 2:
            raise ValueError("Require256 crop and2 segments")
        if self.data.frame_step != 4 or self.data.num_views_per_segment != 3:
            raise ValueError("Require frame_step4 and3 evaluation views")
