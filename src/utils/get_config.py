import hydra
from omegaconf import OmegaConf, DictConfig


@hydra.main(version_base=None, config_path="conf", config_name="config")
def get_config(cfg: DictConfig):
    config = OmegaConf.to_object(cfg)
    return config

