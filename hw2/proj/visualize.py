import hydra
from omegaconf import DictConfig
import torch
import matplotlib.pyplot as plt

from src.model import GAN


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    model = GAN(cfg)

    if cfg.init_weights:
        ckpt = torch.load(cfg.init_weights, map_location="cpu")
        model.load_state_dict(
            {k: v for k, v in ckpt["state_dict"].items() if "total" not in k}
        )

    picture = model().detach().numpy().reshape((28, 28))
    plt.figure(figsize=(8, 6))
    plt.imshow(picture)
    plt.savefig(cfg.save_image, dpi=300)


if __name__ == "__main__":
    main()
