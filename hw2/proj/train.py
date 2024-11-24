import os
import hydra
from omegaconf import DictConfig
import lightning.pytorch as pl

from src.model import GAN
from src.data import MNISTModule


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    pl.seed_everything(42)

    dm = MNISTModule(
        dataloader_num_workers=cfg.data.dataloader_num_workers,
        batch_size=cfg.data.batch_size,
        train_path_csv=cfg.data.train_path_csv,
        val_path_csv=cfg.data.val_path_csv,
    )
    model = GAN(cfg)

    loggers = [pl.loggers.TensorBoardLogger(cfg.logger.tensorboard.path, name=cfg.logger.tensorboard.name)]
    callbacks = []

    if cfg.artifacts.checkpoint.use:
        callbacks = [(
            pl.callbacks.ModelCheckpoint(
                dirpath=os.path.join(cfg.artifacts.checkpoint.dirpath, cfg.artifacts.experiment_name),
                filename=cfg.artifacts.checkpoint.filename,
                monitor=cfg.artifacts.checkpoint.monitor,
                save_top_k=cfg.artifacts.checkpoint.save_top_k,
                every_n_train_steps=cfg.artifacts.checkpoint.every_n_train_steps,
                every_n_epochs=cfg.artifacts.checkpoint.every_n_epochs,
            )
        )]

    trainer = pl.Trainer(
        accelerator=cfg.train.accelerator,
        max_epochs=cfg.train.num_epochs,
        val_check_interval=cfg.train.val_check_interval,
        log_every_n_steps=cfg.train.log_every_n_steps,
        enable_checkpointing=cfg.artifacts.checkpoint.use,
        logger=loggers,
        callbacks=callbacks,
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
