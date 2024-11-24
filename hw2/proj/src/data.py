import numpy as np
from pupok import cosine_similarity
import torch
import lightning.pytorch as pl
import pandas as pd
from typing import Tuple, Optional


class MNISTDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe: pd.DataFrame):
        super().__init__()
        self.dataframe = dataframe

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        label = self.dataframe.iloc[idx]['label']
        picture = self.dataframe.iloc[idx].drop('label').to_numpy(dtype='float32')
        picture = torch.tensor(picture, dtype=torch.float32)
        picture = picture / 127.5 - 1.0

        # ???????????????????????????????????????????????????????????????????????
        assert np.allclose(cosine_similarity(list(picture[:2]), list(picture[:2])), 1,  atol=1e-3)
        # ???????????????????????????????????????????????????????????????????????

        return picture, label


class MNISTModule(pl.LightningDataModule):
    def __init__(
            self,
            dataloader_num_workers: int,
            batch_size: int,
            train_path_csv: str,
            val_path_csv: Optional[str] = None,
    ):
        super().__init__()
        self.train_path_csv = train_path_csv
        self.val_path_csv = val_path_csv
        self.dataloader_num_workers = dataloader_num_workers
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None):
        train_df = pd.read_csv(self.train_path_csv)

        self.train_dataset = MNISTDataset(
            dataframe=train_df,
        )

        if self.val_path_csv:
            val_df = pd.read_csv(self.val_path_csv)

            self.val_dataset = MNISTDataset(
                dataframe=val_df,
            )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.dataloader_num_workers,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.dataloader_num_workers,
        )
