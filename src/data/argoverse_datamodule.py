import torch
import pytorch_lightning as pl

from torch.utils.data import DataLoader, Dataset

from src.data.argoverse_dataset import ArgoverseDataset


class ArgoverseDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    @staticmethod
    def add_data_specific_args(parent_parser):
        # Add dataset-specific config args here
        return parent_parser

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        pass

    def get_dataloader(self, mode, cfg):
        ds = ArgoverseDataset(self.cfg.dataroot, mode=mode, delta=self.cfg.predict_delta,
                              map_features_flag=self.cfg.map_features,
                              social_features_flag=True, heuristic=(not self.cfg.no_heuristic),
                              ifc=self.cfg.IFC, is_oracle=self.cfg.use_oracle)

        shuffle = False if mode == 'val' or mode == 'test' else True
        drop_last = shuffle
        dataloader = DataLoader(ds, batch_size=self.cfg.batch_size, num_workers=self.cfg.workers,
                                pin_memory=True, collate_fn=ArgoverseDataset.collate,
                                shuffle=shuffle, drop_last=drop_last)
        return dataloader

    def train_dataloader(self):
        train_split = 'trainval' if self.cfg.mode == 'trainval' else 'train'
        return self.get_dataloader(train_split, self.cfg)

    def val_dataloader(self):
        return self.get_dataloader('val', self.cfg)

    def test_dataloader(self):
        return self.get_dataloader('test', self.cfg)
