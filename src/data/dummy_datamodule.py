import torch
import pytorch_lightning as pl

from torch.utils.data import DataLoader, Dataset


class DummyDataModule(pl.LightningDataModule):
    def __init__(self, x_dims_list, y_dims, batch_size=32, n_epochs=100):
        super().__init__()
        self.x_dims_list = x_dims_list
        self.y_dims = y_dims
        self.batch_size = batch_size
        self.n_epochs = n_epochs

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.dummy_dataset = DummyDataset(self.batch_size * self.n_epochs,
                                          self.x_dims_list, self.y_dims)

    def train_dataloader(self):
        return DataLoader(self.dummy_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.dummy_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.dummy_dataset, batch_size=self.batch_size)


class DummyDataset(Dataset):
    def __init__(self, n, x_dims_list, y_dims):
        self.x_list = [torch.rand((n,) + x_dims) for x_dims in x_dims_list]
        self.y = torch.rand((n,) + y_dims)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return [x[idx] for x in self.x_list], self.y[idx]
