from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from typing import List, Tuple, Union
from pytorch_lightning import LightningDataModule

class ImageClassificationDataModule(LightningDataModule):
    def __init__(self, root_dir, image_dim: Union[int, List, Tuple]):
        super().__init__()

        self.save_hyperparameters()

    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        pass

    def setup(self, stage):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        pass

    def train_dataloader(self):
        train_split = Dataset(...)
        return DataLoader(train_split)

    def val_dataloader(self):
        val_split = Dataset(...)
        return DataLoader(val_split)

    def test_dataloader(self):
        test_split = Dataset(...)
        return DataLoader(test_split)
   
    def teardown(self):
        # clean up after fit or test
        # called on every process in DDP
        pass

class VideoClassificationDataModule(LightningDataModule):
    def __init__(self, root_dir, image_dim: Union[int, List, Tuple]):
        super().__init__()

        self.save_hyperparameters()

    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        pass

    def setup(self, stage):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        pass

    def train_dataloader(self):
        train_split = Dataset(...)
        return DataLoader(train_split)

    def val_dataloader(self):
        val_split = Dataset(...)
        return DataLoader(val_split)

    def test_dataloader(self):
        test_split = Dataset(...)
        return DataLoader(test_split)
   
    def teardown(self):
        # clean up after fit or test
        # called on every process in DDP
        pass

class ObjectDetectionDataModule(LightningDataModule):
    def __init__(self, root_dir, image_dim: Union[int, List, Tuple]):
        super().__init__()

        self.save_hyperparameters()

    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        pass

    def setup(self, stage):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        pass

    def train_dataloader(self):
        train_split = Dataset(...)
        return DataLoader(train_split)

    def val_dataloader(self):
        val_split = Dataset(...)
        return DataLoader(val_split)

    def test_dataloader(self):
        test_split = Dataset(...)
        return DataLoader(test_split)
   
    def teardown(self):
        # clean up after fit or test
        # called on every process in DDP
        pass

class ImageTransformationDataModule(LightningDataModule):
    def __init__(self, root_dir, image_dim: Union[int, List, Tuple]):
        super().__init__()

        self.save_hyperparameters()

    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        pass

    def setup(self, stage):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        pass

    def train_dataloader(self):
        train_split = Dataset(...)
        return DataLoader(train_split)

    def val_dataloader(self):
        val_split = Dataset(...)
        return DataLoader(val_split)

    def test_dataloader(self):
        test_split = Dataset(...)
        return DataLoader(test_split)
   
    def teardown(self):
        # clean up after fit or test
        # called on every process in DDP
        pass

class FaceRecognitionDataModule(LightningDataModule):
    def __init__(self, root_dir, image_dim: Union[int, List, Tuple]):
        super().__init__()

        self.save_hyperparameters()

    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        pass

    def setup(self, stage):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        pass

    def train_dataloader(self):
        train_split = Dataset(...)
        return DataLoader(train_split)

    def val_dataloader(self):
        val_split = Dataset(...)
        return DataLoader(val_split)

    def test_dataloader(self):
        test_split = Dataset(...)
        return DataLoader(test_split)
   
    def teardown(self):
        # clean up after fit or test
        # called on every process in DDP
        pass