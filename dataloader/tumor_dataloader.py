import os
import lightning as L

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split


class TumorDataModule(L.LightningDataModule):
    def __init__(self, path, size,  batch_size=64, num_workers=0):
        """
        Initialization of inherited lightning data module
        """
        super().__init__()
        self.path = path
        self.size = size
        self.df_train = None
        self.df_val = None
        self.df_test = None
        self.train_data_loader = None
        self.val_data_loader = None
        self.test_data_loader = None
        self.batch_size = batch_size
        self.num_workers = num_workers

        # transforms for images
        self.train_transforms = transforms.Compose([transforms.ToTensor(),
                                                    transforms.RandomChoice([
                                                        transforms.RandomHorizontalFlip(p=0.5),
                                                        transforms.RandomVerticalFlip(p=0.5),
                                                        transforms.RandomRotation(degrees=(-180, 180)),
                                                        transforms.RandomPerspective(distortion_scale=0.4, p=0.5),
                                                        transforms.RandomAffine(degrees=(-90, 90), translate=(0, 0.2),
                                                                                scale=[0.5, 1]),
                                                        transforms.Pad(50, fill=0, padding_mode="symmetric"),
                                                    ]),
                                                    transforms.Resize((self.size, self.size))
                                                    ])

        self.test_transforms = transforms.Compose([transforms.ToTensor(), transforms.Resize((self.size, self.size))])

    def setup(self, stage=None):
        """
        Downloads the data, parse it and split the data into train, test, validation data

        :param stage: Stage - training or testing
        """
        self.df_train = datasets.ImageFolder(root=os.path.join(self.path, "train"), transform=self.train_transforms)
        self.df_test = datasets.ImageFolder(root=os.path.join(self.path, "test"), transform=self.test_transforms)
        self.df_train, self.df_val = random_split(self.df_train, [0.8, 0.2])

    def create_data_loader(self, df, persistence=False):
        """
        Generic data loader function

        :param df: Input tensor

        :return: Returns the constructed data loader
        """
        return DataLoader(df, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=persistence)

    def train_dataloader(self):
        """
        :return: output - Train data loader for the given input
        """
        return self.create_data_loader(self.df_train, persistence=False)

    def val_dataloader(self):
        """
        :return: output - Validation data loader for the given input
        """
        return self.create_data_loader(self.df_val)

    def test_dataloader(self):
        """
        :return: output - Test data loader for the given input
        """
        return self.create_data_loader(self.df_test)
