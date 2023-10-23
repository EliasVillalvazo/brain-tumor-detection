import os
import torch
import shutil
from models.resnet50 import ResNet50
from models.alexnet import AlexNet

from dataloader.tumor_dataloader import TumorDataModule

import mlflow.pytorch
import argparse

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint


parser = argparse.ArgumentParser(description="Tumor Dataset Parameters")

parser.add_argument(
    "--use_pretrained_model",
    default=True,
    metavar="N",
    help="Use pretrained model or train from the scratch",
)

parser.add_argument(
    "--max_epochs",
    type=int,
    default=200,
    metavar="N",
    help="Number of epochs to be used for training",
)

parser.add_argument(
    "--lr",
    type=float,
    default=0.01,
    metavar="LR",
    help="learning rate (default: 0.1)",
)


def main():
    args = parser.parse_args()

    early_stopping = EarlyStopping(
        monitor="val_loss",
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.getcwd(), save_top_k=1, verbose=True, monitor="val_loss", mode="min"
    )

    lr_logger = LearningRateMonitor()
    print(torch.cuda.is_available())
    mlflow.pytorch.autolog()
    dm = TumorDataModule(path=r'C:\Users\15B38LA\Downloads\brain_tumor\brain_mri_scan_images',
                         batch_size=4, size=224)
    dm.setup(stage="fit")
    # model = ResNet50(args.lr, pretrained=args.use_pretrained_model, num_classes=2, seed=1020)
    model = AlexNet(args.lr, pretrained=args.use_pretrained_model, num_classes=2, seed=1020)
    trainer = L.Trainer(max_epochs=args.max_epochs, callbacks=[early_stopping, checkpoint_callback, lr_logger])
    trainer.fit(model, dm)
    trainer.test(datamodule=dm)
    if os.path.exists(os.path.join(os.getcwd(),"test5")):
        shutil.rmtree(os.path.join(os.getcwd(),"test5"))
    mlflow.pytorch.save_model(trainer.lightning_module, os.path.join(os.getcwd(),"test5"))


if __name__ == '__main__':
    main()
