import torch
import wandb
import utils
import att_resnet
import dataset
import random_resized_crop

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.models as models
from pytorch_lightning.loggers import WandbLogger


from torch import nn
from torch import optim
from torch.utils.data import Dataset
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms
from pytorch_lightning.callbacks import ModelCheckpoint

# Base Model, Dataset, Batch Size, Learning Rate
# wandb.init(project="contrastive_learning", entity="natethom")
wandb_logger = WandbLogger(name='resnet50_hair_pretrain_randomResizedCrop', project='contrastive_learning', entity='unr-mpl')

activation = None

if __name__=="__main__":
    args = utils.get_args()

    pl.seed_everything(args.random_seed)

    # Initialize the model
    if args.load == True:
        net = att_resnet.Att_Resnet.load_from_checkpoint(args.load_path + args.load_file, hparams=args)
    else:
        net = att_resnet.Att_Resnet(args)

    train_dataset = dataset.Att_Dataset(
        "training",
        args.image_path,
        args.image_dir,
        args.attr_label_path,
        args.attr_to_use,
        transform=transforms.Compose(
            [random_resized_crop.MyRandomResizedCrop((178, 218))]
        )
    )

    val_dataset = dataset.Att_Dataset(
        "validation",
        args.image_path,
        args.image_dir,
        args.attr_label_path,
        args.attr_to_use,
        transform=transforms.Compose(
            [random_resized_crop.MyRandomResizedCrop((178, 218))]
        )
    )

    test_dataset = dataset.Att_Dataset(
        "testing",
        args.image_path,
        args.image_dir,
        args.attr_label_path,
        args.attr_to_use,
        transform=transforms.Compose(
            [random_resized_crop.MyRandomResizedCrop((178, 218))]
        )
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=10)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=6)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=12)

    if args.save == True:
        checkpoint_callback = ModelCheckpoint(
            monitor='Validation Loss',
            dirpath=args.save_path,
            filename='{epoch:02d}-{Validation Loss:.05f}-resnet50_hair_pretrain_randomResizedCrop',
            save_top_k=25,
            mode='min',
        )

        trainer = pl.Trainer(
            logger=wandb_logger,
            precision=16,
            callbacks=[checkpoint_callback],
            accelerator='ddp',
            gpus=-1,
            num_nodes=1,
            # limit_train_batches=0.5,
            # limit_val_batches=0.5,
            max_epochs=args.train_epochs
        )
    else:
        trainer = pl.Trainer(
            logger=wandb_logger,
            precision=16,
            checkpoint_callback=False,
            accelerator='ddp',
            gpus=-1,
            num_nodes=1,
            # limit_train_batches=0.01,
            max_epochs=args.train_epochs
        )

    if args.train == True:
        trainer.fit(net, train_loader, val_loader)

    if args.val_only == True:
        trainer.test(net, val_loader)

    if args.test == True:
        trainer.test(net, test_loader)