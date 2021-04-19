import argparse
import pandas as pd
import torch
import torchvision
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
import numpy as np
import matplotlib.pyplot as plt
import wandb

# My code
import celeba_dataset

# SimCLR
from simclr import SimCLR
from simclr.modules import NT_Xent, get_resnet
from simclr.modules.transformations import TransformsSimCLR
from simclr.modules.identity import Identity
from simclr.modules.sync_batchnorm import convert_model

from utils import yaml_config_hook


class ContrastiveLearning(LightningModule):
    def __init__(self, args):
        super().__init__()

        self.hparams = args

        # initialize ResNet
        self.encoder = get_resnet(self.hparams.resnet, pretrained=self.hparams.pretrain)
        self.n_features = self.encoder.fc.in_features  # get dimensions of fc layer
        self.model = SimCLR(self.encoder, self.hparams.h_dim, self.hparams.projection_dim, self.n_features, self.hparams.n_classes)
        self.test_outputs = np.array([])
        self.criterion = NT_Xent(
            self.hparams.batch_size, self.hparams.temperature, world_size=1
        )

    def forward(self, x_i, x_j):
        h_i, h_j, z_i, z_j = self.model(x_i, x_j)
        return h_i, h_j, z_i, z_j

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        (x_i, x_j), _ = batch
        h_i, h_j, z_i, z_j = self.forward(x_i, x_j)
        loss = self.criterion(z_i, z_j)
        self.log("Training Loss", loss, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        return loss

    def test_step(self, test_batch, batch_idx):
        (x_i, _), _ = test_batch
        h_i, h_j, z_i, z_j = self.forward(x_i, x_i)
        h_i = h_i.cpu().numpy()

        if len(self.test_outputs) == 0:
            self.test_outputs = h_i
        else:
            self.test_outputs = np.append(self.test_outputs, h_i, axis=0)

    def test_epoch_end(self, outputs):
        output_csv = pd.DataFrame(self.test_outputs)
        output_csv.to_csv(self.hparams.csv_path + self.hparams.model_file[:-4] + "csv", header=False, index=False)

    def configure_criterion(self):
        criterion = NT_Xent(self.hparams.batch_size, self.hparams.temperature)
        return criterion

    def configure_optimizers(self):
        scheduler = None
        if self.hparams.optimizer == "Adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)
        elif self.hparams.optimizer == "LARS":
            # optimized using LARS with linear learning rate scaling
            # (i.e. LearningRate = 0.3 × BatchSize/256) and weight decay of 10−6.
            learning_rate = 0.3 * args.batch_size / 256
            optimizer = LARS(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=args.weight_decay,
                exclude_from_weight_decay=["batch_normalization", "bias"],
            )

            # "decay the learning rate with the cosine decay schedule without restarts"
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, args.epochs, eta_min=0, last_epoch=-1
            )
        else:
            raise NotImplementedError

        if scheduler:
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            return {"optimizer": optimizer}

# Helper function to show a batch
def show_landmarks_batch(batch):
    """Show image with landmarks for a batch of samples."""
    (x_i, x_j), _ = batch
    batch_size = len(x_i)
    im_size = x_i.size(2)
    grid_border_size = 2

    grid = torchvision.utils.make_grid(torch.cat((x_i, x_j)))
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

if __name__ == "__main__":
    wandb_logger = WandbLogger(project="contrastive_learning", entity='unr-mpl')

    parser = argparse.ArgumentParser(description="SimCLR")
    yaml_config = yaml_config_hook("./config/config.yaml")

    sweep = True
    if sweep:
        hyperparameter_defaults = dict(
            h_dim = 512,
            projection_dim = 64,
            temperature = 0.05,
            learning_rate = 0.0003,
        )

        wandb.init(config=hyperparameter_defaults)

        yaml_config = yaml_config_hook("./config/config.yaml")
        wandb.config.update(
            {k:v for k, v in yaml_config.items() if k not in wandb.config}
        )

        for k, v in wandb.config.items():
            parser.add_argument(f"--{k}", default=v, type=type(v))

        args = parser.parse_args()
    else:
        for k, v in yaml_config.items():
            parser.add_argument(f"--{k}", default=v, type=type(v))

        args = parser.parse_args()

    pl.seed_everything(args.seed)

    if args.train:
        if args.dataset == "STL10":
            train_dataset = torchvision.datasets.STL10(
                args.dataset_dir,
                split="unlabeled",
                download=True,
                transform=TransformsSimCLR(size=args.image_size),
            )
        elif args.dataset == "CIFAR10":
            train_dataset = torchvision.datasets.CIFAR10(
                args.dataset_dir,
                download=True,
                transform=TransformsSimCLR(size=args.image_size),
            )
        elif args.dataset == "CelebA":
            train_dataset = celeba_dataset.Att_Dataset(
                args,
                fold="not test",
                transform=TransformsSimCLR(size=(args.image_size_h, args.image_size_w)),
            )

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers, drop_last=True, shuffle=True)

        if args.show_batch:
            for i_batch, sampled_batch in enumerate(train_loader):
                plt.figure()
                show_landmarks_batch(sampled_batch)
                plt.axis('off')
                plt.ioff()
                plt.show()
                if i_batch == 10:
                    print()

    elif args.test:
        if args.dataset == "CelebA":
            test_dataset = celeba_dataset.Att_Dataset(
                args,
                fold="testing",
                transform=TransformsSimCLR(size=(args.image_size_h, args.image_size_w), test_flag=True),
            )

        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.workers, drop_last=False)

        if args.show_batch:
            for i_batch, sampled_batch in enumerate(test_loader):
                plt.figure()
                show_landmarks_batch(sampled_batch)
                plt.axis('off')
                plt.ioff()
                plt.show()
                if i_batch == 10:
                    print()

    if args.reload:
        cl = ContrastiveLearning.load_from_checkpoint(args.model_path + args.model_file, args=args)
    else:
        cl = ContrastiveLearning(args)

    if args.save == True:
        checkpoint_callback = ModelCheckpoint(
            monitor='Training Loss',
            dirpath=args.model_path,
            filename='{epoch:02d}-{Training Loss:.05f}-' + f"{args.h_dim}-" + f"{args.projection_dim}-" +
                     f"{args.temperature}-" + f"{args.learning_rate}",
            save_top_k=1,
            mode='min',
        )

        trainer = pl.Trainer(
            logger=wandb_logger,
            precision=16,
            callbacks=[checkpoint_callback],
            accelerator='ddp',
            plugins=DDPPlugin(find_unused_parameters=False),
            gpus=args.gpus,
            num_nodes=1,
            # limit_train_batches=0.01,
            # limit_val_batches=0.3,
            max_epochs=args.epochs
        )
    else:
        trainer = pl.Trainer(
            logger=wandb_logger,
            precision=16,
            checkpoint_callback=False,
            accelerator='ddp',
            plugins=DDPPlugin(find_unused_parameters=False),
            gpus=args.gpus,
            num_nodes=1,
            # limit_train_batches=0.01,
            max_epochs=args.epochs
        )

    if args.train == True:
        trainer.sync_batchnorm = True
        trainer.fit(cl, train_loader)

    # if args.train == True:
    #     # trainer.fit(net, train_loader, val_loader)
    #     trainer.fit(cl, train_loader)
    #
    # if args.val_only == True:
    #     trainer.test(cl, val_loader)
    #
    elif args.test == True:
        trainer.test(cl, test_loader)
