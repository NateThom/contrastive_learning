import utils
import att_resnet
import dataset
import random_resized_crop
import random_blur
import random_hflip
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchvision import transforms
from pytorch_lightning.callbacks import ModelCheckpoint

if __name__=="__main__":
    args = utils.get_args()

    pl.seed_everything(args.random_seed)

    wandb_logger = WandbLogger(name=args.save_name, project='contrastive_learning', entity='unr-mpl')

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
            # [random_resized_crop.MyRandomResizedCrop((178, 218)), random_blur.MyRandomBlur(15), random_hflip.MyRandomHFlip()]
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

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=12)

    if args.save == True:
        checkpoint_callback = ModelCheckpoint(
            monitor='Validation Loss',
            dirpath=args.save_path,
            filename='{epoch:02d}-{Validation Loss:.05f}-' + args.save_name,
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
            # limit_train_batches=0.3,
            # limit_val_batches=0.3,
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
