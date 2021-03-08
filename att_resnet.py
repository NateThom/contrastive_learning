import torch

import pytorch_lightning as pl
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from torch import nn
from torchvision.models import resnet18, resnet50

class Att_Resnet(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.lr = hparams.lr

        self.save_hyperparameters()

        self.train_accuracy = pl.metrics.Accuracy()
        self.train_precision = pl.metrics.Precision(num_classes=hparams.n_labels)
        self.train_recall = pl.metrics.Recall(num_classes=hparams.n_labels)
        self.train_f1 = pl.metrics.F1(num_classes=hparams.n_labels)

        self.valid_accuracy = pl.metrics.Accuracy()
        self.valid_precision = pl.metrics.Precision(num_classes=hparams.n_labels)
        self.valid_recall = pl.metrics.Recall(num_classes=hparams.n_labels)
        self.valid_f1 = pl.metrics.F1(num_classes=hparams.n_labels)

        self.test_accuracy = pl.metrics.Accuracy()
        self.test_precision = pl.metrics.Precision(num_classes=hparams.n_labels)
        self.test_recall = pl.metrics.Recall(num_classes=hparams.n_labels)
        self.test_f1 = pl.metrics.F1(num_classes=hparams.n_labels)

        self.resnet = resnet50(pretrained=hparams.pretrain, progress=True)
        self.fc_in_feats = self.resnet.fc.in_features

        self.resnet.fc = nn.Linear(self.fc_in_feats, hparams.n_labels, bias=True)
        self.n_labels = hparams.n_labels

    def forward(self, input):
        output = self.resnet(input)

        return output

    def training_step(self, train_batch, batch_idx):
        inputs, attribute_labels = train_batch['image'], train_batch['attributes']

        attribute_preds = self(inputs)
        attribute_preds = torch.sigmoid(attribute_preds)

        loss = F.binary_cross_entropy_with_logits(attribute_preds, attribute_labels.float(), reduction='mean')

        self.log("Training Accuracy Step", self.train_accuracy(attribute_preds, attribute_labels), on_step=True, on_epoch=False, logger=True)
        self.log("Training Precision Step", self.train_precision(attribute_preds, attribute_labels), on_step=True, on_epoch=False, logger=True)
        self.log("Training Recall Step", self.train_recall(attribute_preds, attribute_labels), on_step=True, on_epoch=False, logger=True)
        self.log("Training F1 Step", self.train_f1(attribute_preds, attribute_labels), on_step=True, on_epoch=False, logger=True)
        self.log('Training Loss', loss, on_step=True, on_epoch=True, logger=True, prog_bar=True)

        return loss

    def training_epoch_end(self, outputs):
        self.log("Training Accuracy Epoch", self.train_accuracy.compute(), on_step=False, on_epoch=True, logger=True)
        self.log("Training Precision Epoch", self.train_precision.compute(), on_step=False, on_epoch=True, logger=True)
        self.log("Training Recall Epoch", self.train_recall.compute(), on_step=False, on_epoch=True, logger=True)
        self.log("Training F1 Epoch", self.train_f1.compute(), on_step=False, on_epoch=True, logger=True)

    # def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
    #     if self.show_batch_flag == True:
    #         if self.segment_flag == True:
    #             batch_inputs, batch_attribute_labels, batch_mask_labels = batch['image'], batch['attributes'], batch['masks']
    #         else:
    #             batch_inputs, batch_attribute_labels = batch['image'], batch['attributes']
    #
    #             ########## SHOW INPUT ##########
    #             for index, image in enumerate(batch_inputs):
    #                 image = image.numpy()
    #                 image = image.transpose(1, 2, 0)
    #                 # Note that there are two ways to view an image. Save the image and open it, or
    #                 #      show the image while the program is running. Either uncomment imshow and waitKey
    #                 #      or imwrite
    #                 plt.imshow(image)
    #                 plt.show()
    #                 input("Press 'Enter' for next input image.")
    #                 # cv2.imwrite(f"batch_image_{index}.png", image)
    #
    #             if self.segment_flag == True:
    #                 ########## SHOW MASKS ##########
    #                 for index1, sample_masks in enumerate(batch_mask_labels):
    #                     for index2, mask in enumerate(sample_masks):
    #                         mask = (mask.numpy() * 255).astype(int)
    #                         # Note that there are two ways to view an image. Save the image and open it, or
    #                         #      show the image while the program is running. Either uncomment imshow and
    #                         #      waitKey or imwrite
    #                         plt.imshow(mask)
    #                         plt.show()
    #
    #                         # mask_prediction = (outputs)
    #
    #                     input("Press 'Enter' for next sample's masks.")

    def validation_step(self, val_batch, batch_idx):
        inputs, attribute_labels = val_batch['image'], val_batch['attributes']

        attribute_preds = self(inputs)
        attribute_preds = torch.sigmoid(attribute_preds)

        loss = F.binary_cross_entropy_with_logits(attribute_preds, attribute_labels.float(), reduction='mean')

        self.log("Validation Accuracy Step", self.valid_accuracy(attribute_preds, attribute_labels), on_step=True, on_epoch=False, sync_dist=True, logger=True)
        self.log("Validation Precision Step", self.valid_precision(attribute_preds, attribute_labels), on_step=True, on_epoch=False, sync_dist=True, logger=True)
        self.log("Validation Recall Step", self.valid_recall(attribute_preds, attribute_labels), on_step=True, on_epoch=False, sync_dist=True, logger=True)
        self.log("Validation F1 Step", self.valid_f1(attribute_preds, attribute_labels), on_step=True, on_epoch=False, sync_dist=True, logger=True)
        self.log('Validation Loss', loss, on_step=True, on_epoch=True, sync_dist=True, logger=True)

        return loss

    def validation_epoch_end(self, outputs):
        self.log("Validation Accuracy Epoch", self.valid_accuracy.compute(), on_step=False, on_epoch=True, sync_dist=True, logger=True)
        self.log("Validation Precision Epoch", self.valid_precision.compute(), on_step=False, on_epoch=True, sync_dist=True, logger=True)
        self.log("Validation Recall Epoch", self.valid_recall.compute(), on_step=False, on_epoch=True, sync_dist=True, logger=True)
        self.log("Validation F1 Epoch", self.valid_f1.compute(), on_step=False, on_epoch=True, sync_dist=True, logger=True)

    def test_step(self, val_batch, batch_idx):
        inputs, attribute_labels = val_batch['image'], val_batch['attributes']
        attribute_preds = self(inputs)

        loss = F.binary_cross_entropy_with_logits(attribute_preds, attribute_labels, reduction='mean')

        self.log("Test Accuracy Step", self.test_accuracy(attribute_preds, attribute_labels), on_step=True,
                 on_epoch=False, sync_dist=True, logger=True)
        self.log("Test Precision Step", self.test_precision(attribute_preds, attribute_labels), on_step=True,
                 on_epoch=False, sync_dist=True, logger=True)
        self.log("Test Recall Step", self.test_recall(attribute_preds, attribute_labels), on_step=True,
                 on_epoch=False, sync_dist=True, logger=True)
        self.log("Test F1 Step", self.test_f1(attribute_preds, attribute_labels), on_step=True, on_epoch=False,
                 sync_dist=True, logger=True)
        self.log('Test Loss Step', loss, on_step=True, on_epoch=False, sync_dist=True, logger=True)

    def test_epoch_end(self, outputs):
        self.log("Test Accuracy Epoch", self.test_accuracy.compute(), on_step=False, on_epoch=True,
                 sync_dist=True, logger=True)
        self.log("Test Precision Epoch", self.test_precision.compute(), on_step=False, on_epoch=True,
                 sync_dist=True, logger=True)
        self.log("Test Recall Epoch", self.test_recall.compute(), on_step=False, on_epoch=True, sync_dist=True,
                 logger=True)
        self.log("Test F1 Epoch", self.test_f1.compute(), on_step=False, on_epoch=True, sync_dist=True,
                 logger=True)

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.0001)
        lmbda = lambda epoch: 0.9
        lr_scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer=optimizer, lr_lambda=lmbda, last_epoch=-1, verbose=False)
        # lr_scheduler = {
            # 'scheduler': optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=self.patience, verbose=False),
            # 'monitor': 'Validation Loss Step'
            # 'scheduler': optim.lr_scheduler.MultiplicativeLR(optimizer=optimizer, lr_lambda=lmbda, last_epoch=-1, verbose=False),
        # }

        return [optimizer], [lr_scheduler]
        # return optimizer