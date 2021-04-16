import torch
import time
import pytorch_lightning as pl
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from torch import nn
import matplotlib.pyplot as plt
from torchvision.models import resnet18, resnet50, resnet152

class Att_Resnet(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.save_name = hparams.save_name
        self.attr_to_use = hparams.attr_to_use
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

        self.blonde_accuracy = pl.metrics.Accuracy()
        self.blonde_precision = pl.metrics.Precision(num_classes=1)
        self.blonde_recall = pl.metrics.Recall(num_classes=1)
        self.blonde_f1 = pl.metrics.F1(num_classes=1)

        self.black_accuracy = pl.metrics.Accuracy()
        self.black_precision = pl.metrics.Precision(num_classes=1)
        self.black_recall = pl.metrics.Recall(num_classes=1)
        self.black_f1 = pl.metrics.F1(num_classes=1)

        self.gray_accuracy = pl.metrics.Accuracy()
        self.gray_precision = pl.metrics.Precision(num_classes=1)
        self.gray_recall = pl.metrics.Recall(num_classes=1)
        self.gray_f1 = pl.metrics.F1(num_classes=1)

        self.brown_accuracy = pl.metrics.Accuracy()
        self.brown_precision = pl.metrics.Precision(num_classes=1)
        self.brown_recall = pl.metrics.Recall(num_classes=1)
        self.brown_f1 = pl.metrics.F1(num_classes=1)

        self.other_accuracy = pl.metrics.Accuracy()
        self.other_precision = pl.metrics.Precision(num_classes=1)
        self.other_recall = pl.metrics.Recall(num_classes=1)
        self.other_f1 = pl.metrics.F1(num_classes=1)

        # self.tp_count = 0
        # self.tn_count = 0
        # self.fp_count = 0
        # self.fn_count = 0

        if hparams.model == "resnet18":
            self.net = resnet18(pretrained=hparams.pretrain, progress=True)
        elif hparams.model == "resnet50":
            self.net = resnet50(pretrained=hparams.pretrain)
        elif hparams.model == resnet152:
            self.net = resnet152(pretrained=hparams.pretrain)

        if hparams.model[:5] == "resnet":
            self.fc_in_feats = self.net.fc.in_features
            self.net.fc = nn.Linear(self.fc_in_feats, hparams.n_labels, bias=True)
            self.n_labels = hparams.n_labels

        elif hparams.model == "simclr":
            dim_mlp = self.net.fc.in_features
            self.net.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.net.fc)

    def forward(self, input):
        output = self.net(input)

        print(self.net)

        return output

    def training_step(self, train_batch, batch_idx):
        inputs, attribute_labels = train_batch['image'], train_batch['label']

        # for i, image in enumerate(inputs):
        #     plt.imshow(image.cpu().numpy().transpose((1,2,0)))
        #     plt.savefig(f"/home/nthom/Downloads/train_{batch_idx}_{i}.jpg")

        attribute_preds = self(inputs)

        if self.simclr == False:
            loss = F.binary_cross_entropy_with_logits(attribute_preds, attribute_labels, reduction='mean')

            attribute_preds = torch.sigmoid(attribute_preds)

            self.log("Training Accuracy Step", self.train_accuracy(attribute_preds, attribute_labels.int()), on_step=True, on_epoch=False, logger=True)
            self.log("Training Precision Step", self.train_precision(attribute_preds, attribute_labels.int()), on_step=True, on_epoch=False, logger=True)
            self.log("Training Recall Step", self.train_recall(attribute_preds, attribute_labels.int()), on_step=True, on_epoch=False, logger=True)
            self.log("Training F1 Step", self.train_f1(attribute_preds, attribute_labels.int()), on_step=True, on_epoch=False, logger=True)
            self.log('Training Loss', loss, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        else:
            logits, labels = self.info_nce_loss(attribute_preds)
            loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='mean')

            self.log("Training Accuracy Step", self.train_accuracy(logits, labels), on_step=True, on_epoch=False, logger=True)
            self.log("Training Precision Step", self.train_precision(logits, labels), on_step=True, on_epoch=False, logger=True)
            self.log("Training Recall Step", self.train_recall(logits, labels), on_step=True, on_epoch=False, logger=True)
            self.log("Training F1 Step", self.train_f1(logits, labels), on_step=True, on_epoch=False, logger=True)
            self.log('Training Loss', loss, on_step=True, on_epoch=True, logger=True, prog_bar=True)

        # sigmoid_preds = self.pre_criterion(attribute_preds)
        # print(sigmoid_preds.type(), attribute_labels.type())
        # loss = self.criterion(sigmoid_preds, attribute_labels)

        return loss

    def training_epoch_end(self, outputs):
        self.log("Training Accuracy Epoch", self.train_accuracy.compute(), on_step=False, on_epoch=True, logger=True)
        self.log("Training Precision Epoch", self.train_precision.compute(), on_step=False, on_epoch=True, logger=True)
        self.log("Training Recall Epoch", self.train_recall.compute(), on_step=False, on_epoch=True, logger=True)
        self.log("Training F1 Epoch", self.train_f1.compute(), on_step=False, on_epoch=True, logger=True)

    def info_nce_loss(self, features):
        labels = torch.cat([torch.arange(self.batch_size) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.temperature
        return logits, labels

    def validation_step(self, val_batch, batch_idx):
        inputs, attribute_labels = val_batch['image'], val_batch['label']

        # for i, image in enumerate(inputs):
        #     plt.imshow(image.cpu().numpy().transpose((1,2,0)))
        #     plt.savefig(f"/home/nthom/Downloads/val_{batch_idx}_{i}.jpg")

        attribute_preds = self(inputs)

        if self.simclr == False:
            loss = F.binary_cross_entropy_with_logits(attribute_preds, attribute_labels, reduction='mean')

            attribute_preds = torch.sigmoid(attribute_preds)

            self.log("Validation Accuracy Step", self.valid_accuracy(attribute_preds, attribute_labels.int()), on_step=True, on_epoch=False, sync_dist=True, logger=True)
            self.log("Validation Precision Step", self.valid_precision(attribute_preds, attribute_labels.int()), on_step=True, on_epoch=False, sync_dist=True, logger=True)
            self.log("Validation Recall Step", self.valid_recall(attribute_preds, attribute_labels.int()), on_step=True, on_epoch=False, sync_dist=True, logger=True)
            self.log("Validation F1 Step", self.valid_f1(attribute_preds, attribute_labels.int()), on_step=True, on_epoch=False, sync_dist=True, logger=True)
            self.log('Validation Loss', loss, on_step=True, on_epoch=True, sync_dist=True, logger=True)

            self.log("Blonde Accuracy Step", self.blonde_accuracy(attribute_preds[:, 2], attribute_labels[:, 2].int()),
                     on_step=True, on_epoch=False, sync_dist=True, logger=True)
            self.log("Blonde Precision Step",
                     self.blonde_precision(attribute_preds[:, 2], attribute_labels[:, 2].int()),
                     on_step=True, on_epoch=False, sync_dist=True, logger=True)
            self.log("Blonde Recall Step", self.blonde_recall(attribute_preds[:, 2], attribute_labels[:, 2].int()),
                     on_step=True, on_epoch=False, sync_dist=True, logger=True)
            self.log("Blonde F1 Step", self.blonde_f1(attribute_preds[:, 2], attribute_labels[:, 2].int()),
                     on_step=True,
                     on_epoch=False, sync_dist=True, logger=True)

            self.log("Black Accuracy Step", self.black_accuracy(attribute_preds[:, 1], attribute_labels[:, 1].int()),
                     on_step=True, on_epoch=False, sync_dist=True, logger=True)
            self.log("Black Precision Step", self.black_precision(attribute_preds[:, 1], attribute_labels[:, 1].int()),
                     on_step=True, on_epoch=False, sync_dist=True, logger=True)
            self.log("Black Recall Step", self.black_recall(attribute_preds[:, 1], attribute_labels[:, 1].int()),
                     on_step=True, on_epoch=False, sync_dist=True, logger=True)
            self.log("Black F1 Step", self.black_f1(attribute_preds[:, 1], attribute_labels[:, 1].int()), on_step=True,
                     on_epoch=False, sync_dist=True, logger=True)

            self.log("Brown Accuracy Step", self.brown_accuracy(attribute_preds[:, 3], attribute_labels[:, 3].int()),
                     on_step=True, on_epoch=False, sync_dist=True, logger=True)
            self.log("Brown Precision Step", self.brown_precision(attribute_preds[:, 3], attribute_labels[:, 3].int()),
                     on_step=True, on_epoch=False, sync_dist=True, logger=True)
            self.log("Brown Recall Step", self.brown_recall(attribute_preds[:, 3], attribute_labels[:, 3].int()),
                     on_step=True, on_epoch=False, sync_dist=True, logger=True)
            self.log("Brown F1 Step", self.brown_f1(attribute_preds[:, 3], attribute_labels[:, 3].int()), on_step=True,
                     on_epoch=False, sync_dist=True, logger=True)

            self.log("Gray Accuracy Step", self.gray_accuracy(attribute_preds[:, 4], attribute_labels[:, 4].int()),
                     on_step=True, on_epoch=False, sync_dist=True, logger=True)
            self.log("Gray Precision Step", self.gray_precision(attribute_preds[:, 4], attribute_labels[:, 4].int()),
                     on_step=True, on_epoch=False, sync_dist=True, logger=True)
            self.log("Gray Recall Step", self.gray_recall(attribute_preds[:, 4], attribute_labels[:, 4].int()),
                     on_step=True, on_epoch=False, sync_dist=True, logger=True)
            self.log("Gray F1 Step", self.gray_f1(attribute_preds[:, 4], attribute_labels[:, 4].int()), on_step=True,
                     on_epoch=False, sync_dist=True, logger=True)

            self.log("Other Accuracy Step", self.other_accuracy(attribute_preds[:, 0], attribute_labels[:, 0].int()),
                     on_step=True, on_epoch=False, sync_dist=True, logger=True)
            self.log("Other Precision Step", self.other_precision(attribute_preds[:, 0], attribute_labels[:, 0].int()),
                     on_step=True, on_epoch=False, sync_dist=True, logger=True)
            self.log("Other Recall Step", self.other_recall(attribute_preds[:, 0], attribute_labels[:, 0].int()),
                     on_step=True, on_epoch=False, sync_dist=True, logger=True)
            self.log("Other F1 Step", self.other_f1(attribute_preds[:, 0], attribute_labels[:, 0].int()), on_step=True,
                     on_epoch=False, sync_dist=True, logger=True)
        else:
            logits, labels = self.info_nce_loss(attribute_preds)
            loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='mean')

            self.log("Validation Accuracy Step", self.valid_accuracy(logits, labels), on_step=True, on_epoch=False, sync_dist=True, logger=True)
            self.log("Validation Precision Step", self.valid_precision(logits, labels), on_step=True, on_epoch=False, sync_dist=True, logger=True)
            self.log("Validation Recall Step", self.valid_recall(logits, labels), on_step=True, on_epoch=False, sync_dist=True, logger=True)
            self.log("Validation F1 Step", self.valid_f1(logits, labels), on_step=True, on_epoch=False, sync_dist=True, logger=True)
            self.log('Validation Loss', loss, on_step=True, on_epoch=True, sync_dist=True, logger=True)

        # attribute_positive_preds = torch.ge(attribute_preds, 1)
        # attribute_negative_preds = torch.lt(attribute_positive_preds, 1)
        # attribute_positive_labels = torch.ge(attribute_labels, 1)
        # attribute_negative_labels = torch.lt(attribute_positive_labels, 1)
        #
        # self.tp_count += torch.sum((attribute_positive_preds & attribute_positive_labels).int(), dim=0)
        # self.fp_count += torch.sum((attribute_positive_preds & attribute_negative_labels).int(), dim=0)
        # self.tn_count += torch.sum((attribute_negative_preds & attribute_negative_labels).int(), dim=0)
        # self.fn_count += torch.sum((attribute_negative_preds & attribute_positive_labels).int(), dim=0)

        # sigmoid_preds = self.pre_criterion(attribute_preds)
        # print(sigmoid_preds.type(), attribute_labels.type())
        # loss = self.criterion(sigmoid_preds.cpu(), attribute_labels.cpu())

        # return loss?
        # return loss

    def validation_epoch_end(self, outputs):
        self.log("Validation Accuracy Epoch", self.valid_accuracy.compute(), on_step=False, on_epoch=True,
                 sync_dist=True, logger=True)
        self.log("Validation Precision Epoch", self.valid_precision.compute(), on_step=False, on_epoch=True,
                 sync_dist=True, logger=True)
        self.log("Validation Recall Epoch", self.valid_recall.compute(), on_step=False, on_epoch=True, sync_dist=True,
                 logger=True)
        self.log("Validation F1 Epoch", self.valid_f1.compute(), on_step=False, on_epoch=True, sync_dist=True,
                 logger=True)

        self.log("blonde Accuracy Epoch", self.blonde_accuracy.compute(), on_step=False, on_epoch=True, sync_dist=True,
                 logger=True)
        self.log("blonde Precision Epoch", self.blonde_precision.compute(), on_step=False, on_epoch=True,
                 sync_dist=True, logger=True)
        self.log("blonde Recall Epoch", self.blonde_recall.compute(), on_step=False, on_epoch=True, sync_dist=True,
                 logger=True)
        self.log("blonde F1 Epoch", self.blonde_f1.compute(), on_step=False, on_epoch=True, sync_dist=True, logger=True)

        self.log("black Accuracy Epoch", self.black_accuracy.compute(), on_step=False, on_epoch=True, sync_dist=True,
                 logger=True)
        self.log("black Precision Epoch", self.black_precision.compute(), on_step=False, on_epoch=True, sync_dist=True,
                 logger=True)
        self.log("black Recall Epoch", self.black_recall.compute(), on_step=False, on_epoch=True, sync_dist=True,
                 logger=True)
        self.log("black F1 Epoch", self.black_f1.compute(), on_step=False, on_epoch=True, sync_dist=True, logger=True)

        self.log("brown Accuracy Epoch", self.brown_accuracy.compute(), on_step=False, on_epoch=True, sync_dist=True,
                 logger=True)
        self.log("brown Precision Epoch", self.brown_precision.compute(), on_step=False, on_epoch=True, sync_dist=True,
                 logger=True)
        self.log("brown Recall Epoch", self.brown_recall.compute(), on_step=False, on_epoch=True, sync_dist=True,
                 logger=True)
        self.log("brown F1 Epoch", self.brown_f1.compute(), on_step=False, on_epoch=True, sync_dist=True, logger=True)

        self.log("gray Accuracy Epoch", self.gray_accuracy.compute(), on_step=False, on_epoch=True, sync_dist=True,
                 logger=True)
        self.log("gray Precision Epoch", self.gray_precision.compute(), on_step=False, on_epoch=True, sync_dist=True,
                 logger=True)
        self.log("gray Recall Epoch", self.gray_recall.compute(), on_step=False, on_epoch=True, sync_dist=True,
                 logger=True)
        self.log("gray F1 Epoch", self.gray_f1.compute(), on_step=False, on_epoch=True, sync_dist=True, logger=True)

        self.log("other Accuracy Epoch", self.other_accuracy.compute(), on_step=False, on_epoch=True, sync_dist=True,
                 logger=True)
        self.log("other Precision Epoch", self.other_precision.compute(), on_step=False, on_epoch=True, sync_dist=True,
                 logger=True)
        self.log("other Recall Epoch", self.other_recall.compute(), on_step=False, on_epoch=True, sync_dist=True,
                 logger=True)
        self.log("other F1 Epoch", self.other_f1.compute(), on_step=False, on_epoch=True, sync_dist=True, logger=True)

        # accuracy, accuracy_pos, accuracy_neg, precision, recall, f1 = self.compute_metrics()
        # self.output_metrics(accuracy, accuracy_pos, accuracy_neg, precision, recall, f1, console=True, file=True, csv=True)

    def compute_metrics(self):
        precision = (self.tp_count / (self.tp_count + self.fp_count))
        recall = (self.tp_count / (self.tp_count + self.fn_count))
        f1 = 2 * ((precision * recall) / (precision + recall))
        accuracy = ((self.tp_count + self.tn_count) / (
                self.tp_count + self.tn_count + self.fp_count + self.fn_count))
        accuracy_pos = (self.tp_count / (self.tp_count + self.fn_count))
        accuracy_neg = (self.tn_count / (self.tn_count + self.fp_count))

        precision = torch.where(torch.isnan(precision), torch.zeros_like(precision), precision)
        recall = torch.where(torch.isnan(recall), torch.zeros_like(recall), recall)
        f1 = torch.where(torch.isnan(f1), torch.zeros_like(f1), f1)
        accuracy = torch.where(torch.isnan(accuracy), torch.zeros_like(accuracy), accuracy)
        accuracy_pos = torch.where(torch.isnan(accuracy_pos), torch.zeros_like(accuracy_pos), accuracy_pos)
        accuracy_neg = torch.where(torch.isnan(accuracy_neg), torch.zeros_like(accuracy_neg), accuracy_neg)

        return accuracy, accuracy_pos, accuracy_neg, precision, recall, f1

    def output_metrics(self, accuracy, accuracy_pos, accuracy_neg, precision, recall, f1_score, console=True,
                       file=False, csv=False):
        ### TO CONSOLE ###
        if console == True:
            print(
                "{0:29} {1:13} {2:13} {3:13} {4:13} {5:13} {6:13}".format("\nAttributes", "Acc", "Acc_pos", "Acc_neg",
                                                                          "Precision", "Recall", "F1"))
            print('-' * 103)

            for attr, acc, acc_pos, acc_neg, prec, rec, f1 in zip(self.attr_to_use, accuracy, accuracy_pos,
                                                                  accuracy_neg,
                                                                  precision, recall, f1_score):
                print("{0:19} {1:13.3f} {2:13.3f} {3:13.3f} {4:13.3f} {5:13.3f} {6:13.3f}".format(attr, acc, acc_pos,
                                                                                                  acc_neg, prec, rec,
                                                                                                  f1))
            print('-' * 103)
            print("{0:19} {1:13.3f} {2:13.3f} {3:13.3f} {4:13.3f} {5:13.3f} {6:13.3f}".format('', torch.mean(accuracy),
                                                                                              torch.mean(accuracy_pos),
                                                                                              torch.mean(accuracy_neg),
                                                                                              torch.mean(precision),
                                                                                              torch.mean(recall),
                                                                                              torch.mean(f1_score)))
        ### TO FILE ###
        if file == True:
            fout = open(
                "/home/nthom/Documents/contrastive_learning/per_attribute_metrics/" + self.save_name + f"_{time.time()}" + ".txt",
                "w+")
            fout.write(
                "{0:29} {1:13} {2:13} {3:13} {4:13} {5:13} {6:13}\n".format("\nAttributes", "Acc", "Acc_pos", "Acc_neg",
                                                                            "Precision", "Recall", "F1"))
            fout.write('-' * 103)
            fout.write("\n")

            for attr, acc, acc_pos, acc_neg, prec, rec, f1 in zip(self.attr_to_use, accuracy, accuracy_pos,
                                                                  accuracy_neg,
                                                                  precision, recall, f1_score):
                fout.write(
                    "{0:19} {1:13.3f} {2:13.3f} {3:13.3f} {4:13.3f} {5:13.3f} {6:13.3f}\n".format(attr, acc, acc_pos,
                                                                                                  acc_neg, prec, rec,
                                                                                                  f1))
            fout.write('-' * 103)
            fout.write('\n')

            fout.write(
                "{0:19} {1:13.3f} {2:13.3f} {3:13.3f} {4:13.3f} {5:13.3f} {6:13.3f}\n".format('', torch.mean(accuracy),
                                                                                              torch.mean(accuracy_pos),
                                                                                              torch.mean(accuracy_neg),
                                                                                              torch.mean(precision),
                                                                                              torch.mean(recall),
                                                                                              torch.mean(f1_score)))
            fout.close()
        ### TO CSV ###
        if csv == True:
            output_preds = []
            output_preds.append(accuracy.tolist())
            output_preds.append(accuracy_pos.tolist())
            output_preds.append(accuracy_neg.tolist())
            output_preds.append(precision.tolist())
            output_preds.append(recall.tolist())
            output_preds.append(f1_score.tolist())
            output_preds_df = pd.DataFrame(output_preds)
            output_preds_df.to_csv(
                "/home/nthom/Documents/contrastive_learning/per_attribute_metrics_csv/" + self.save_name + f"_{time.time()}" + ".csv",
                sep=','
            )
        ##############

    def test_step(self, test_batch, batch_idx):
        inputs, attribute_labels = test_batch['image'], test_batch['label']

        # for i, image in enumerate(inputs):
        #     plt.imshow(image.cpu().numpy().transpose((1,2,0)))
        #     plt.savefig(f"/home/nthom/Downloads/val_{batch_idx}_{i}.jpg")

        attribute_preds = self(inputs)

        loss = F.binary_cross_entropy_with_logits(attribute_preds, attribute_labels, reduction='mean')

        # attribute_positive_preds = torch.ge(attribute_preds, 1)
        # attribute_negative_preds = torch.lt(attribute_positive_preds, 1)
        # attribute_positive_labels = torch.ge(attribute_labels, 1)
        # attribute_negative_labels = torch.lt(attribute_positive_labels, 1)
        #
        # self.tp_count += torch.sum((attribute_positive_preds & attribute_positive_labels).int(), dim=0)
        # self.fp_count += torch.sum((attribute_positive_preds & attribute_negative_labels).int(), dim=0)
        # self.tn_count += torch.sum((attribute_negative_preds & attribute_negative_labels).int(), dim=0)
        # self.fn_count += torch.sum((attribute_negative_preds & attribute_positive_labels).int(), dim=0)

        attribute_preds = torch.sigmoid(attribute_preds)

        self.log("Test Accuracy Step", self.valid_accuracy(attribute_preds, attribute_labels.int()), on_step=True,
                 on_epoch=False, sync_dist=True, logger=True)
        self.log("Test Precision Step", self.valid_precision(attribute_preds, attribute_labels.int()), on_step=True,
                 on_epoch=False, sync_dist=True, logger=True)
        self.log("Test Recall Step", self.valid_recall(attribute_preds, attribute_labels.int()), on_step=True,
                 on_epoch=False, sync_dist=True, logger=True)
        self.log("Test F1 Step", self.valid_f1(attribute_preds, attribute_labels.int()), on_step=True, on_epoch=False,
                 sync_dist=True, logger=True)
        self.log('Test Loss', loss, on_step=True, on_epoch=True, sync_dist=True, logger=True)

        self.log("Blonde Accuracy Step", self.blonde_accuracy(attribute_preds[:, 2], attribute_labels[:, 2].int()),
                 on_step=True, on_epoch=False, sync_dist=True, logger=True)
        self.log("Blonde Precision Step", self.blonde_precision(attribute_preds[:, 2], attribute_labels[:, 2].int()),
                 on_step=True, on_epoch=False, sync_dist=True, logger=True)
        self.log("Blonde Recall Step", self.blonde_recall(attribute_preds[:, 2], attribute_labels[:, 2].int()),
                 on_step=True, on_epoch=False, sync_dist=True, logger=True)
        self.log("Blonde F1 Step", self.blonde_f1(attribute_preds[:, 2], attribute_labels[:, 2].int()), on_step=True,
                 on_epoch=False, sync_dist=True, logger=True)

        self.log("Black Accuracy Step", self.black_accuracy(attribute_preds[:, 1], attribute_labels[:, 1].int()),
                 on_step=True, on_epoch=False, sync_dist=True, logger=True)
        self.log("Black Precision Step", self.black_precision(attribute_preds[:, 1], attribute_labels[:, 1].int()),
                 on_step=True, on_epoch=False, sync_dist=True, logger=True)
        self.log("Black Recall Step", self.black_recall(attribute_preds[:, 1], attribute_labels[:, 1].int()),
                 on_step=True, on_epoch=False, sync_dist=True, logger=True)
        self.log("Black F1 Step", self.black_f1(attribute_preds[:, 1], attribute_labels[:, 1].int()), on_step=True,
                 on_epoch=False, sync_dist=True, logger=True)

        self.log("Brown Accuracy Step", self.brown_accuracy(attribute_preds[:, 3], attribute_labels[:, 3].int()),
                 on_step=True, on_epoch=False, sync_dist=True, logger=True)
        self.log("Brown Precision Step", self.brown_precision(attribute_preds[:, 3], attribute_labels[:, 3].int()),
                 on_step=True, on_epoch=False, sync_dist=True, logger=True)
        self.log("Brown Recall Step", self.brown_recall(attribute_preds[:, 3], attribute_labels[:, 3].int()),
                 on_step=True, on_epoch=False, sync_dist=True, logger=True)
        self.log("Brown F1 Step", self.brown_f1(attribute_preds[:, 3], attribute_labels[:, 3].int()), on_step=True,
                 on_epoch=False, sync_dist=True, logger=True)

        self.log("Gray Accuracy Step", self.gray_accuracy(attribute_preds[:, 4], attribute_labels[:, 4].int()),
                 on_step=True, on_epoch=False, sync_dist=True, logger=True)
        self.log("Gray Precision Step", self.gray_precision(attribute_preds[:, 4], attribute_labels[:, 4].int()),
                 on_step=True, on_epoch=False, sync_dist=True, logger=True)
        self.log("Gray Recall Step", self.gray_recall(attribute_preds[:, 4], attribute_labels[:, 4].int()),
                 on_step=True, on_epoch=False, sync_dist=True, logger=True)
        self.log("Gray F1 Step", self.gray_f1(attribute_preds[:, 4], attribute_labels[:, 4].int()), on_step=True,
                 on_epoch=False, sync_dist=True, logger=True)

        self.log("Other Accuracy Step", self.other_accuracy(attribute_preds[:, 0], attribute_labels[:, 0].int()),
                 on_step=True, on_epoch=False, sync_dist=True, logger=True)
        self.log("Other Precision Step", self.other_precision(attribute_preds[:, 0], attribute_labels[:, 0].int()),
                 on_step=True, on_epoch=False, sync_dist=True, logger=True)
        self.log("Other Recall Step", self.other_recall(attribute_preds[:, 0], attribute_labels[:, 0].int()),
                 on_step=True, on_epoch=False, sync_dist=True, logger=True)
        self.log("Other F1 Step", self.other_f1(attribute_preds[:, 0], attribute_labels[:, 0].int()), on_step=True,
                 on_epoch=False, sync_dist=True, logger=True)

    def test_epoch_end(self, outputs):
        self.log("Test Accuracy Epoch", self.valid_accuracy.compute(), on_step=False, on_epoch=True, sync_dist=True,
                 logger=True)
        self.log("Test Precision Epoch", self.valid_precision.compute(), on_step=False, on_epoch=True, sync_dist=True,
                 logger=True)
        self.log("Test Recall Epoch", self.valid_recall.compute(), on_step=False, on_epoch=True, sync_dist=True,
                 logger=True)
        self.log("Test F1 Epoch", self.valid_f1.compute(), on_step=False, on_epoch=True, sync_dist=True, logger=True)

        self.log("blonde Accuracy Epoch", self.blonde_accuracy.compute(), on_step=False, on_epoch=True, sync_dist=True,
                 logger=True)
        self.log("blonde Precision Epoch", self.blonde_precision.compute(), on_step=False, on_epoch=True,
                 sync_dist=True, logger=True)
        self.log("blonde Recall Epoch", self.blonde_recall.compute(), on_step=False, on_epoch=True, sync_dist=True,
                 logger=True)
        self.log("blonde F1 Epoch", self.blonde_f1.compute(), on_step=False, on_epoch=True, sync_dist=True, logger=True)

        self.log("black Accuracy Epoch", self.black_accuracy.compute(), on_step=False, on_epoch=True, sync_dist=True,
                 logger=True)
        self.log("black Precision Epoch", self.black_precision.compute(), on_step=False, on_epoch=True, sync_dist=True,
                 logger=True)
        self.log("black Recall Epoch", self.black_recall.compute(), on_step=False, on_epoch=True, sync_dist=True,
                 logger=True)
        self.log("black F1 Epoch", self.black_f1.compute(), on_step=False, on_epoch=True, sync_dist=True, logger=True)

        self.log("brown Accuracy Epoch", self.brown_accuracy.compute(), on_step=False, on_epoch=True, sync_dist=True,
                 logger=True)
        self.log("brown Precision Epoch", self.brown_precision.compute(), on_step=False, on_epoch=True, sync_dist=True,
                 logger=True)
        self.log("brown Recall Epoch", self.brown_recall.compute(), on_step=False, on_epoch=True, sync_dist=True,
                 logger=True)
        self.log("brown F1 Epoch", self.brown_f1.compute(), on_step=False, on_epoch=True, sync_dist=True, logger=True)

        self.log("gray Accuracy Epoch", self.gray_accuracy.compute(), on_step=False, on_epoch=True, sync_dist=True,
                 logger=True)
        self.log("gray Precision Epoch", self.gray_precision.compute(), on_step=False, on_epoch=True, sync_dist=True,
                 logger=True)
        self.log("gray Recall Epoch", self.gray_recall.compute(), on_step=False, on_epoch=True, sync_dist=True,
                 logger=True)
        self.log("gray F1 Epoch", self.gray_f1.compute(), on_step=False, on_epoch=True, sync_dist=True, logger=True)

        self.log("other Accuracy Epoch", self.other_accuracy.compute(), on_step=False, on_epoch=True, sync_dist=True,
                 logger=True)
        self.log("other Precision Epoch", self.other_precision.compute(), on_step=False, on_epoch=True, sync_dist=True,
                 logger=True)
        self.log("other Recall Epoch", self.other_recall.compute(), on_step=False, on_epoch=True, sync_dist=True,
                 logger=True)
        self.log("other F1 Epoch", self.other_f1.compute(), on_step=False, on_epoch=True, sync_dist=True, logger=True)

        # accuracy, accuracy_pos, accuracy_neg, precision, recall, f1 = self.compute_metrics()
        # self.output_metrics(accuracy, accuracy_pos, accuracy_neg, precision, recall, f1, console=True, file=True, csv=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer)

        return [optimizer], [lr_scheduler]
