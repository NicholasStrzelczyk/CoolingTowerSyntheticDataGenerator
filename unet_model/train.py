import os
from datetime import datetime

import torch
from torch import optim
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryF1Score, BinaryPrecision, BinaryRecall
from torchsummary import summary
from tqdm import tqdm

from custom_ds import CustomDS
from unet_model import UNet
from utils.helper import *


def train(model, loss_fn, optimizer, scheduler, train_loader, val_loader, n_epochs, device):
    global model_version, save_path

    precision = BinaryPrecision(threshold=0.5).to(device=device)
    recall = BinaryRecall(threshold=0.5).to(device=device)
    f1_score = BinaryF1Score(threshold=0.5).to(device=device)

    losses_train, losses_val = [], []
    precision_train, precision_val = [], []
    recall_train, recall_val = [], []
    f1_train, f1_val = [], []

    # --- iterate through all epochs --- #
    print("{} starting training for model {}...".format(datetime.now(), model_version))
    for epoch in range(n_epochs):

        # --- training step --- #
        model.train()
        epoch_loss, epoch_bp, epoch_br, epoch_bf1 = 0.0, 0.0, 0.0, 0.0
        for images, targets in tqdm(train_loader, desc="epoch {} train progress".format(epoch + 1)):
            images = images.to(device=device)
            targets = targets.to(device=device)
            outputs = model(images)
            loss = loss_fn(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_bp += precision(outputs, targets).item()
            epoch_br += recall(outputs, targets).item()
            epoch_bf1 += f1_score(outputs, targets).item()
            del images, targets, outputs

        losses_train.append(epoch_loss / len(train_loader))
        precision_train.append(epoch_bp / len(train_loader))
        recall_train.append(epoch_br / len(train_loader))
        f1_train.append(epoch_bf1 / len(train_loader))

        # --- validation step --- #
        model.eval()
        epoch_loss, epoch_bp, epoch_br, epoch_bf1 = 0.0, 0.0, 0.0, 0.0
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc="epoch {} val progress".format(epoch + 1)):
                images = images.to(device=device)
                targets = targets.to(device=device)
                outputs = model(images)
                loss = loss_fn(outputs, targets)
                epoch_loss += loss.item()
                epoch_bp += precision(outputs, targets).item()
                epoch_br += recall(outputs, targets).item()
                epoch_bf1 += f1_score(outputs, targets).item()
                del images, targets, outputs

        scheduler.step(epoch_loss)  # using validation loss

        losses_val.append(epoch_loss / len(val_loader))
        precision_val.append(epoch_bp / len(val_loader))
        recall_val.append(epoch_br / len(val_loader))
        f1_val.append(epoch_bf1 / len(val_loader))

        # --- print epoch results --- #
        log_and_print("{} epoch {}/{} metrics:".format(datetime.now(), epoch + 1, n_epochs))
        log_and_print("\t[train] loss: {:.9f}, precision: {:.9f}, recall: {:.9f}, f1_score: {:.9f}".format(
            losses_train[epoch], precision_train[epoch], recall_train[epoch], f1_train[epoch]))
        log_and_print("\t[valid] loss: {:.9f}, precision: {:.9f}, recall: {:.9f}, f1_score: {:.9f}".format(
            losses_val[epoch], precision_val[epoch], recall_val[epoch], f1_val[epoch]))

    # --- save weights and plot metrics --- #
    torch.save(model.state_dict(), os.path.join(save_path, "model_{}_weights.pth".format(model_version)))
    metrics_history = [
        ("loss", losses_train, losses_val),
        ("precision", precision_train, precision_val),
        ("recall", recall_train, recall_val),
        ("f1_score", f1_train, f1_val),
    ]
    print_metric_plots(metrics_history, model_version, save_path)


if __name__ == '__main__':
    # hyperparameters
    model_version = 1
    n_epochs = 20  # num of epochs
    batch_sz = 1  # batch size
    lr = 0.0001  # learning rate
    momentum = 0.99  # used in U-Net paper
    resize_shape = (512, 512)  # used in U-Net paper for training
    list_path, save_path = get_os_dependent_paths(model_version, partition='train')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # set deterministic seed
    make_deterministic(2024)

    # initialize logger
    setup_logger(os.path.join(save_path, 'training.log'))

    # set up dataset(s)
    x_train, y_train, x_val, y_val = get_data_from_list(list_path, split=0.2)
    train_ds = CustomDS(x_train, y_train, resize_shape=resize_shape)
    val_ds = CustomDS(x_val, y_val, resize_shape=resize_shape)
    train_loader = DataLoader(train_ds, batch_size=batch_sz, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_sz, shuffle=False)

    # compile model
    model = UNet()
    model.to(device=device)

    # init model training parameters
    loss_fn = torch.nn.BCELoss()
    optimizer = optim.SGD(params=model.parameters(), lr=lr, momentum=momentum)  # SGD used in U-Net paper
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # not sure if needed

    # run torch summary report
    summary(model, input_size=(3, resize_shape[0], resize_shape[1]))

    # train model
    train(model, loss_fn, optimizer, scheduler, train_loader, val_loader, n_epochs, device)
