'''
This module contains methods for training models with different loss functions.
'''

import torch
from sympy.physics.quantum.identitysearch import np
from torch.nn import functional as F
from torch import nn

from Losses.loss import cross_entropy, focal_loss, focal_loss_adaptive, label_smoothing, soft_ece, smooth_soft_ece, \
    dual_focal_loss, mse_loss
from Losses.loss import mmce, mmce_weighted
from Losses.loss import brier_score

mean_reduction_losses = ['MMCE', 'SoftECE', 'SmoothSoftECE', 'LS-0.05',"MSE"]

loss_function_dict = {
    'NLL': cross_entropy,
    'FLSD-3': focal_loss,
    'FLSD-53': focal_loss_adaptive,
    'MMCE': mmce_weighted,
    'mmce_weighted': mmce_weighted,
    'Brier': brier_score,
    'LS-0.05': label_smoothing,
    'SoftECE': soft_ece,
    'SmoothSoftECE': smooth_soft_ece,
    'DFL':dual_focal_loss,
    "MSE": mse_loss
}


def train_single_epoch(epoch,
                       model,
                       train_loader,
                       optimizer,
                       device,
                       loss_function='cross_entropy',
                       gamma=1.0,
                       lamda=1.0,
                       loss_mean=False):
    '''
    Util method for training a model for a single epoch.
    '''
    log_interval = 10
    model.train()
    train_loss = 0
    num_samples = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        data = data.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(data)
        if loss_function in mean_reduction_losses:
            loss = (len(data) * loss_function_dict[loss_function](logits, labels, gamma=gamma, lamda=lamda, device=device))
        else:
            loss = loss_function_dict[loss_function](logits, labels, gamma=gamma, lamda=lamda, device=device)

        if loss_mean:
            loss = loss / len(data)

        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 2)
        train_loss += loss.item()
        optimizer.step()
        num_samples += len(data)

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader) * len(data),
                100. * batch_idx / len(train_loader),
                loss.item()))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / num_samples))
    return train_loss / num_samples



def test_single_epoch(epoch,
                      model,
                      test_val_loader,
                      device,
                      loss_function='cross_entropy',
                      gamma=1.0,
                      lamda=1.0):
    '''
    Util method for testing a model for a single epoch.
    '''
    logit_list = []
    label_list = []
    feature_list = []
    model.eval()
    loss = 0
    num_samples = 0
    with torch.no_grad():
        for i, (data, labels) in enumerate(test_val_loader):
            data = data.to(device)
            labels = labels.to(device)

            logit,feature = model(data,return_features=True)
            if loss_function in mean_reduction_losses:
                loss += (len(data) * loss_function_dict[loss_function](logit, labels, gamma=gamma, lamda=lamda, device=device).item())
            else:
                loss += loss_function_dict[loss_function](logit, labels, gamma=gamma, lamda=lamda, device=device).item()
            num_samples += len(data)
            feature_list.append(feature.cpu().numpy())
            logit_list.append(logit.cpu().numpy())
            label_list.append(labels.cpu().numpy())

    print('======> Test set loss: {:.4f}'.format(
        loss / num_samples))
    logit_list = np.vstack(logit_list)
    label_list = np.hstack(label_list)
    feature_list = np.vstack(feature_list)


    return loss / num_samples, logit_list, feature_list, label_list