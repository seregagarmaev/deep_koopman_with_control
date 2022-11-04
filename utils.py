from datasets import NCMAPSS
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, TensorDataset


def prepare_ncmapss_data(**params):
    dataset = NCMAPSS(params['path'])
    Xs_scaler = dataset.scale(dataset.X_s_var)
    w_scaler = dataset.scale(dataset.W_var)
    theta_scaler = dataset.scale(dataset.T_var)

    dataset.shift_data()
    dataset.keep_regime(params['regime'])

    eval_data = dataset.get_eval_data(params['eval_unit'], params['eval_cycle'])
    eval_Xs_k0 = eval_data[dataset.X_s_var]
    eval_W_k0 = eval_data[dataset.W_var]
    eval_theta_k0 = eval_data[dataset.T_var]
    eval_Xs_k1 = eval_data[[col + dataset.shift_suffix for col in dataset.X_s_var]]

    dataset.keep_train_units(params['train_units'])
    dataset.keep_test_units(params['test_units'])
    dataset.subsample_cycles(frac=params['cycles_frac'])

    train_Xs_k0, test_Xs_k0 = dataset.get_data(dataset.X_s_var)
    train_W_k0, test_W_k0 = dataset.get_data(dataset.W_var)
    train_theta_k0, test_theta_k0 = dataset.get_data(dataset.T_var)
    train_Xs_k1, test_Xs_k1 = dataset.get_data([col + dataset.shift_suffix for col in dataset.X_s_var])

    dataset = TensorDataset(torch.Tensor(train_Xs_k0.values),
                            torch.Tensor(train_W_k0.values),
                            torch.Tensor(train_theta_k0.values),
                            torch.Tensor(train_Xs_k1.values))
    training_loader = DataLoader(dataset, batch_size=params['bs'], shuffle=True, drop_last=True)

    dataset = TensorDataset(torch.Tensor(test_Xs_k0.values),
                            torch.Tensor(test_W_k0.values),
                            torch.Tensor(test_theta_k0.values),
                            torch.Tensor(test_Xs_k1.values))
    testing_loader = DataLoader(dataset, batch_size=params['bs'], shuffle=False, drop_last=False)
    return training_loader, testing_loader, \
           (eval_Xs_k0, eval_W_k0, eval_theta_k0, eval_Xs_k1), \
           (Xs_scaler, w_scaler, theta_scaler)

def train_model(model, optimizer, dataloader):
    '''
    Trains model for a single epoch.
    '''
    model.train()
    device = next(model.parameters()).device
    total_loss = 0
    mseLoss = nn.MSELoss()

    for Xs_k0, W_k0, theta_k0, Xs_k1 in dataloader:
        Xs_k1_hat = model(torch.cat((Xs_k0, W_k0), dim=1).to(device), theta_k0.to(device))

        loss = mseLoss(Xs_k1, Xs_k1_hat)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        total_loss = total_loss + loss.detach()

    return total_loss

def test_model(model, dataloader):
    model.eval()
    device = next(model.parameters()).device
    test_loss = 0
    mseLoss = nn.MSELoss()

    for Xs_k0, W_k0, theta_k0, Xs_k1 in dataloader:
        Xs_k1_hat = model(torch.cat((Xs_k0, W_k0), dim=1).to(device), theta_k0.to(device))
        loss = mseLoss(Xs_k1, Xs_k1_hat)
        test_loss = test_loss + loss.detach()

    return test_loss

def forward_evaluation(model, data):
    '''
    Generate the X_s trajectory for the whole flight having X_s at 0 timestep.
    '''
    model.eval()
    device = next(model.parameters()).device

    Xs_cols = data[0].columns
    Xs_k0, W_k0, theta_k0, Xs_k1 = [Tensor(elem.values) for elem in data]
    Xs_k1_hat = torch.full(Xs_k1.shape, float('nan'))

    Xs_k1_hat[0, :] = model(torch.cat((Xs_k0[0, :], W_k0[0, :])).reshape(1, -1).to(device),
                            theta_k0[0, :].reshape(1, -1).to(device))

    for step in range(1, Xs_k0.shape[0]):
        Xs_k1_hat[step, :] = model(torch.cat((Xs_k1_hat[step-1, :], W_k0[step, :])).reshape(1, -1).to(device),
                                   theta_k0[step, :].reshape(1, -1).to(device))

    # for step in range(Xs_k0.shape[0]):
    #     Xs_k1_hat[step, :] = model(torch.cat((Xs_k0[step, :], W_k0[step, :])).reshape(1, -1).to(device),
    #                                theta_k0[step, :].reshape(1, -1).to(device))
    plot_Xs(Xs_k1, Xs_k1_hat, Xs_cols)
    return None

def plot_Xs(Xs, Xs_hat, cols):
    Xs, Xs_hat = np.array(Xs), np.array(Xs_hat)

    for i, col in enumerate(cols):
        plt.figure(figsize=(10, 5))
        plt.plot(Xs[:, i], label=r'$x_s$')
        plt.plot(Xs_hat[:, i], label=r'$\^x_s$')
        plt.title(col)
        plt.legend()
        plt.xlabel('time step')
        plt.grid()
        plt.savefig(f'plots/forward_prediction_{col}.png', bbox_inches='tight')
        plt.close()
    return None