from datasets import NCMAPSS
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset


def prepare_ncmapss_data(**params):
    dataset = NCMAPSS(params['path'])
    dataset.keep_regime(params['regime'])
    dataset.keep_train_units(params['train_units'])
    dataset.keep_test_units(params['test_units'])
    dataset.subsample_cycles(frac=params['cycles_frac'])
    Xs_normalizer = dataset.normalize(dataset.X_s_var)
    w_normalizer = dataset.normalize(dataset.W_var)
    theta_normalizer = dataset.normalize(dataset.T_var)
    dataset.shift_data()

    train_Xs_k0, test_Xs_k0 = dataset.get_data(dataset.X_s_var)
    train_W_k0, test_W_k0 = dataset.get_data(dataset.W_var)
    train_theta_k0, test_theta_k0 = dataset.get_data(dataset.T_var)
    train_Xs_k1, test_Xs_k1 = dataset.get_data([col + dataset.shift_suffix for col in dataset.X_s_var])

    dataset = TensorDataset(torch.Tensor(train_Xs_k0.values),
                            torch.Tensor(train_W_k0.values),
                            torch.Tensor(train_theta_k0.values),
                            torch.Tensor(train_Xs_k1.values))
    training_loader = DataLoader(dataset, batch_size=params['bs'], shuffle=False, drop_last=True)

    dataset = TensorDataset(torch.Tensor(test_Xs_k0.values),
                            torch.Tensor(test_W_k0.values),
                            torch.Tensor(test_theta_k0.values),
                            torch.Tensor(test_Xs_k1.values))
    testing_loader = DataLoader(dataset, batch_size=params['bs'], shuffle=False, drop_last=False)
    return training_loader, testing_loader, (Xs_normalizer, w_normalizer, theta_normalizer)


def train_model(model, optimizer, dataloader):
    '''
    Trains model for a single epoch.
    '''
    model.train()
    total_loss = 0
    mseLoss = nn.MSELoss()

    for Xs_k0, W_k0, theta_k0, Xs_k1 in dataloader:
        Xs_k1_hat = model(torch.cat((Xs_k0, W_k0), dim=1), theta_k0)

        loss = mseLoss(Xs_k1, Xs_k1_hat)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss = total_loss + loss.detach()

    return total_loss

def test_model(model, dataloader):
    model.eval()
    test_loss = 0
    mseLoss = nn.MSELoss()

    for Xs_k0, W_k0, theta_k0, Xs_k1 in dataloader:
        Xs_k1_hat = model(torch.cat((Xs_k0, W_k0), dim=1), theta_k0)
        loss = mseLoss(Xs_k1, Xs_k1_hat)
        test_loss = test_loss + loss.detach()

    return test_loss

