from config import *
from utils import *
from models import *
import torch
from torch import Tensor
import joblib
import matplotlib.pyplot as plt
import pandas as pd


print('Preparing data')
Xs_scaler = joblib.load('models/Xs_scaler.save')
w_scaler = joblib.load('models/w_scaler.save')
theta_scaler = joblib.load('models/theta_scaler.save')

Xs_k0, W_k0, theta_k0, Xs_k1, cycle = prepare_data_for_inverse(
    path=ncmapss_path,
    regime=regime,
    Xs_scaler=Xs_scaler,
    w_scaler=w_scaler,
    theta_scaler=theta_scaler,
    eval_unit=eval_unit,
)


print('Reading the model')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepKoopmanControl(model_params).to(device)
model.load_state_dict(torch.load('models/model_epoch_50.pth'))


print('Evaluating the model')
theta_hat, yk0, yk1 = model.inverse(
    torch.cat((Tensor(Xs_k0.values), Tensor(W_k0.values)), dim=1).to(device),
    Tensor(Xs_k1.values).to(device),
)


theta_df = pd.concat((
    theta_k0.reset_index(drop=True),
    pd.DataFrame(theta_hat, columns=theta_k0.columns).add_suffix('_hat'),
    pd.DataFrame(yk0, columns=[f'yk0_{i}' for i in range(yk0.shape[1])]),
    pd.DataFrame(yk1, columns=[f'yk1_{i}' for i in range(yk1.shape[1])]),
    cycle.reset_index(drop=True),
), axis=1)

# theta_df_mean = theta_df.groupby('cycle').mean()
# theta_df_std = theta_df.groupby('cycle').std()
#
# print('Visualizing the results')
# for i, health_parameter in enumerate(theta_k0.columns):
#     plt.figure(figsize=(10, 5))
#     plt.plot(theta_df_mean[health_parameter], label=r'$\theta$')
#     plt.plot(theta_df_mean[health_parameter + '_hat'], alpha=0.8, label=r'$\^\theta$', color='gray')
#     plt.fill_between(
#         theta_df_mean.index,
#         theta_df_mean[health_parameter + '_hat'] - theta_df_std[health_parameter + '_hat'],
#         theta_df_mean[health_parameter + '_hat'] + theta_df_std[health_parameter + '_hat'],
#         alpha=0.5,
#         color='gray',
#     )
#     plt.grid()
#     plt.title(health_parameter)
#     plt.legend()
#     plt.xlabel('cycle')
#     plt.savefig(f'plots/{health_parameter}.png', bbox_inches='tight')
#     plt.close()


theta_sample = theta_df.sample(n=1000)

for i, health_parameter in enumerate(theta_k0.columns):
    for j in range(yk0.shape[1]):
        plt.figure(figsize=(10, 10))
        plt.scatter(theta_sample[health_parameter + '_hat'], theta_sample[f'yk0_{j}'])
        plt.xlabel(f'{health_parameter} pred')
        plt.ylabel(f'Learned latent dimension yk0_{j}')
        plt.savefig(f'plots/scatters/{health_parameter}_vs_yk0_{j}.png', bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(10, 10))
        plt.scatter(theta_sample[health_parameter + '_hat'], theta_sample[f'yk1_{j}'])
        plt.xlabel(f'{health_parameter} pred')
        plt.ylabel(f'Learned latent dimension yk1_{j}')
        plt.savefig(f'plots/scatters/{health_parameter}_vs_yk1_{j}.png', bbox_inches='tight')
        plt.close()