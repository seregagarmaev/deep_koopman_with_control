import torch
import torch.nn as nn


class DeepKoopmanControl(nn.Module):
    def __init__(self, params):
        super(DeepKoopmanControl, self).__init__()
        self.observable = nn.Sequential(
            nn.Linear(params['indim'], params['hidden_dim']),
            nn.SELU(inplace=True),
            nn.Linear(params['hidden_dim'], params['hidden_dim']),
            nn.SELU(inplace=True),
            nn.Linear(params['hidden_dim'], params['obsdim'])
        )

        self.K = nn.Linear(params['obsdim'], params['obsdim'], bias=False)

        self.B = nn.Linear(params['controldim'], params['obsdim'], bias=False)
        self.B_inv = nn.Linear(params['obsdim'], params['controldim'], bias=False)

        self.recovery = nn.Linear(params['obsdim'], params['outdim'], bias=False)
        self.recovery_inv = nn.Linear(params['obsdim'], params['outdim'], bias=False)

    def forward(self, x0, u):
        yk0 = self.observable(x0)
        Ky0 = self.K(yk0)
        Bu = self.B(u)
        yk1 = torch.add(Ky0, Bu)
        x1 = self.recovery(yk1)
        return x1

    def inverse(self, x0, x1):
        yk0 = self.observable(x0)
        Ky0 = self.K(yk0)

        self.recovery_inv.weight = nn.Parameter(torch.linalg.inv(self.recovery.weight))
        yk1 = self.recovery_inv(x1)

        self.B_inv.weight = nn.Parameter(torch.linalg.pinv(self.B.weight))
        theta_hat = self.B_inv(yk1 - Ky0)
        return theta_hat.detach(), yk0.detach(), yk1.detach()