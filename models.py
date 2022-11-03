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

        self.K = nn.Linear(params['obsdim'], params['obsdim'])

        self.B = nn.Linear(params['obsdim'], params['controldim'])

        self.recovery = nn.Linear(params['obsdim'], params['outdim'])

    def forward(self, x0, u):
        yk0 = self.observable(x0)
        Ky0 = self.K(yk0)
        Bu = self.B(u)
        yk1 = torch.add(Ky0, Bu)
        x1 = self.recovery(yk1)
        return x1


