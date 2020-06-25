import torch
import torch.nn as nn
import torch.nn.functional as F
from inverse_render_mixture_model import InverseRenderMixtureModel
from physics import emissivity_model
import torch.optim as optim
import matplotlib.pyplot as plt
from utility import to_numpy

class InverseRender(nn.Module):
    def __init__(self,  wavenumbers=None, param_file=None, dtype=torch.float64, device='cuda'):
        """
        wavenumbers: torch tensor, emissivity is evaluated at each wavenumber passed
        n_freqs: int, number of resonant frequencies in the system
        """
        super().__init__()

        if wavenumbers is None or param_file is None:
            print('must initialize InfraRenderProject with wavenumbers and param file')
            return

        # setup convolutional layers
        self.device = device
        self.dtype = dtype
        self.wavenumbers = wavenumbers
        self.mixture_model = InverseRenderMixtureModel(paramFile=param_file, wavenumbers=wavenumbers,
                                                       dtype=dtype, device=device)
        self.conv1 = nn.Conv1d(1, 3, 5)
        self.conv2 = nn.Conv1d(3, 6, 4)
        self.conv3 = nn.Conv1d(6, 12, 5)
        self.conv4 = nn.Conv1d(12, 24, 4)
        self.pool = nn.MaxPool1d(2)

        # use dummy to compute size of convolutional layer output
        with torch.no_grad():
            dummy = torch.empty((1, wavenumbers.shape[0]))
            dummy = self.convolutions(dummy)
            dummy = torch.flatten(dummy, start_dim=1)

        self.fc1 = nn.Linear(dummy.shape[1], 150)
        self.fc2 = nn.Linear(150, 150)

        # setup fully connected layers and renderer
        self.fc_freqs_dict = nn.ModuleDict()
        self.fc_gammas_dict = nn.ModuleDict()
        self.fc_rhos_dict = nn.ModuleDict()
        self.fc_epsilon_dict = nn.ModuleDict()
        self.fc_mode_weight_dict = nn.ModuleDict()

        for key, endmember in self.mixture_model.endmemberModels.items():
            self.fc_freqs_dict[key] = nn.ModuleList()
            self.fc_gammas_dict[key] = nn.ModuleList()
            self.fc_rhos_dict[key] = nn.ModuleList()
            self.fc_epsilon_dict[key] = nn.ModuleList()
            self.fc_mode_weight_dict[key] = nn.ModuleList()
            for mode_idx, mode in enumerate(endmember.modes):
                self.fc_freqs_dict[key].append(nn.Linear(150, mode.freqs.shape[0]))
                self.fc_gammas_dict[key].append(nn.Linear(150, mode.gammas.shape[0]))
                self.fc_rhos_dict[key].append(nn.Linear(150, mode.rhos.shape[0]))
                self.fc_epsilon_dict[key].append(nn.Linear(150, 1))
                self.fc_mode_weight_dict[key].append(nn.Linear(150, 1))

        self.fc_abundances = nn.Linear(150, self.mixture_model.endmemberModels.__len__())
        self.endmemberSpectra = None
        self.abundances = None
        self.pred_spectra = None
        self.mse = torch.nn.MSELoss(reduction='mean')



    def convolutions(self, x):
        x = F.relu(self.conv1(x.unsqueeze(1)))
        x = F.relu(self.conv2(x))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        return x

    def fully_connected(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

    def predict_parameters(self, x):
        for key, endmember in self.mixture_model.endmemberModels.items():
            for mode_idx, mode in enumerate(endmember.modes):
                endmember.modes[mode_idx].freqs_scalar = F.sigmoid(self.fc_freqs_dict[key][mode_idx](x))
                endmember.modes[mode_idx].gammas_scalar = F.sigmoid(self.fc_gammas_dict[key][mode_idx](x))
                endmember.modes[mode_idx].rhos_scalar = F.sigmoid(self.fc_rhos_dict[key][mode_idx](x))
                endmember.modes[mode_idx].epsilon_scalar = F.sigmoid(self.fc_epsilon_dict[key][mode_idx](x)).squeeze(1)
                endmember.modes[mode_idx].mode_weight_scalar = F.sigmoid(self.fc_mode_weight_dict[key][mode_idx](x)).squeeze(1)
        self.abundances = F.softmax(self.fc_abundances(x))

    def forward(self, spectra):
        x = self.convolutions(spectra)
        x = self.fully_connected(x)
        self.predict_parameters(x)
        self.endmemberSpectra = self.mixture_model.renderEndmembers()
        self.pred_spectra = torch.matmul(self.endmemberSpectra, self.abundances.unsqueeze(2)).squeeze(2)
        return self.pred_spectra, self.abundances

    def criterion(self, pred_spectra, pred_abundances, target_spectra, target_abundances):
        loss = self.mse(pred_spectra, target_spectra)
        loss += self.mse(pred_abundances, target_abundances)
        return loss

    def fit(self, trainloader, epochs=20, learning_rate=0.01, betas=(0.9, 0.999), weight_decay=0):
        self.train()
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay)
        for epoch in range(epochs):  # loop over the dataset multiple times
            self.running_loss = 0.0
            for i, data in enumerate(trainloader):
                self.optimizer.zero_grad()
                spectra = data['spectra'].type(self.dtype).to(self.device)
                abundances = data['abundances'].type(self.dtype).to(self.device)
                pred_spectra, pred_abundances = self.forward(spectra)
                loss = self.criterion(pred_spectra, pred_abundances, spectra, abundances)
                loss.backward()
                self.optimizer.step()
                self.running_loss += loss.item()

            print('[Epoch: %d] loss: %.6f' % (epoch + 1, self.running_loss / (i + 1)))
        print('Finished Training')



    def predict(self, testloader):
        # switch to train mode
        running_loss = 0.0
        spectra_running_loss = 0.0
        abundance_running_loss = 0.0
        N = 0
        self.eval()
        with torch.no_grad():
            for data in testloader:
                spectra = data['spectra'].type(self.dtype).to(self.device)
                abundances = data['abundances'].type(self.dtype).to(self.device)
                pred_spectra, pred_abundances = self.forward(spectra)
                loss = self.criterion(pred_spectra, pred_abundances, spectra, abundances)
                running_loss += loss.item()
                spectra_running_loss += self.mse(pred_spectra, spectra).item()
                abundance_running_loss += ((pred_abundances - abundances) ** 2).sum(dim=1).mean().item()
                N+=1

                # plt.figure()
                # plt.plot(to_numpy(self.wavenumbers), to_numpy(spectra[0]))
                # plt.plot(to_numpy(self.wavenumbers), to_numpy(pred_spectra[0]), '--')
                # plt.show()


        print('average spectra error', spectra_running_loss/N)
        print('average abundance error', abundance_running_loss/N)
        print('average abundance error per class', abundance_running_loss/(N*(abundances.shape[1])))
        print('average error over all parameters', running_loss/N)
        print('Finished Testing')