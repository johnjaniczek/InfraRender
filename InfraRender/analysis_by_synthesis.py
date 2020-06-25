import torch
import torch.nn as nn
import torch.nn.functional as functional
import numpy as np
import matplotlib.pyplot as plt
from dispersion_model import DispersionModel
from dispersion_model_dict import DispersionModelDict
from torch.optim import Adam
from scipy.optimize import nnls
from utility import clamp_with_tensor, to_numpy, compareInterval
from linear_unmixing import pNorm_unmix
import h5py


# noinspection PyPep8Naming
class AnalysisBySynthesis(nn.Module):
    def __init__(self, paramFile=None, wavenumbers=None, p=0.99, lam=0.001, dtype=torch.float64, device='cpu'):
        super().__init__()
        self.wavenumbers = wavenumbers
        self.num_wavenumbers = wavenumbers.shape[0]
        endmemberModels = DispersionModelDict(paramFile, wavenumbers=wavenumbers, dtype=dtype, device=device)
        self.num_endmembers = len(endmemberModels.keys())
        self.endmemberModels = nn.ModuleDict(modules=endmemberModels)
        self.endmemberSpectra = torch.empty((self.num_wavenumbers, self.num_endmembers), dtype=dtype, device=device)
        self.abundances = torch.empty(len(endmemberModels.keys()), dtype=dtype, device=device)
        self.trueSpectra = None
        self.predictedSpectra = torch.empty_like(self.wavenumbers)
        self.loss = None
        self.optimizer = None
        self.cumulativeLoss = 0
        self.p = p
        self.p_lambda = lam
        self.dtype = dtype
        self.device = device

    def fit(self, trueSpectra, epochs=500, learningRate=1e-4, betas=(0.9, 0.999)):
        self.trueSpectra = trueSpectra
        self.optimizer = Adam(self.parameters(), lr=learningRate, betas=betas)
        for epoch in range(epochs):
            self.step()
            if compareInterval(epoch, 5):
                self.display(epoch)

    def forward(self):
        self.endmemberSpectra = self.renderEndmembers()
        pnorm_abundances = pNorm_unmix(to_numpy(self.endmemberSpectra), to_numpy(self.trueSpectra),
                                       lam=self.p_lambda, p=self.p)
        self.abundances = torch.from_numpy(pnorm_abundances).type(self.dtype).to(self.device)
        self.predictedSpectra = torch.matmul(self.endmemberSpectra, self.abundances)
        return self.predictedSpectra

    def step(self):
        self.optimizer.zero_grad()
        self.forward()
        self.loss = functional.mse_loss(self.predictedSpectra, self.trueSpectra)
        self.loss += self.compute_regularization()
        self.cumulativeLoss += self.loss.item()
        self.loss.backward()
        self.optimizer.step()
        self.apply_constraints()


    def renderEndmembers(self):
        endmemberSpectra = torch.empty_like(self.endmemberSpectra)
        for i, model in enumerate(self.endmemberModels.values()):
            endmemberSpectra[:, i] = model.forward()
        return endmemberSpectra

    def apply_constraints(self):
        for endmember in self.endmemberModels.values():
            endmember.apply_constraints()

    def compute_regularization(self):
        return self.p_lambda * torch.norm(self.abundances, self.p)

    def display(self, epoch):
        print("\r Epoch:" + str(epoch) + "\t Loss: " + str(self.cumulativeLoss), end='')
        self.cumulativeLoss = 0

    def set_default_params(self):
        for model in self.endmemberModels.values():
            model.default_params()

    def write_results(self, file, group_name='default'):
        f = h5py.File(file, 'a')
        if group_name in f.keys():
            grp = f[group_name]
            grp.clear()
        else:
            grp = f.create_group(group_name)

        grp['spectra'] = to_numpy(self.renderEndmembers())
        grp['abundances'] = to_numpy(self.abundances)
        f.close()

    def fit_to_csv(self, trueSpectra, directory, epochs=500, learningRate=1e-4, betas=(0.9, 0.999)):
        self.trueSpectra = trueSpectra
        self.optimizer = Adam(self.parameters(), lr=learningRate, betas=betas)
        for epoch in range(epochs):
            cur_filename = directory + 'generated_endmembers' + str(epoch) + '.csv'
            cur_endmembers = to_numpy(self.renderEndmembers())
            np.savetxt(cur_filename, cur_endmembers, delimiter=',')
            self.step()
            if compareInterval(epoch, 5):
                self.display(epoch)



