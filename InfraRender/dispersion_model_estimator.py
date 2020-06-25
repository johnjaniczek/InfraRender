import torch
from utility import compareInterval, get_indices, show_parameters
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as functional
from dispersion_model import DispersionModel
import matplotlib.pyplot as plt
import numpy as np
from torchviz import make_dot


class DispersionModelEstimator(nn.Module):
    def __init__(self, wavenumbers, trueSpectra, learningRate=1e-4, device='cuda',
                 dtype=torch.float64, numModes=2, numFreqs=15, betas=(0.9, 0.999), rho_sparsity=0, out='emissivity'):
        """
        :param wavenumbers: (tensor) sequence of wavenumbers, x axis of the spectra
        :param numFreqs: (int) number of resonant frequencies
        """

        super().__init__()
        self.wavenumbers = wavenumbers
        self.trueSpectra = trueSpectra
        self.device = device
        self.dtype = dtype
        self.dispersionModel = DispersionModel(wavenumbers=wavenumbers, out=out)
        self.dispersionModel.createRandomModes(numModes=numModes, numFreqs=numFreqs)
        self.predictedSpectra = self.dispersionModel.forward()
        self.loss = torch.zeros(1, dtype=dtype, device=device)
        self.prev_loss = torch.zeros_like(self.loss)
        self.cumulativeLoss = 0
        self.learningRate = learningRate
        self.betas = betas
        self.optimizer = Adam(self.parameters(), lr=learningRate, betas=betas)
        self.freqOptimizer = Adam(self.allFreqs(), lr=learningRate*10, betas=betas)
        self.mode_sparsity = None
        self.rho_sparsity = rho_sparsity

    def fit(self, epochs=5000, epsilon=1e-11, display_interval=250, prune_interval=500, min_rho=1e-4, min_gamma=1e-5):
        for epoch in range(epochs):
            self.step()
            if self.convergence(epsilon=epsilon):
                return
            if compareInterval(epoch, display_interval):
                self.display(epoch)
            if compareInterval(epoch, prune_interval):
                self.prune(min_rho=min_rho, min_gamma=min_gamma)

    def prune(self, min_rho=1e-4, min_gamma=1e-5):
        for mode in self.dispersionModel.modes:
            prune_list = get_indices(mode.gammas < min_gamma)
            prune_list.extend(get_indices(mode.rhos < min_rho))
            if len(prune_list) > 0:
                for i in sorted(list(set(prune_list)), reverse=True):
                    mode.remove_params(i)
                self.optimizer = Adam(self.parameters(), lr=self.learningRate, betas=self.betas)
                self.freqOptimizer = Adam(self.allFreqs(), lr=self.learningRate * 1000, betas=self.betas)

    def step(self):
        self.optimizer.zero_grad()
        self.freqOptimizer.zero_grad()
        self.predictedSpectra = self.dispersionModel.forward()
        self.prev_loss = self.loss
        self.loss = self.computeMSE()
        self.cumulativeLoss += self.loss.item()
        self.regularize()
        self.loss.backward()
        self.optimizer.step()#ur a cool guy
        self.freqOptimizer.step()
        self.dispersionModel.apply_constraints()


    def computeMSE(self, stochastic=False):
        if stochastic:
            length = self.predictedSpectra.shape[0]
            indices = np.random.choice(length, int(length/4), replace=False)
            return functional.mse_loss(self.predictedSpectra[indices], self.trueSpectra[indices])
        else:
            return functional.mse_loss(self.predictedSpectra, self.trueSpectra)\

    def regularize(self):
        for mode in self.dispersionModel.modes:
            self.loss += mode.rhos.sum()*self.rho_sparsity

    def display(self, epoch):
        print("\r Epoch:" + str(epoch) + "\t Loss: " + str(self.cumulativeLoss), end='')
        self.cumulativeLoss = 0

    def allFreqs(self):
        return (mode.freqs for mode in self.dispersionModel.modes)

    def convergence(self, epsilon=1e-10):
        return abs(self.prev_loss.item() - self.loss.item()) < epsilon





