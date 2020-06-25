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
import h5py


class InverseRenderMixtureModel(nn.Module):
    def __init__(self, paramFile=None, wavenumbers=None, batch_size=None, dtype=torch.float64, device='cpu'):
        super().__init__()
        self.wavenumbers = wavenumbers
        self.num_wavenumbers = wavenumbers.shape[0]
        endmemberModels = DispersionModelDict(paramFile, wavenumbers=wavenumbers, dtype=dtype, device=device, learning='InfraRenderProject')
        self.num_endmembers = len(endmemberModels.keys())
        self.endmemberModels = nn.ModuleDict(modules=endmemberModels)
        self.endmemberSpectra = None
        self.abundances = torch.empty(len(endmemberModels.keys()), dtype=dtype, device=device)
        self.predictedSpectra = torch.empty_like(self.wavenumbers)
        self.cumulativeLoss = 0
        self.dtype = dtype
        self.device = device

    def forward(self):
        self.endmemberSpectra = self.renderEndmembers()
        self.predictedSpectra = torch.matmul(self.endmemberSpectra, self.abundances)
        return self.predictedSpectra

    def renderEndmembers(self):
        endmemberSpectra = []
        for i, model in enumerate(self.endmemberModels.values()):
            spectra = model.forward().t()
            endmemberSpectra.append(spectra)
        return torch.stack(endmemberSpectra, dim=2)

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







