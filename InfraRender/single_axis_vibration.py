import torch
from physics import emissivity_model, reflectivity_model
import torch.nn as nn
import h5py
from utility import create_wavenumbers, clamp_with_tensor, to_numpy


class SingleAxisVibration(nn.Module):
    def __init__(self, wavenumbers=None, freqs=None, gammas=None, rhos=None, epsilon=None,
                 freqs_max=None, freqs_min=None, gammas_max=None, gammas_min=None,
                 rhos_max=None, rhos_min=None, epsilon_max=None, epsilon_min=None,
                 mode_weight=None, mode_weight_max=None, mode_weight_min=None,
                 device='cpu', dtype=torch.float64, learning='optimization'):
        super().__init__()

        if wavenumbers is None:
            wavenumbers = create_wavenumbers(2000).type(dtype).to(device)
        if freqs is None:
            freqs = torch.empty(15, dtype=dtype, device=device).uniform_(wavenumbers.min(), wavenumbers.max())
        if freqs_max is None:
            freqs_max = torch.full_like(freqs, wavenumbers.max()*1.15)
        if freqs_min is None:
            freqs_min = torch.full_like(freqs, max(0, wavenumbers.min()*.85))
        if gammas is None:
            gammas = torch.empty_like(freqs).uniform_(1e-2, 0.1)
        if gammas_max is None:
            gammas_max = torch.full_like(freqs, 3)
        if gammas_min is None:
            gammas_min = torch.full_like(freqs, 1e-6)
        if rhos is None:
            rhos = torch.empty_like(freqs).uniform_(1e-2, 0.1)
        if rhos_max is None:
            rhos_max = torch.full_like(freqs, 3)
        if rhos_min is None:
            rhos_min = torch.full_like(freqs, 1e-7)
        if epsilon is None:
            epsilon = torch.empty(1).uniform_(1, 3).type(dtype).to(device)
        if epsilon_max is None:
            epsilon_max = torch.full_like(epsilon, 8)
        if epsilon_min is None:
            epsilon_min = torch.full_like(epsilon, 1)
        if mode_weight is None:
            mode_weight = torch.ones(1, dtype=dtype, device=device)
        if mode_weight_max is None:
            mode_weight_max = torch.full((1,), 1, dtype=dtype, device=device)
        if mode_weight_min is None:
            mode_weight_min = torch.zeros(1, dtype=dtype, device=device)

        self.learning = learning
        self.freqs = nn.Parameter(freqs) if self.learning == 'optimization' else freqs
        self.gammas = nn.Parameter(gammas) if self.learning == 'optimization' else gammas
        self.rhos = nn.Parameter(rhos) if self.learning == 'optimization' else rhos
        self.epsilon = nn.Parameter(epsilon) if self.learning == 'optimization' else epsilon
        self.mode_weight = nn.Parameter(mode_weight) if self.learning == 'optimization' else mode_weight

        self.device = device
        self.dtype = dtype
        self.wavenumbers = wavenumbers if self.learning == 'optimization' else wavenumbers.unsqueeze(-1)

        self.freqs_max = freqs_max
        self.freqs_min = freqs_min
        self.gammas_max = gammas_max
        self.gammas_min = gammas_min
        self.rhos_max = rhos_max
        self.rhos_min = rhos_min
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.mode_weight_max = mode_weight_max
        self.mode_weight_min = mode_weight_min
        self.device = device
        self.dtype = dtype

        self.freqs_scalar = torch.full_like(self.freqs, 0.5)
        self.gammas_scalar = torch.full_like(self.gammas, 0.5)
        self.rhos_scalar = torch.full_like(self.rhos, 0.5)
        self.mode_weight_scalar = torch.full_like(self.mode_weight, 0.5)
        self.epsilon_scalar = torch.full_like(self.epsilon, 0.5)

        self.spectra_unscaled = None
        self.spectra = None
        self.forward()

    def forward(self, out='emissivity'):
        if out == 'emissivity':
            self.spectra_unscaled = emissivity_model(self.freqs, self.rhos, self.gammas, self.wavenumbers, self.epsilon)
        elif out == 'reflectivity':
            self.spectra_unscaled = reflectivity_model(self.freqs, self.rhos, self.gammas, self.wavenumbers, self.epsilon)
        self.spectra = self.spectra_unscaled * self.mode_weight
        return self.spectra

    def sort_freqs(self):
        def sort_by_args(argsort):
            def sort(x):
                x.data = x[argsort]
            return sort
        argsort = torch.argsort(self.freqs)
        sort = sort_by_args(argsort)
        sort(self.freqs)
        sort(self.gammas)
        sort(self.rhos)
        sort(self.freqs_max)
        sort(self.freqs_min)
        sort(self.gammas_max)
        sort(self.gammas_min)
        sort(self.rhos_max)
        sort(self.rhos_min)

    def constrain_params(self):
        clamp_with_tensor(self.freqs, min=self.freqs_min, max=self.freqs_max)
        clamp_with_tensor(self.rhos, min=self.rhos_min, max=self.rhos_max)
        clamp_with_tensor(self.gammas, min=self.gammas_min, max=self.gammas_max)
        clamp_with_tensor(self.mode_weight, min=self.mode_weight_min, max=self.mode_weight_max)
        clamp_with_tensor(self.epsilon, min=self.epsilon_min, max=self.epsilon_max)

    def remove_params(self, i):
        freqs = torch.cat((self.freqs[:i], self.freqs[i + 1:]))
        gammas= torch.cat((self.gammas[:i], self.gammas[i + 1:]))
        rhos = torch.cat((self.rhos[:i], self.rhos[i + 1:]))
        self.freqs = nn.Parameter(freqs) if self.learning == 'optimization' else freqs
        self.gammas = nn.Parameter(gammas) if self.learning == 'optimization' else gammas
        self.rhos = nn.Parameter(rhos) if self.learning == 'optimization' else rhos
        self.freqs_max = torch.cat((self.freqs_max[:i], self.freqs_max[i + 1:]))
        self.freqs_min = torch.cat((self.freqs_min[:i], self.freqs_min[i + 1:]))
        self.gammas_max = torch.cat((self.gammas_max[:i], self.gammas_max[i + 1:]))
        self.gammas_min = torch.cat((self.gammas_min[:i], self.gammas_min[i + 1:]))
        self.rhos_max = torch.cat((self.rhos_max[:i], self.rhos_max[i + 1:]))
        self.rhos_min = torch.cat((self.rhos_min[:i], self.rhos_min[i + 1:]))

    def add_params(self, freq):
        self.freqs_max = torch.cat((self.freqs_max, self.freqs_max[-1].unsqueeze(0)))
        self.freqs_min = torch.cat((self.freqs_min, self.freqs_min[-1].unsqueeze(0)))
        freqs = torch.cat((self.freqs, freq.unsqueeze(0)))
        self.gammas_max = torch.cat((self.gammas_max, self.gammas_max[-1].unsqueeze(0)))
        self.gammas_min = torch.cat((self.gammas_min, self.gammas_min[-1].unsqueeze(0)))
        new_gamma = torch.empty_like(freq).uniform_(0.01, 0.1)
        gammas = torch.cat((self.gammas, new_gamma.unsqueeze(0)))
        self.rhos_max = torch.cat((self.rhos_max, self.rhos_max[-1].unsqueeze(0)))
        self.rhos_min = torch.cat((self.rhos_min, self.rhos_min[-1].unsqueeze(0)))
        new_rho = torch.empty_like(freq).uniform_(0.01, .1)
        rhos = torch.cat((self.rhos, new_rho.unsqueeze(0)))

        self.freqs = nn.Parameter(freqs) if self.learning == 'optimization' else freqs
        self.gammas = nn.Parameter(gammas) if self.learning == 'optimization' else gammas
        self.rhos = nn.Parameter(rhos) if self.learning == 'optimization' else rhos

    def set_constraint_tolerance(self, freq_tolerance=None, gamma_tolerance=None,
                                 rho_tolerance=None, epsilon_tolerance=None, mode_weight_tolerance=None):
        if freq_tolerance is not None:
            self.freqs_min = self.freqs - self.freqs * freq_tolerance
            self.freqs_max = self.freqs + self.freqs * freq_tolerance
        if gamma_tolerance is not None:
            self.gammas_min = self.gammas - self.gammas * gamma_tolerance
            self.gammas_max = self.gammas + self.gammas * gamma_tolerance
        if rho_tolerance is not None:
            self.rhos_min = self.rhos - self.rhos * rho_tolerance
            self.rhos_max = self.rhos + self.rhos * rho_tolerance
        if epsilon_tolerance is not None:
            self.epsilon_min = self.epsilon - self.epsilon*epsilon_tolerance
            self.epsilon_max = self.epsilon + self.epsilon*epsilon_tolerance
        if mode_weight_tolerance is not None:
            self.mode_weight_min = self.mode_weight - self.mode_weight * mode_weight_tolerance
            self.mode_weight_max = self.mode_weight + self.mode_weight * mode_weight_tolerance

    def write_hdf(self, file, modelName='default'):
        f = h5py.File(file, 'a')
        if modelName in f.keys():
            grp = f[modelName]
            grp.clear()
        else:
            grp = f.create_group(modelName)
        self.write_hdf_group(grp)
        f.close()

    def write_hdf_group(self, group):
        group['wavenumbers'] = to_numpy(self.wavenumbers)
        group['freqs'] = to_numpy(self.freqs)
        group['gammas'] = to_numpy(self.gammas)
        group['rhos'] = to_numpy(self.rhos)
        group['epsilon'] = to_numpy(self.epsilon)
        group['freqs_max'] = to_numpy(self.freqs_max)
        group['gammas_max'] = to_numpy(self.gammas_max)
        group['rhos_max'] = to_numpy(self.rhos_max)
        group['freqs_min'] = to_numpy(self.freqs_min)
        group['gammas_min'] = to_numpy(self.gammas_min)
        group['rhos_min'] = to_numpy(self.rhos_min)
        group['epsilon_min'] = to_numpy(self.epsilon_min)
        group['epsilon_max'] = to_numpy(self.epsilon_max)
        group['mode_weight'] = to_numpy(self.mode_weight)
        group['mode_weight_max'] = to_numpy(self.mode_weight_max)
        group['mode_weight_min'] = to_numpy(self.mode_weight_min)


def create_from_file(paramFile, wavenumbers=None, dtype=torch.float64, device='cpu', learning='optimization'):
    if wavenumbers is None:
        wavenumbers = torch.from_numpy(paramFile['wavenumbers'][:])
    freqs = torch.from_numpy(paramFile['freqs'][:]).type(dtype).to(device)
    gammas = torch.from_numpy(paramFile['gammas'][:]).type(dtype).to(device)
    rhos = torch.from_numpy(paramFile['rhos'][:]).type(dtype).to(device)
    epsilon = torch.from_numpy(paramFile['epsilon'].value).type(dtype).to(device)
    freqs_max = torch.from_numpy(paramFile['freqs_max'][:]).type(dtype).to(device)
    gammas_max = torch.from_numpy(paramFile['gammas_max'][:]).type(dtype).to(device)
    rhos_max = torch.from_numpy(paramFile['rhos_max'][:]).type(dtype).to(device)
    epsilon_max = torch.from_numpy(paramFile['epsilon_max'].value).type(dtype).to(device)
    freqs_min = torch.from_numpy(paramFile['freqs_min'][:]).type(dtype).to(device)
    gammas_min = torch.from_numpy(paramFile['gammas_min'][:]).type(dtype).to(device)
    rhos_min = torch.from_numpy(paramFile['rhos_min'][:]).type(dtype).to(device)
    epsilon_min = torch.from_numpy(paramFile['epsilon_min'].value).type(dtype).to(device)
    mode_weight = torch.tensor(paramFile['mode_weight'].value).type(dtype).to(device)
    mode_weight_max = torch.tensor(paramFile['mode_weight_max'].value).type(dtype).to(device)
    mode_weight_min = torch.tensor(paramFile['mode_weight_min'].value).type(dtype).to(device)
    model = SingleAxisVibration(wavenumbers=wavenumbers, freqs=freqs, gammas=gammas, rhos=rhos, epsilon=epsilon,
                                freqs_max=freqs_max, gammas_max=gammas_max, rhos_max=rhos_max, epsilon_max=epsilon_max,
                                freqs_min=freqs_min, gammas_min=gammas_min, rhos_min=rhos_min, epsilon_min=epsilon_min,
                                mode_weight=mode_weight, mode_weight_min=mode_weight_min,
                                mode_weight_max = mode_weight_max,
                                dtype=dtype, device=device, learning=learning)
    return model

