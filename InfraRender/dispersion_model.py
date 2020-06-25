import torch
import torch.nn as nn
import h5py
from single_axis_vibration import SingleAxisVibration, create_from_file


class DispersionModel(nn.Module):
    def __init__(self, mode_hdf_group=None, wavenumbers=None, device='cpu', dtype=torch.float64, out='emissivity',
                 learning='optimization'):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.modes = nn.ModuleList()
        self.spectra = None
        self.wavenumbers = wavenumbers
        self.out = out
        self.learning = learning

        if mode_hdf_group is not None:
            self.create_modes_from_hdf_group(mode_hdf_group)
            self.apply_constraints()

    def forward(self):
        if self.learning == 'InfraRenderProject':
            self.scale_params()
        for i, mode in enumerate(self.modes):
            if i == 0:
                self.spectra = mode.forward(out=self.out)
            else:
                self.spectra += mode.forward(out=self.out)
        return self.spectra

    def apply_constraints(self):
        for mode in self.modes:
            mode.constrain_params()
        self.normalizeModeWeights()

    def normalizeModeWeights(self):
        total = 0
        for mode in self.modes:
            total += mode.mode_weight.data
        for mode in self.modes:
            mode.mode_weight.data /= total

    def removeMode(self, i):
        self.modes.__delitem__(i)

    def create_modes_from_hdf_group(self, modes):
        for i, key in enumerate(modes):
            modelParams = modes[key]
            model = create_from_file(modelParams, wavenumbers=self.wavenumbers,
                                     dtype=self.dtype, device=self.device, learning=self.learning)
            self.modes.append(model)

    def createRandomModes(self, numModes=2, numFreqs=15):
        for i in range(numModes):
            mode = self.createRandomMode(numFreqs)
            self.modes.append(mode)

    def createRandomMode(self, numFreqs):
        minFreq, maxFreq = max(0, self.wavenumbers.min()*0.8), self.wavenumbers.max()*1.2
        freqs = torch.empty(numFreqs, dtype=self.dtype, device=self.device).uniform_(minFreq, maxFreq)
        model = SingleAxisVibration(freqs=freqs, wavenumbers=self.wavenumbers, device=self.device, dtype=self.dtype)
        return model

    def set_constraint_tolerance(self, freq_tolerance=None, gamma_tolerance=None,
                                 rho_tolerance=None, epsilon_tolerance=None, mode_weight_tolerance=None):
        for mode in self.modes:
            mode.set_constraint_tolerance(freq_tolerance=freq_tolerance, gamma_tolerance=gamma_tolerance,
                                          rho_tolerance=rho_tolerance, epsilon_tolerance=epsilon_tolerance,
                                          mode_weight_tolerance=mode_weight_tolerance)

    def write_hdf(self, file, modelName='default'):
        f = h5py.File(file, 'a')
        if modelName in f.keys():
            grp = f[modelName]
            grp.clear()
        else:
            grp = f.create_group(modelName)

        for i, mode in enumerate(self.modes):
            name = 'mode' + str(i)
            modeGroup = grp.create_group(name)
            mode.write_hdf_group(modeGroup)
        f.close()

    def perturb_params(self):
        def rand_scale(param, param_min, param_max):
            param.data = torch.rand_like(param) * (param_max - param_min) + param_min
        for mode in self.modes:
            rand_scale(mode.freqs, mode.freqs_min, mode.freqs_max)
            rand_scale(mode.gammas, mode.gammas_min, mode.gammas_max)
            rand_scale(mode.rhos, mode.rhos_min, mode.rhos_max)
            rand_scale(mode.mode_weight, mode.mode_weight_min, mode.mode_weight_max)
            rand_scale(mode.epsilon, mode.epsilon_min, mode.epsilon_max)

    def default_params(self):
        def default_scale(param, param_min, param_max):
            param.data = (param_max - param_min) / 2
        for mode in self.modes:
            default_scale(mode.freqs, mode.freqs_min, mode.freqs_max)
            default_scale(mode.gammas, mode.gammas_min, mode.gammas_max)
            default_scale(mode.rhos, mode.rhos_min, mode.rhos_max)
            default_scale(mode.mode_weight, mode.mode_weight_min, mode.mode_weight_max)
            default_scale(mode.epsilon, mode.epsilon_min, mode.epsilon_max)

    def scale_params(self):
        def scale(scalar, param_min, param_max):
            return scalar * (param_max - param_min) + param_min
        for mode in self.modes:
            mode.freqs = scale(mode.freqs_scalar, mode.freqs_min, mode.freqs_max)
            mode.gammas = scale(mode.gammas_scalar, mode.gammas_min, mode.gammas_max)
            mode.rhos = scale(mode.rhos_scalar, mode.rhos_min, mode.rhos_max)
            mode.mode_weight = scale(mode.mode_weight_scalar, mode.mode_weight_min, mode.mode_weight_max)
            mode.epsilon = scale(mode.epsilon_scalar, mode.epsilon_min, mode.epsilon_max)



