import h5py
from dispersion_model import DispersionModel
import torch

class DispersionModelDict(dict):
    def __init__(self, filename, wavenumbers=None, dtype=torch.float64, device='cpu', learning='optimization'):
        super().__init__()
        f = h5py.File(filename, 'r')
        for modelName, modeGroup in f.items():
            self[modelName] = DispersionModel(mode_hdf_group=modeGroup, wavenumbers=wavenumbers,
                                              dtype=dtype, device=device, learning=learning)


