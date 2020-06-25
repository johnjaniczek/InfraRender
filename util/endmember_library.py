import numpy as np
import pandas as pd


class EndmemberLibrary(object):
    def __init__(self, ascii_spectra="", meta_csv=""):
        """
        expects endmember spectra ascii and meta csv file per asu thermal emission spectral library format
        """
        self.spectra = np.loadtxt(ascii_spectra)
        self.spectra = np.delete(self.spectra, 0, 1)  # delete first spectra column (wavenumbers)
        self.bands = np.loadtxt(ascii_spectra, usecols=0)
        self.meta = pd.read_csv(meta_csv)
        self.names = self.meta.sample_name.tolist()
        self.category = self.meta.category.tolist()

    def returnDict(self):
        endmemberDict = {}
        for i, name in enumerate(self.names):
            endmemberDict[name] = self.spectra[:, i]
        return endmemberDict



