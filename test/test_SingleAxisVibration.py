from unittest import TestCase
import torch
import numpy as np
import matplotlib.pyplot as plt
from physics import emissivity_model, create_wavenumbers
from InfraRender.single_axis_vibration import SingleAxisVibration, create_from_file
import h5py

def readDispersionModelFromHdf(filename, modelName='default'):
    f = h5py.File(filename, 'r')
    modelParams = f[modelName]
    model = create_from_file(modelParams)
    f.close()
    return model

class TestDispersionModel(TestCase):
    def setUp(self) -> None:
        self.trueFreqs = torch.tensor([1215., 1161., 1067., 795., 694., 449., 393.], dtype=torch.float64)
        self.trueGammas = torch.tensor([0.4, 0.007, 0.008, 0.0115, 0.01, 0.012, 0.012], dtype=torch.float64)
        self.trueFour_pr = torch.tensor([0.03, 0.007, 0.67, 0.11, 0.01, 0.815, 0.4], dtype=torch.float64)
        self.true_epsilon = torch.tensor([2.356], dtype=torch.float64)
        numberOfWaves = 2000
        self.wavenumbers = create_wavenumbers(numberOfWaves)
        self.trueEmissivity = emissivity_model(self.trueFreqs, self.trueFour_pr, self.trueGammas, self.wavenumbers,
                                               self.true_epsilon)
        self.model = SingleAxisVibration(wavenumbers=self.wavenumbers, freqs=self.trueFreqs,
                                   gammas=self.trueGammas, rhos=self.trueFour_pr, epsilon=self.true_epsilon)

    def testSpectra(self) -> None:
        self.assertTrue(torch.allclose(self.trueEmissivity, self.model.spectra))

    def testHdf(self) -> None:
        self.model.write_hdf('../input/dispersionModelParameters/test.hdf', modelName='test')
        modelFromHdf = readDispersionModelFromHdf('../input/dispersionModelParameters/test.hdf', modelName='test')
        self.assertTrue(torch.allclose(self.trueEmissivity, modelFromHdf.spectra))





