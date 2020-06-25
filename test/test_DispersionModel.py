from unittest import TestCase
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from physics import emissivity_model, create_wavenumbers
from InfraRender.single_axis_vibration import SingleAxisVibration
from InfraRender import DispersionModel
from InfraRender import DispersionModelDict


class TestDispersionModel(TestCase):
    def setUp(self) -> None:
        self.mode_0 = {'freqs': torch.tensor([1215., 1161., 1067., 795., 694., 449., 393.], dtype=torch.float64),
                      'gammas': torch.tensor([0.4, 0.007, 0.008, 0.0115, 0.01, 0.012, 0.012], dtype=torch.float64),
                      'four_pr': torch.tensor([0.03, 0.007, 0.67, 0.11, 0.01, 0.815, 0.4], dtype=torch.float64),
                      'modeWeight': 0.33,
                       'epsilon': torch.tensor([2.356], dtype=torch.float64)}
        self.mode_1 = {'freqs': torch.tensor([1222., 1074., 776., 509., 494., 364.], dtype=torch.float64),
                      'gammas': torch.tensor([0.15, 0.008, 0.011, 0.016, 0.014, 0.019], dtype=torch.float64),
                      'four_pr': torch.tensor([0.011, 0.67, 0.1, 0.015, 0.7, 0.68], dtype=torch.float64),
                      'modeWeight': 0.67,
                       'epsilon': torch.tensor([2.383], dtype=torch.float64)}

        self.modeWeights = [torch.tensor([0.33], dtype=torch.float64), torch.tensor([0.67], dtype=torch.float64)]
        numberOfWaves = 2000
        self.wavenumbers = create_wavenumbers(numberOfWaves)

        # compute directly the true emissivity of multi-modal dispersion model
        mode0_spectra = emissivity_model(self.mode_0['freqs'], self.mode_0['four_pr'],
                                         self.mode_0['gammas'], self.wavenumbers, self.mode_0['epsilon'])
        mode1_spectra = emissivity_model(self.mode_1['freqs'], self.mode_1['four_pr'],
                                         self.mode_1['gammas'], self.wavenumbers, self.mode_0['epsilon'])
        self.trueEmissivity = mode0_spectra*self.mode_0['modeWeight'] + mode1_spectra*self.mode_1['modeWeight']

        # construct multi-axis dispersion model
        self.modelMode0 = SingleAxisVibration(self.wavenumbers, self.mode_0['freqs'], gammas=self.mode_0['gammas'],
                                        rhos=self.mode_0['four_pr'], mode_weight=self.modeWeights[0],
                                        epsilon=self.mode_0['epsilon'])
        self.modelMode1 = SingleAxisVibration(self.wavenumbers, self.mode_1['freqs'], gammas=self.mode_1['gammas'],
                                        rhos=self.mode_1['four_pr'], mode_weight=self.modeWeights[1],
                                        epsilon=self.mode_0['epsilon'])


        self.model = DispersionModel(wavenumbers=self.wavenumbers)
        modes = [self.modelMode0, self.modelMode1]
        self.model.modes = nn.ModuleList(modules=modes)
        self.model.forward()

    def testSpectra(self) -> None:
        # compare torch model computation, vs. standard computation
        filename = '../input/dispersionModelParameters/multiAxisTest.hdf'
        self.assertTrue(torch.allclose(self.trueEmissivity, self.model.spectra))
        self.model.write_hdf(filename, modelName='test')
        modelDict = DispersionModelDict(filename, wavenumbers=self.wavenumbers)
        modelFromHdf = modelDict['test']
        self.assertTrue(torch.allclose(self.trueEmissivity, modelFromHdf.spectra))

    def testHdf(self) -> None:
        self.model.write_hdf('../input/dispersionModelParameters/multiAxisTest.hdf', modelName='test')
        modelDict = DispersionModelDict('../input/dispersionModelParameters/multiAxisTest.hdf',
                                        wavenumbers=self.wavenumbers)
        modelFromHdf = modelDict['test']
        modelFromHdf.forward()
        self.assertTrue(torch.allclose(self.trueEmissivity, modelFromHdf.spectra, atol=1e-10))





