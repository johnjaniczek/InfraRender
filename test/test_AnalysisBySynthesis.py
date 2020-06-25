from unittest import TestCase
import torch
import numpy as np
from InfraRender import DispersionModel
from util import to_numpy
from util import create_wavenumbers
from InfraRender import AnalysisBySynthesis
import matplotlib.pyplot as plt
import h5py
import sys
device = 'cpu'

class TestAnalysisBySynthesis(TestCase):
    def setUp(self) -> None:
        # test parameters
        num_endmembers = 5
        num_modes = 2
        num_freqs = 7
        self.wavenumbers = create_wavenumbers(2000).type(torch.float64).to(device)
        self.modelFilename = '../input/dispersionModelParameters/UnmixTest.hdf'

        # clear test file, to start from scratch
        f = h5py.File(self.modelFilename, 'a')
        f.clear()
        f.close()

        for i in range(num_endmembers):
            model = DispersionModel(device=device, wavenumbers=self.wavenumbers)
            model.createRandomModes(numModes=num_modes, numFreqs=num_freqs)
            model.set_constraint_tolerance(freq_tolerance=0.025)
            model.apply_constraints()
            model.write_hdf(self.modelFilename, modelName=str(i))

    def testInit(self) -> None:
        self.unmixModel = AnalysisBySynthesis(paramFile=self.modelFilename, wavenumbers=self.wavenumbers,
                                      dtype=torch.float64, device=device)

    def testFit(self):
        self.unmixModel = AnalysisBySynthesis(paramFile=self.modelFilename, wavenumbers=self.wavenumbers,
                                      dtype=torch.float64, device=device)
        trueAbundances = torch.empty(len(self.unmixModel.endmemberModels), dtype=torch.float64, device=device).uniform_(0, 1)
        trueAbundances /= trueAbundances.sum()
        trueSpectra = torch.zeros(len(self.wavenumbers), dtype=torch.float64, device=device)
        for i, model in enumerate(self.unmixModel.endmemberModels.values()):
            for parameter in model.parameters():
                parameter.requires_grad_(False)
                perturb = np.random.uniform(-.0001, 0.0001)
                parameter.add_(perturb)
            trueSpectra += model.forward()*trueAbundances[i]

        self.unmixModel = AnalysisBySynthesis(paramFile=self.modelFilename, wavenumbers=self.wavenumbers,
                                      dtype=torch.float64, device=device)
        trueSpectra = to_numpy(trueSpectra)
        trueSpectra = torch.from_numpy(trueSpectra).type(torch.float64).to(device)
        self.unmixModel.fit(trueSpectra, epochs=100, learningRate=1e-6)
        self.assertTrue(torch.allclose(trueSpectra, self.unmixModel.predictedSpectra, rtol=5e-2))
        self.assertTrue(torch.allclose(trueAbundances, self.unmixModel.abundances, rtol=1e-1))

        plt.plot(self.wavenumbers.cpu().detach().numpy(),
                 trueSpectra.cpu().detach().numpy(),
                 label='truth')
        plt.plot(self.wavenumbers.cpu().detach().numpy(),
                 self.unmixModel.predictedSpectra.cpu().detach().numpy(),
                 '--',
                 label='model')
        plt.title('Test Fit')
        plt.legend()
        plt.show()


