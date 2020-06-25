import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os, sys
import datetime
now = datetime.datetime.now()


def compareInterval(epoch, interval):
    if epoch % interval == interval-1:
        return True
    else:
        return False

def create_wavenumbers(num_waves):
    wavenumbers = np.arange(1, num_waves, dtype='float64')
    return torch.from_numpy(wavenumbers)


def clamp_with_tensor(tensor, min=None, max=None):
    if max is not None:
        tensor.data -= torch.max(torch.zeros_like(tensor), tensor.data - max)
    if min is not None:
        tensor.data += torch.max(torch.zeros_like(tensor), min - tensor.data)

def to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def get_indices(condition):
    return condition.nonzero().squeeze(1).tolist()


def show_parameters(model, modelName='model'):
    print(modelName, 'parameters')
    for i, mode in enumerate(model.modes):
        print('mode:', '\t', i, '\t', 'mode weight', mode.mode_weight.item(), '\t', 'epsilon:', '\t', mode.epsilon.item())
        for j in range(len(mode.freqs)):
            print('mode:', '\t', i, '\t', 'freq' + str(j), '\t', mode.freqs[j].item(), '\t', 'gamma' + str(j), '\t',
                  mode.gammas[j].item(), '\t', 'rho' + str(j), '\t', mode.rhos[j].item())
    print('\n')




