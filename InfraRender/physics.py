import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from utility import create_wavenumbers

def get_optical_constants(freq, four_pr, gamma, waveNum, epsilon):
    waveNum = waveNum.unsqueeze(-1)

    a = freq**2 - waveNum**2
    denom = a**2 + gamma**2 * (freq**2 * waveNum**2)
    alpha = torch.sum((four_pr * freq**2 * a) / denom, dim=-1)
    phi = torch.sum(((four_pr/2) * freq**2 * gamma * freq * waveNum) / denom, dim=-1)
    theta = epsilon + alpha
    a2 = torch.sqrt(theta**2 + 4 * (phi**2))
    n = torch.sqrt((theta + a2) / 2)
    k = phi / n

    return n, k


def get_reflectivity(n, k):
    return ((n-1)**2 + k**2) / ((n+1)**2 + k**2)


def get_emissivity(reflectivity):
    return 1 - reflectivity


def emissivity_model(freq, four_pr, gamma, wavenumbers, epsilon):
    n, k = get_optical_constants(freq, four_pr, gamma, wavenumbers, epsilon)
    reflectivity = get_reflectivity(n, k)
    emissivity = get_emissivity(reflectivity)

    return emissivity

def reflectivity_model(freq, four_pr, gamma, wavenumbers, epsilon):
    n, k = get_optical_constants(freq, four_pr, gamma, wavenumbers, epsilon)
    reflectivity = get_reflectivity(n, k)

    return reflectivity


def main():

    plt.rc('font', family='serif', serif='Times')
    plt.rc('text', usetex=True)
    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)
    plt.rc('axes', labelsize=8)

    # width as measured in inkscape
    width = 3.487
    height = width / 1.618
    fig, ax1 = plt.subplots()
    fig.subplots_adjust(left=.12, bottom=.22, right=.99, top=.97)

    epsilon = torch.tensor([2.356], dtype=torch.float64)
    freq = torch.tensor([1161.], dtype=torch.float64)
    gamma = torch.tensor([0.1], dtype=torch.float64)
    four_pr = torch.tensor([0.67], dtype=torch.float64)
    number_of_waves = 2000
    waveNum = create_wavenumbers(number_of_waves)
    emissivity = emissivity_model(freq, four_pr, gamma, waveNum, epsilon)

    freq = freq + 100
    emissivity2 = emissivity_model(freq, four_pr, gamma, waveNum, epsilon)

    freq = freq - 100
    gamma = gamma*1.2
    emissivity3 = emissivity_model(freq, four_pr, gamma, waveNum, epsilon)

    gamma = gamma / 1.1
    four_pr = four_pr*1.2
    emissivity4 = emissivity_model(freq, four_pr, gamma, waveNum, epsilon)

    plt.plot(waveNum.data.numpy(), emissivity.data.numpy(), label='initial')
    plt.plot(waveNum.data.numpy(), emissivity2.data.numpy(), label=r'$\omega_0 + 100$')
    plt.plot(waveNum.data.numpy(), emissivity3.data.numpy(), label=r'$\gamma * 120\%$')
    plt.plot(waveNum.data.numpy(), emissivity4.data.numpy(), label=r'$\rho * 120\%$')
    plt.ylabel('Emissivity')
    plt.xlabel('Wavenumber')
    plt.legend()

    plt.grid(False)
    plt.legend(prop={'size': 8}, frameon=False)
    fig.set_size_inches(width, height)
    fig.savefig('figures/output/parameters.png', dpi=600)

    plt.show()


if __name__ == '__main__':
    main()





