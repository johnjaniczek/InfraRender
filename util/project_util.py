import torch
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os, sys
import datetime
now = datetime.datetime.now()


def create_pytorch_compatible_name(modelName):
    return modelName.split('.')[0]


def create_plot_filename(modelName):
    return 'results/plots/modelFit/' + modelName.split(' ')[0]


def create_title(modelName):
    return modelName + ' dispersion model'

def cropFeelyEndmembers(endmembs):
    # crop spectra at 400 wavenumbers
    endmembs.spectra = endmembs.spectra[104:, :]
    endmembs.bands = endmembs.bands[104:]


def save_list_of_dicts(metrics, name, result_path, today, i):
    """
    :param metrics: list[dict1, dict2, ...]
    :param method: string, name of method
    :return:
    """
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(result_path + today + "_%s%s.csv" % (name, i))


def mask_TES73(spectra):
    spectra = np.ma.array(spectra)
    spectra[26] = np.ma.masked
    return spectra


def plot_model_spectra(wavenumbers, modelSpectra, trueSpectra=None, title='Model', filename=None):

    if trueSpectra is not None:
        plt.plot(wavenumbers, trueSpectra, label='truth', )
    plt.plot(wavenumbers, modelSpectra, '--', label='model')
    plt.title(title)
    plt.legend()

    if filename is not None:
        plt.savefig(filename)

    plt.show()


def create_result_directory():
    # create directory for today's date
    today = now.strftime("%Y-%m-%d")
    todays_results = "results/metrics/" + today
    try:
        os.mkdir(todays_results)
    except OSError:
        print("Creation of the directory %s failed" % todays_results)
    else:
        print("Successfully created the directory %s " % todays_results)
    result_path = todays_results + "/"

    return result_path, today

def create_experiment_directory(result_path, experiment_name):
    sweep_id = get_next_index(result_path + experiment_name, extension='')
    experiment_directory = result_path + experiment_name + str(sweep_id)
    try:
        os.mkdir(experiment_directory)
    except OSError:
        print("Creation of the directory %s failed" % experiment_directory)
    else:
        print("Successfully created the directory %s " % experiment_directory)
    experiment_path = experiment_directory + '/'
    return experiment_path



def get_next_index(filename, extension='.csv'):
    i = 0
    while os.path.exists(filename + str(i) + extension):
        i += 1
    return i


def truncate_nonzero(spectra, wavenumbers):
    nonzeroIdx = np.where(spectra > 0)
    truncSpectra = torch.from_numpy(spectra[nonzeroIdx])
    truncWavenumbers = wavenumbers[nonzeroIdx]

    return truncSpectra, truncWavenumbers



def get_metrics(modelAbundances=None, trueAbundances=None, thresh=1e-2,
                modelSpectra=None, trueSpectra=None, idxPosTruth=None):
    metrics = {}
    if modelSpectra is not None and trueSpectra is not None:
        metrics["RMS"] = torch.sqrt(((modelSpectra - trueSpectra) ** 2).mean()).item()
    if modelAbundances is not None and trueAbundances is not None:
        metrics["error_L1"] = torch.sum(torch.abs(modelAbundances - trueAbundances)).item()
        metrics["error_L2"] = torch.sqrt(torch.sum((modelAbundances - trueAbundances)**2)).item()
        metrics['squared_error'] = torch.sum((modelAbundances - trueAbundances)**2).item()
        modelPresence = (modelAbundances > thresh)
        truePresence = (trueAbundances > thresh)
        metrics["precision"] = sum(modelPresence[truePresence] == 1).item() / (sum(modelPresence == 1).item()+1e-32)
        metrics["recall"] = sum(modelPresence[truePresence] == 1).item() / (sum(truePresence == 1).item()+1e-32)
        metrics["accuracy"] = sum(modelPresence == truePresence).item() / (len(modelPresence)+1e-32)
    return metrics

def consolidate_feely(abundances):
    """
    consolidate endmember abundances into mineral abundances
    to compare ground truth (mineral category) with prediction (endmembers)
    """
    abundances_feely = []
    # amphiboles
    idx = [0, 4, 18]
    abundances_feely.append(abundances[idx].sum().item())
    # consolidate biotite-chlorite
    idx = [6, 10, 28]
    abundances_feely.append(abundances[idx].sum().item())
    # consolidate calcite-dolomite
    idx = [9, 13]
    abundances_feely.append(abundances[idx].sum().item())
    # epidote
    abundances_feely.append(abundances[14].item())
    # feldspar
    idx = [1, 2, 3, 20, 21, 24, 27, 31]
    abundances_feely.append(abundances[idx].sum().item())
    # garnet, obsidian, glaucophane, kyanite, muscovite
    for j in [16, 23, 17, 19, 22]:
        abundances_feely.append(abundances[j].item())
    # olivine
    idx = [25, 26]
    abundances_feely.append(abundances[idx].sum().item())
    # pyroxene
    idx = [5, 8, 11, 12, 14, 36]
    abundances_feely.append(abundances[idx].sum().item())
    # quartz
    idx = [29, 30]
    abundances_feely.append(abundances[idx].sum().item())
    # serpentine
    idx = [32, 33]
    abundances_feely.append(abundances[idx].sum().item())
    #  talc, vesuvianite, zoisite
    for j in [34, 35, 37]:
        abundances_feely.append(abundances[j].item())

    abundances_feely = torch.tensor(abundances_feely, dtype=abundances.dtype, device=abundances.device)
    abundances_feely.data /= abundances.sum()

    return abundances_feely

def to_numpy(tensor):
    return tensor.detach().cpu().numpy()

def create_wavenumbers(num_waves):
    wavenumbers = np.arange(1, num_waves, dtype='float64')
    return torch.from_numpy(wavenumbers)

