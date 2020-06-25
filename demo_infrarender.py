import torch
from InfraRender import InverseRender
from util import MixedSpectra
from util import get_metrics, save_list_of_dicts, create_result_directory, get_next_index, consolidate_feely

# experiment hyperparameters
rho_tol = 1e-1
gamma_tol = 1e-1
eps_tol = 1e-5
freq_tol = 1e-2
mode_weight_tol = 1e-5
dtype = torch.float64
device = 'cuda'
param_path = 'input/models/feely_InfraRender_v1-0.params'
dispersion_params = 'input/dispersionModelParameters/feely_params.hdf'

def main():
    result_path, today = create_result_directory()
    run_id = get_next_index(result_path + today + 'InfraRender_metrics')
    mixtures = MixedSpectra(ascii_spectra="input/test/mtes_kimmurray_rocks_full_tab.txt",
                            meta_csv="input/test/mtes_kimmurray_rocks_full_meta.csv")

    spectra = torch.from_numpy(mixtures.spectra[104:, :].T).type(dtype).to(device)
    abundances = torch.from_numpy(mixtures.abundances).type(dtype).to(device)
    wavenumbers = torch.from_numpy(mixtures.bands[104:]).type(dtype).to(device)

    model = load_model(model_params=param_path, dispersion_params=dispersion_params, wavenumbers=wavenumbers)
    metrics = []
    squared_error = 0
    j, count = 0, 0
    for cur_spectra, cur_abundances in zip(spectra, abundances):
        if isValidMixture(mixtures.category[j]):
            pred_spectra, pred_abundances = model.forward(cur_spectra.unsqueeze(0))
            pred_abundances = consolidate_feely(pred_abundances[0])
            metrics.append(get_metrics(modelSpectra=pred_spectra, trueSpectra=cur_spectra,
                                       modelAbundances=pred_abundances, trueAbundances=cur_abundances))
            squared_error += ((pred_abundances - cur_abundances) ** 2).sum().item()
            count += 1
        j += 1
    print('average error', squared_error/count)
    save_list_of_dicts(metrics, "InfraRender_metrics", result_path, today, run_id)



def isValidMixture(category):
    return category != 'invalid'


def load_model(model_params, dispersion_params, wavenumbers):
    model = InverseRender(wavenumbers=wavenumbers, param_file=dispersion_params, dtype=dtype, device=device)
    for name, endmember in model.mixture_model.endmemberModels.items():
        endmember.set_constraint_tolerance(freq_tolerance=freq_tol, gamma_tolerance=gamma_tol,
                                           epsilon_tolerance=eps_tol,
                                           rho_tolerance=rho_tol, mode_weight_tolerance=mode_weight_tol)
    model.type(dtype).to(device)
    model.load_state_dict(torch.load(param_path))
    model.eval()
    return model


if __name__ == '__main__':
    main()
