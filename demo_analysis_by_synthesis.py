import torch
from InfraRender import AnalysisBySynthesis
from util import MixedSpectra
from util import get_metrics, save_list_of_dicts, create_result_directory, consolidate_feely, create_experiment_directory
import pandas as pd


# experiment hyperparameters
min_endmemb = 0.05
epochs = 1
lr = 1e-2
betas = (0.9, 0.999)
dtype = torch.float64
device = 'cuda'
p = 0.95
p_lambda = 1e-4
rho_tol = 5e-2
gamma_tol = 5e-3
eps_tol = 1e-4
freq_tol = 1e-2
mode_weight_tol = 1e-4
model_file ='input/dispersionModelParameters/feely_params.hdf'

# setup results directory
result_path, today = create_result_directory()
experiment_path = create_experiment_directory(result_path, 'analysis_by_synthesis')

# load test data
mixtures = MixedSpectra(ascii_spectra="input/test/mtes_kimmurray_rocks_full_tab.txt",
                        meta_csv="input/test/mtes_kimmurray_rocks_full_meta.csv")
spectra = mixtures.spectra[104:, :].T
wavenumbers = mixtures.bands[104:]
abundances = mixtures.abundances
wavenumbers = torch.tensor(wavenumbers, dtype=dtype, device=device)

run_id = 0
# save model parameters
experiment_params = [{"min_endmemb": min_endmemb,
                      "epochs": epochs,
                      "learningRate": lr,
                      "betas": betas,
                      'p': p,
                      'p_lambda': p_lambda,
                      'rho_tol': rho_tol,
                      'gamma_tol': gamma_tol,
                      'eps_tol': eps_tol,
                      'freq_tol': freq_tol,
                      'mode_weight_tol': mode_weight_tol,
                      'model_file': model_file
                      }]

save_list_of_dicts(experiment_params, "labmix_params", experiment_path, today, run_id)
model = AnalysisBySynthesis(paramFile=model_file,
                  wavenumbers=wavenumbers, dtype=dtype, device=device)

# set constraint tolerances
for name, endmember in model.endmemberModels.items():
    endmember.set_constraint_tolerance(freq_tolerance=freq_tol, gamma_tolerance=gamma_tol, epsilon_tolerance=eps_tol,
                                       rho_tolerance=rho_tol, mode_weight_tolerance=mode_weight_tol)

metrics = []
j = 0
for trueSpectra, trueAbundances in zip(spectra, mixtures.abundances):
    if mixtures.category[j] == 'invalid':
        pass
    else:
        trueSpectra = torch.from_numpy(trueSpectra).type(dtype).to(device)
        trueAbundances = torch.from_numpy(trueAbundances).type(dtype).to(device)
        model = AnalysisBySynthesis(paramFile=model_file, p=p, lam=p_lambda,
                            wavenumbers=wavenumbers, dtype=dtype, device=device)
        for name, endmember in model.endmemberModels.items():
            endmember.set_constraint_tolerance(freq_tolerance=freq_tol,
                                               gamma_tolerance=gamma_tol,
                                               epsilon_tolerance=eps_tol,
                                               rho_tolerance=rho_tol,
                                               mode_weight_tolerance=mode_weight_tol)
        model.fit(trueSpectra, epochs=epochs, learningRate=lr, betas=betas)
        modelAbundances = consolidate_feely(model.abundances)
        metrics.append(get_metrics(modelSpectra=model.predictedSpectra, trueSpectra=trueSpectra, thresh=min_endmemb,
                                   modelAbundances=modelAbundances, trueAbundances=trueAbundances))
        model.write_results(experiment_path + today + '_labmix_results%d.hdf' % run_id, group_name=str(j))
        j += 1


# save metrics as a csv file
save_list_of_dicts(metrics, "labmix", experiment_path, today, run_id)

# convert metrics to a pandas dataframe and display average results
metrics =  pd.DataFrame(metrics)
print('squared error:')
print(metrics["squared_error"].mean())
print('\n')
