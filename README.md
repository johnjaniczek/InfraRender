# Project Description

This repository corresponds to the work in our paper written by the following authors

Title: Differentiable Programming for Hyperspectral Unmixing Using a Physics-based Dispersion Model

Paper Authors: John Janiczek, Suren Jayasuriya, Gautam Dasarathy, Christopher Edwards, Phil Christensen

Software written by: John Janiczek
________________________________________________________________________________________________________________________
# Abstract
Hyperspectral unmixing is an important remote sensing task with applications including material identification and
analysis. Characteristic spectral features make many pure materials identifiable from their visible-to-infrared 
spectra, but quantifying their presence within a mixture is a challenging task due to nonlinearities and factors of 
variation. In this paper, spectral variation is considered from a physics-based approach and incorporated into an 
end-to-end spectral unmixing algorithm via differentiable programming. The dispersion model is introduced to simulate 
realistic spectral variation, and an efficient method to fit the parameters is presented. Then, this dispersion model 
is utilized as a generative model within an analysis-by-synthesis spectral unmixing algorithm. Further, a technique 
for inverse rendering using a convolutional neural network to predict parameters of the generative model is introduced 
to enhance performance and speed when training data is available. Results achieve state-of-the-art on both infrared 
and visible-to-near-infrared (VNIR) datasets as compared to baselines, and show promise for the synergy between 
physics-based models and deep learning in hyperspectral unmixing in the future.
________________________________________________________________________________________________________________________
# Acknowledgements
This material is based upon work supported by the National Science Foundation under NSF IIS-1909192.
The authors acknowledge Research Computing at Arizona State University for providing GPU resources 
that have contributed to the research results reported within this paper. We would also like to acknowledge 
Alina Zare, Christopher Haberle, Deanna Rogers for their helpful discussions which contributed to this paper.
We would also like to thank Kim Murray (formerly Kim Feely) for providing the laboratory measurements and analysis
contributing to this paper
________________________________________________________________________________________________________________________

# Installation Instructions:
1) Download github repository
2) Install dependencies (virutal environment reccomended)
-torch (version 1.0 used in development)
-numpy
-scipy
-pandas
-h5py
-matplotlib
3) Run example code to test files
________________________________________________________________________________________________________________________
# Example Code
1) Run demo script for analysis by synthesis (optimization based) spectral unmixing with dispersion model in the loop
- Average error for the Feely dataset will be print in output
- Results will be stored in results/<todays date>/<test name>
- Results include performance metrics in a csv file and predictions in a .hdf file
  
2) Run demo script for InfraRender (neural network based) spectral unmixing with dispersion model in the loop
- Average error for the Feely dataset will be print in output
- Results will be stored in results/<todays date>/<test name>
- Results include performance metrics in a csv file and predictions in a .hdf file

3) Run demo script for dispersion model parameter estimation
- Reconstruction error between modelled and observed spectra will print on screen as the optimization routine finds
the dispersion parameters
- Dispersion params will be saved in .hdf format in input/dispersionModelParameters/feely_params_demo0.hdf (file increments)

________________________________________________________________________________________________________________________

# InfraRender Package

The __init__.py file allows the following modules to be imported from the InfraRender package to generalize code to new experiments

1) AnalysisBySynthesis(self, paramFile=None, wavenumbers=None, p=0.99, lam=0.001, dtype=torch.float64, device='cpu')
- paramFile: path to dispersion model parameters
- wavenumbers: torch vector of wavenumbers used in spectral analysis
- p: (see lp norm spectral unmixing papers for enforcing sparsity)
- lam: weight of lp norm penalty
- dtype: torch datatype expected
- device: device expected by pytorch (GPU reccomended if available)
- Description: object for unmixing using the analysis by synthesis technique
- Usage from example code:
```python
# initialize module
model = AnalysisBySynthesis(paramFile=model_file, p=p, lam=p_lambda,
                            wavenumbers=wavenumbers, dtype=dtype, device=device)
# set constraints for allowed dispersion model variation
for name, endmember in model.endmemberModels.items():
        endmember.set_constraint_tolerance(freq_tolerance=freq_tol,
                                           gamma_tolerance=gamma_tol,
                                           epsilon_tolerance=eps_tol,
                                           rho_tolerance=rho_tol,
                                           mode_weight_tolerance=mode_weight_tol)
# fit abundances and predictions
model.fit(trueSpectra, epochs=epochs, learningRate=lr, betas=betas) 

# extract abundances
abundances = model.abundances
 ```

2) Inverse Render(wavenumbers=None, param_file=None, dtype=torch.float64, device='cuda')
- Description: object to perform spectral unmixing using Inverse Rendering (neural network based)
- paramFile: path to dispersion model parameters
- wavenumbers: torch vector of wavenumbers used in spectral analysis
- dtype: torch datatype expected
- device: device expected by pytorch (GPU reccomended if available)
- Usage from example code:
```python
# function to load inverse render model from pretrained model_params
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

# initialize model
model = load_model(model_params=param_path, dispersion_params=dispersion_params, wavenumbers=wavenumbers)

# forward pass through model to predict abundances and reconstructed spectra (with dispersion model)
pred_spectra, pred_abundances = model.forward(cur_spectra.unsqueeze(0)
```

3) DispersionModelEstimator(wavenumbers, trueSpectra, learningRate=1e-4, device='cuda',
                 dtype=torch.float64, numModes=2, numFreqs=15, betas=(0.9, 0.999), rho_sparsity=0, out='emissivity')
- Object to estimate dispersion model parameters given an input true spectra
- wavenumbers: wavenumbers in spectral analysis
- trueSpectra: observed spectra
- learningRate: learning rate of optimization
- device: torch device
- dtype: torch datatype
- numModes: number of symetric axes in dispersion model (see quartz dispersion model)
- numFreqs: number of frequencies in each mode (unnecesary frequencies will be pruned by algorithm)
- betas: betas in Adam optimizaton routine
- rho_sparsity: sparsity applied to dispersion model rho parameter (strength of band) to encourage sparse predictions
- out: 'emissivity' or 'reflectivity'

```python
# initialize estimator (randomly)
model = DispersionModelEstimator(truncWavenumbers, truncSpectra, learningRate=learning_rate,
                                                     numModes=num_modes, numFreqs=num_freqs, betas=betas, device='cuda',
                                                     rho_sparsity=1e-9)
# fit model to observed spectra
model.fit(epochs=20000, epsilon=1e-15)

# write dispersion parameters to hdf file
model.dispersionModel.write_hdf(filename, modelName=name)
          min_rho=1e-6, min_gamma=1e-5)
```


