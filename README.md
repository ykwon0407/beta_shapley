# Beta Shapley: a Unified and Noise-reduced Data Valuation Framework for Machine Learning

This repository is for AISTATS 2022 anonymous review. We provide key python codes used in the submission 'Beta Shapley: a Unified and Noise-reduced Data Valuation Framework for Machine Learning'. 

### Files

`src/run_check_condition.py`: runs the experiments for Figure 1.

`src/run_experiment.py`: runs the experiments in Figure 2 and Section 5.

`src/ShapEngine.py`: defines the main function for Beta-Shapley value.

`src/config.py`: handles experiment settings. One configuration is defined for each dataset and model.

`src/data.py`: handles loading and preprocessing datasets.

### Quick start

`python3 {path}/src/launcher.py run -e 001CL --run-id 0` will run an experiment with the Gaussian dataset using a logistic model

`python3 {path}/src/launcher.py run -e 003CL --run-id 0` will run an experiment with the Gaussian dataset using a support vector machine model

`python3 {path}/src/run_check_condition.py --problem classification --rid 0` will run an experiment in Figure 1.