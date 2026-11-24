# On the Limits of Synthetic Compositionality Benchmarks for Morphological Generalization


## Environment

- The environment was created using uv (pyproject.toml + uv.lock)
- There are also files available for creating a mamba environment with the exact same packages (mamba_environment.yaml + requirements.txt)
- Flash attention was installed separately after creating the mamba environment

## Data

- The code for processing data and creating the various data splits is in data/
- Examples for data processing in Estonian and Swedish are in data/Process_data.ipynb
- For processing Estonian data you'll need a key for the [Ekilex API](https://github.com/keeleinstituut/ekilex/wiki/Ekilex-API)
- Examples for splitting data into train and test sets for the different compositionality tasks are in data/Create_data_splits.ipynb
- The data splits are available on [Hugging Face](https://huggingface.co/datasets/liu-nlp/morphological_compositional_generalization)

## Model training

- The main model training script is main_final.py
- Code for distributed training is in hpo_qwen/
    - The files trial.py and final.py contain code for hyperparameter search and the final training run respectively
    - The driver_*.py files are the ones that queue the hyperparameter search trials and once those are done the final training run


## Evaluation

- The main evaluation code is in evaluate.py
- Evaluation results are available in evaluation_results/
- A random sample of 50 errors per model and random test split is available in error_samples/

