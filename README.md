# OCL Survey Code

![Screenshot from 2023-10-12 18-33-31](https://github.com/AlbinSou/ocl_survey/assets/36189710/554aec40-a211-4218-afe1-9937a751eddb)

Code for the paper **A Comprehensive Empirical Evaluation on Online Continual Learning**, *Albin Soutif--Cormerais, Antonio Carta, Andrea Cossu, Julio Hurtado, Hamed Hemati, Vincenzo Lomonaco, Joost van de Weijer*, ICCV Workshop 2023 [arxiv](https://arxiv.org/abs/2308.10328)

This repository is meant to serve as an extensible codebase to perform experiments on the Online Continual Learning setting. It is based on the [avalanche](https://github.com/ContinualAI/avalanche) library. Feel free to use it for your own experiments. You can also contribute and add your own method and benchmarks to the comparison by doing a pull request !

# Installation

Clone this repository

```
git clone https://github.com/AlbinSou/ocl_survey.git
```

Create a new environment with python 3.11

```
conda create -n ocl_survey python=3.11
conda activate ocl_survey
```

Install specific ocl_survey repo dependencies

```
pip install -r requirements.txt
```

Set your PYTHONPATH as the root of the project

```
conda env config vars set PYTHONPATH=/home/.../ocl_survey
```

In order to let the scripts know where to fetch and log data, you should also create a **deploy config**, indicating where the results should be stored and the datasets fetched. Either add a new one or **change the content of config/deploy/default.yaml**

Lastly, test the environment by launching main.py

```
cd experiments/
python main.py strategy=er experiment=split_cifar100
```

By default, experiment strategies use `device: auto`. This selects CUDA when a CUDA-capable PyTorch device is available and falls back to CPU otherwise. This makes the same experiment commands suitable for a local CPU environment and GPU-backed environments such as Google Colab. An explicit device can still be supplied when needed, for example `strategy.device=cpu` or `strategy.device=cuda`.

# Structure

The code is structured as follows:

```
├── avalanche.git # Avalanche-Lib code
├── config # Hydra config files
│   ├── benchmark
│   ├── best_configs # Best configs found by main_hp_tuning.py are stored here
│   ├── deploy # Contains machine specific results and data path
│   ├── evaluation # Manage evaluation frequency and parrallelism
│   ├── experiment # Manage general experiment settings
│   ├── model
│   ├── optimizer
│   ├── scheduler
│   └── strategy
├── experiments
│   ├── main_hp_tuning.py # Main script used for hyperparameter optimization
│   ├── main.py # Main script used to launch single experiments
│   └── spaces.py
├── notebooks
├── results # Exemple results structure containing results for ER
├── scripts
    └── get_results.py # Easily collect results from multiple seeds
├── src
│   ├── factories # Contains the Benchmark, Method, and Model creation
│   ├── strategies # Contains code for additional strategies or plugins
│   └── toolkit
└── tests
```

# Experiments launching

To launch an experiment, start from the default config file and change the part that needs to change

```
python main.py strategy=er_ace experiment=split_cifar100 evaluation=parallel
```

It's also possible to override more fine-grained arguments

```
python main.py strategy=er_ace experiment=split_cifar100 evaluation=parallel strategy.alpha=0.7 optimizer.lr=0.05
```

Finally, to use the parameters found by the hyperparameter search, use

```
python main.py strategy=er_ace experiment=split_cifar100 +best_configs=split_cifar100/er_ace
```

Before running the script, you can display the full config with "-c job" option

```
python main.py strategy=er_ace experiment=split_cifar100 evaluation=parallel -c job
```

Results will be saved in the directory specified in results.yaml. Under the following structure:

```
<results_dir>/<strategy_name>_<benchmark_name>/<seed>/
```

## Notebook / Google Colab workflow

The Skill Memory comparison notebook is designed to consume the same result format as the existing OCL Survey strategies. For a fresh checkout or Colab runtime, run the experiments first and then open `notebooks/skill_memory_strategy_comparison.ipynb` to compare the generated results.

The experiment entry point automatically selects CPU or CUDA when `strategy.device=auto`, so no notebook-specific device configuration is required. In Google Colab, enable a GPU runtime to use CUDA; on a machine without CUDA, the same command falls back to CPU.

For example:

```
cd experiments/
python main.py strategy=skill_memory experiment=split_cifar100
```

This changes only the execution device. The benchmark, model, optimizer, memory size, seed, training schedule, and result/evaluation pipeline remain controlled by the same Hydra configuration, so CPU and GPU runs follow the same experiment protocol.

# Hyperparameter selection

Modify the strategy specific search parameters, search range etc ... inside main_hp_tuning.py then run

```
python main_hp_tuning.py strategy=er_ace experiment=split_cifar100
```

# Citation

If you use this repo for a research project please use the following citation:

```
@inproceedings{soutif2023comprehensive,
  title={A comprehensive empirical evaluation on online continual learning},
  author={Soutif-Cormerais, Albin and Carta, Antonio and Cossu, Andrea and Hurtado, Julio and Lomonaco, Vincenzo and Van de Weijer, Joost and Hemati, Hamed},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={3518--3528},
  year={2023}
}
```