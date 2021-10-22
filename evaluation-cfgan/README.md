# Evaluation of Collaborative Filtering Generative Adversarial Networks

This repository was developed by [Fernando B. Pérez Maurera](https://github.com/fernandobperezm). Fernando is a Ph.D.
student at Politecnico di Milano. This repository makes use of 
[this evaluation framework](https://github.com/fernandobperezm/recsys-framework-evaluation) 
(installed as a Python package). 

This repository contains the source code of the following articles:
* (Under Review) An Evaluation Study of Generative Adversarial Networks for Collaborative Filtering. Submitted to ECIR 2022.

See our [website](http://recsys.deib.polimi.it/) for more information on our research group. We are actively pursuing 
this research direction in evaluation and reproducibility, we are open to collaboration with other researchers. Follow 
our project on [ResearchGate](https://www.researchgate.net/project/Recommender-systems-reproducibility-and-evaluation)

## Repository organization

### Starting point
The only starting point to execute the experiments is the script [main.py](main.py), which executes certain experiments and
prints their results based on console arguments. Please, refer to the [Installation](#installation) and then the 
[Experiments](#experiments-source-code) sections to know how to install the dependencies of the project in your OS and how to run
our experiments, respectively.

### CFGAN source code.
The `conferences` folder contains the original implementation of [CFGAN](https://doi.org/10.1145/3269206.3271743) inside the 
`conferences.cikm.cfgan.original_source_code` folder. Source code in this folder is denoted as "Original Source Code"
in our paper.

Our porting of CFGAN is inside the `conferences.cikm.cfgan.our_implementation` folder. There you'll find the following 
folders:
- `dataset_reader`: Contains the 
  [CFGANDatasetReader](conferences/cikm/cfgan/our_implementation/dataset_reader/CFGANDatasetReader.py) class to read the
  original CFGAN data splits.
- `models`: Contains the Tensorflow implementation of CFGAN. The original source code needs Tensorflow v1. We migrated
  the source code to Tensorflow v2 through the compatibility layer, i.e., it runs the same source code in Tensorflow v1
  but under a Tensorflow v2 installation. There you'll find a class for each CFGAN model mentioned in the article
  (e.g., CFGAN with random noise is in 
  [RandomNoiseCFGANModel.py](conferences/cikm/cfgan/our_implementation/models/v1_compat/RandomNoiseCFGANModel.py))
- `original`: A copy of the "Original Source Code" with slight modifications to make it suitable to run the code from 
  an outside script and run the [replicability experiments](experiments/replication.py).
- `recommenders`: Contains classes that use CFGAN as model to generate recommendations. There you'll find a class for 
  each CFGAN recommender mentioned the article. (e.g., CFGAN with random noise is in 
  [RandomNoiseCFGANRecommender.py](conferences/cikm/cfgan/our_implementation/recommenders/RandomNoiseCFGANRecommender.py))

Additionally, you'll find the following scripts:
- `constants.py`: Contains useful enums to describe CFGAN features, like `mode`, `variant`, etc.
- `parameters.py`: Contains classes describing the hyper-parameters of CFGAN and other implementations. 

### Experiments source code

The `experiments` folder contains scripts related to each of our experiments in the paper, specifically:
- [commons.py](experiments/commons.py): Contains common code used by two or more experiments.
- [replication.py](experiments/replication.py): Contains the source code to run and print the results of the replication experiments 
   (RQ1 in the paper.)
- [reproducibility.py](experiments/reproducibility.py): Contains the source code to run and print the results of the reproducibility experiments 
   (RQ2 in the paper.)
- [concerns.py](experiments/concerns.py): Contains the source code to run and print the results of the concerns experiments (RQ3 in the paper.)

## Installation

Note that this repository requires `Python 3.9`, `poetry`, `Cython`, and the 
[recsys_framework](../recsys-framework/README.md) and we tested our installation against [Linux](#linux-installation) 
and [macOS](#macos-installation). Currently, we do not support installations on Windows.

All Cython algorithms needs to be compiled for your specific environment:
- [Linux](#linux-installation)
- [macOS](#macos-installation). 

Be aware that during the compilation you may see some warnings. The installation procedures for your OS will guide you 
through all the steps needed to execute our experiments.

### Linux Installation
- Download repo:
  ```bash
  git clone https://github.com/fernandobperezm/an-evaluation-of-GAN-for-CF.git
  cd an-evaluation-of-GAN-for-CF/evaluation-cfgan/
  ```
- Install dependencies for `pyenv`, `poetry`, and the repo source code (this includes a C/C++ compiler).
  ```bash
  sudo apt-get update -y; sudo apt-get install gcc make python3-dev gifsicle build-essential libssl-dev zlib1g-dev \
  libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
  libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev -y
  ```
- `Python 3.9.7` using [pyenv](https://github.com/pyenv/pyenv#installation)
  ```bash
  curl https://pyenv.run | bash
  ```
  - Remember to include `pyenv` in your shell
    [Section 2: Configure your shell's environment for Pyenv](https://github.com/pyenv/pyenv#basic-github-checkout).
  - Reload your shell (simple: quit and open again).
- `Poetry` using `pyenv`
   ```bash
   pyenv install 3.9.7
   pyenv local 3.9.7
   curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | python3 -
   ```
    - Ensure to add `export PATH="/home/<your user>/.local/bin:$PATH"` to your bash profile (e.g., `~/.bashrc`
      , `~/.bash_profile`, etc)
- Download dependencies using `poetry`
  ```bash
  poetry install
  ``` 

### macOS Installation
- Download repo:
  ```bash
  git clone https://github.com/fernandobperezm/evaluation-cfgan.git
  cd evaluation-cfgan/
  ```
- `Command Line Tools for Xcode` from the [Apple's Developer website](https://developer.apple.com/download/more/?=xcode)
  . Required to have a `C` compiler installed in your Mac. You'll need a free Apple ID to access these resources.
  ```bash
  xcode-select --install
  ```
- `Homebrew` from [this page](https://brew.sh)
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   brew update
   brew install openssl readline sqlite3 xz zlib hdf5 c-blosc
   ```
- `Python 3.9.7`
    - Using [pyenv](https://github.com/pyenv/pyenv#installation)
      ```bash
      curl https://pyenv.run | bash
      ```
- `Poetry` using `pyenv`
    ```bash
    pyenv install 3.9.7
    pyenv local 3.9.7
    curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | python3 -
    ```
- Download dependencies using `poetry`
  ```bash
  poetry install
  ```

## Experiments
You can re-run all the experiments that by following the instructions in the [replicability](#replicability-of-cfgan),
[reproducibility](#reproducibility-of-cfgan), and [concerns](#concerns-of-cfgan) sections. 

### Parallelize execution
This repository uses [dask](https://dask.org) to parallelize the experiments using processes. We used an AWS 
[c5.9xlarge](https://aws.amazon.com/ec2/instance-types/c5/?nc1=h_ls) instance to run all the experiments. In particular, 
we used `14` different processes. By default, this repository uses `4` different processes, you can however, change this 
default by changing the `num_workers` key inside the `pyproject.toml` file. For example, if you want to use 2 processes, 
then change the key to `num_workers = 2`. If you want to disable parallelism, then set the key to `num_workers = 1`.

### Download results
If you are only interested in exporting our results, then you must download our data splits from 
[here](https://polimi365-my.sharepoint.com/:u:/g/personal/10565493_polimi_it/ESajcZZTCFhKhzPWpRP47zEBSdzRLxPl_rESEP2wcoEklg?e=AadWHv) 
and our trained models (a folder called `result_experiments`) from 
[here](https://polimi365-my.sharepoint.com/:u:/g/personal/10565493_polimi_it/Eeg2czvCPY9FtT8220wu3e8BBh4f60QZLz35MAnNJVQJxQ?e=46EH6c). 
Once downloaded, uncompress these files inside the `evaluation_cfgan` folder. Your tree must look like the following:

```
an-evaluation-of-GAN-for-CF-ecir-2022-submission/
  |
  |----> evaluation-cfgan/
  |      |
  |      |---->conferences/
  |      |---->experiments/
  |      |---->data_split/ (uncompressed data split folder)
  |      |---->result_experiments/ (uncompressed result_experiments folder)
  |      | ...
  |----> recsys_framework/
  |      | ...
  ...
```

### Replicability of CFGAN.
The main script [main.py](main.py) executes and prints the results of the _replicability_ experiments.
```bash
poetry run python main.py \
 --run_replicability \
 --print_replicability_results
```

The file [replicability.py](experiments/replication.py) contains all the source code related to the replicability 
experiments. If you wish to change the number of replicability executions, just reassign the variable 
`NUMBER_OF_EXECUTIONS`. Originally, it is set as `NUMBER_OF_EXECUTIONS = 30`.

### Reproducibility of CFGAN.

The main script [main.py](main.py) executes and prints the results of the _reproducibility_ experiments.

```bash
poetry run python main.py \
 --run_reproducibility \
 --include_baselines \
 --include_cfgan \
 --print_reproducibility_results
```

The file [reproducibility.py](experiments/reproducibility.py) contains all the source code related to the 
reproducibility experiments.

### Concerns of CFGAN.

The main script [main.py](main.py) executes and prints the results of the _concerns_ experiments.

```bash
poetry run python main.py \
 --run_concerns \
 --include_cfgan_with_class_condition  \
 --include_cfgan_with_random_noise \
 --include_cfgan_without_early_stopping \
 --print_concerns_results
```

The file [concerns.py](experiments/concerns.py) contains all the source code related to the concerns experiments.

### Further help
If you want to know what each console argument does, you can run `poetry run python main.py --help` and it will display 
all the accepted command-line arguments, default values, and a description of them. Issues for clarifications and 
improvements are always welcomed as well. 

### Contacts.
- Fernando B. PEREZ MAURERA - [email](mailto:fernandobenjamin.perez@polimi.it)
- Maurizio FERRARI DACREMA - [email](mailto:maurizio.ferrari@polimi.it)
