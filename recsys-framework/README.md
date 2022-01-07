# Recommender Systems Evaluation Framework

This repository is a modified clone of 
[this evaluation framework](https://github.com/MaurizioFD/RecSys2019_DeepLearning_Evaluation). This repository 
*does not* contain any experiment, instead, it is used as a RecSys Framework (*as a Python package*) by other 
repositories/projects. This repository *contains* the implementation of baselines and the source code related to run 
hyper-parameter tuning of baselines, evaluation, data fetching, utilities, plotting, and more.

A small example on how to use this repo is in `run_example_usage.py`.

We are actively pursuing this research direction in evaluation and reproducibility, we are open to collaboration 
with other researchers. See our
[website](http://recsys.deib.polimi.it/) for more information on our research group and also follow our project on 
[ResearchGate](https://www.researchgate.net/project/Recommender-systems-reproducibility-and-evaluation).

Please cite our articles if you use this repository or our implementations of baseline algorithms.

## Installation
Given that this repo is considered a package (exported as `recsys_framework`), you probably won't need to install it
by hand. Instead, the project that depends on this package will indicate it as a dependency and `Poetry` will build and 
make available this package to the project.

If you wish to use this framework in future developments, we *strongly* suggest that you use this source code as a
package and not write your experiments in this repo (an example of how this repo is used as a package can be seen 
[here](https://github.com/fernandobperezm/an-evaluation-of-GAN-for-CF))

Furthermore, if you really need to build this project, then you'll need to follow your OS-specific instructions (Note
that we do not support Windows installations at the moment):
- [Linux](#linux-installation)
- [macOS](#macos-installation). 

We do our best to ensure that the installation is successful in most mainstream OS. If the installation fails or the 
execution crashes, please do not hesitate and raise an issue in the repo.

### Linux Installation

- Download repo:
  ```bash
  git clone https://github.com/fernandobperezm/an-evaluation-of-GAN-for-CF.git
  cd an-evaluation-of-GAN-for-CF/recsys-framework/
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

## Code organization
This repository is organized in several subfolders.

### Baseline algorithms
Folders inside `recsys_framework.Recommenders`, such as `KNN`, `GraphBased`, `MatrixFactorization`, `SLIM`, and 
`EASE_R` contain all the baseline algorithms we used in our experiments. The complete list is as follows, details on 
all algorithms and references can be found [HERE](DL_Evaluation_TOIS_Additional_material.pdf):
- Random: recommends a list of random items,
- TopPop: recommends the most popular items, 
- UserKNN: User-based collaborative KNN, 
- ItemKNN: Item-based collaborative KNN, 
- UserKNN CBF: User-based content-based KNN, 
- ItemKNN CBF: Item-based content-based KNN, 
- UserKNN CFCBF: User-based hybrid content-based collaborative KNN, 
- ItemKNN CFCBF: Item-based hybrid content-based collaborative KNN, 
- P3alpha: collaborative graph-based algorithm, 
- RP3beta: collaborative graph-based algorithm with reranking, 
- PureSVD: SVD decomposition of the user-item matrix, 
- NMF: Non-negative matrix factorization of the user-item matrix, 
- IALS: Implicit alternating least squares, 
- MatrixFactorization BPR (BPRMF): machine learning based matrix factorization optimizing ranking with BPR, 
- MatrixFactorization FunkSVD: machine learning based matrix factorization optimizing prediction accuracy with MSE, 
- EASE_R: collaborative shallow autoencoder, 
- SLIM BPR: Item-based machine learning algorithm optimizing ranking with BPR, 
- SLIM ElasticNet: Item-based machine learning algorithm optimizing prediction accuracy with MSE.

The following similarities are available for all KNN models: `cosine`, `adjusted cosine`, `pearson correlation`, `dice`,
`jaccard`, `asymmetric cosine`, `tversky`, `euclidean`.

### Evaluation
The script `recsys_framework.Evaluation.Evaluator.py` contains the two evaluator objects (`EvaluatorHoldout`, 
`EvaluatorNegativeSample`) which compute all the metrics we report.

### Data
The folder `recsys_framework.Data_manager` contains a number of `DataReader` objects each associated to a specific 
dataset, which are used to read datasets. Whenever a new dataset is downloaded and parsed, the preprocessed data is 
saved in a new folder called `Data_manager_split_datasets`, which contains a sub-folder for each dataset. The data split 
used for the experimental evaluation is saved within the result folder for the relevant algorithm, in a sub-folder 
`data`. 

### Hyper-parameter optimization
The folder `recsys_framework.HyperparameterTuning` contains all the code required to tune the hyper-parameters of the 
baselines. The object `recsys_framework.HyperparameterTuning.SearchBayesianSkopt.py::SearchBayesianSkopt` does the 
hyper-parameter optimization for a given recommender instance and hyper-parameter space, saving the explored 
configuration and corresponding recommendation quality. 
