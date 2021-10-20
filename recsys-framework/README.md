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

Furthermore, if you really need to build this project, then you'll need to follow your OS-specific instructions: 
- [Windows](#windows-installation)
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

### Windows Installation

- Download repo:
  ```bash
  git clone https://github.com/fernandobperezm/evaluation-cfgan.git
  cd evaluation-cfgan/
  ```
- `Build Tools for Visual Studio 2019` from
  [this page](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2019). Required to have a
  `C` compiler installed in your PC. While inside the installer make sure to tick the `C++ Build Tools` or
  `Desktop development with C++` checkbox, then proceed with the installation. You'll need to reboot your machine after
  it finishes the installation.
- `Chocolatey` from [this page](https://chocolatey.org/install#install-step2) (using Powershell as administrator)
  ```bash
  Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
  ```
- `Python 3.9.6`
    - Using [pyenv](https://github.com/pyenv-win/pyenv-win#installation) (using Powershell as administrator)
  ```bash
  choco install pyenv-win
  ```
    - Close and reopen your Powershell windows to ensure environment variables are available.
    - NOTE: If you are running Windows 10 1905 or newer, you might need to disable the built-in Python launcher via
      Start > "Manage App Execution Aliases" and turning off the "App Installer" aliases for Python.
- `Poetry` using `pyenv` on Powershell (no need for administrator privileges)
  ```bash
  pyenv install 3.9.6
  pyenv local 3.9.6
  (Invoke-WebRequest -Uri https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py -UseBasicParsing).Content | python3 -
  ```
    - Add `Poetry` to the `PATH`
      variable [intructions here](https://www.architectryan.com/2018/03/17/add-to-the-path-on-windows-10/)
        - Poetry is installed by default on `%APPDATA%\Python\Scripts\`.
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

### Further development
If you wish to contribute, by creating different algorithms or suggesting changes, we recommend installing the following tools
```console
pip install -r requirements_lint.txt
```

This will install mypy (type checker), and black (code formatter), which are tools used in the development of this repo.

To integrate mypy with PyCharm IDE, please visit [the plugin docs](https://github.com/leinardi/mypy-pycharm) and 
make sure to set the severity of mypy to `error`.
 
To integrate black with PyCharm IDE to run after every file save, please visit 
[the docs](https://github.com/psf/black/blob/master/docs/editor_integration.md#pycharmintellij-idea) 
 
Before committing, ensure that these command do not encounter errors:
```console
$ mypy <files to commit>
$ black <files to commit>
```

We are planning to include linting tools like `flake8` or `pylama` in the near future.


Set hibernation agent to user-data [how to](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/user-data.html#user-data-console)
[instance page](https://us-east-2.console.aws.amazon.com/ec2/v2/home?region=us-east-2#EditUserData:instanceId=i-0a8416f8104312ec5)
```bash
#!/bin/bash
/usr/bin/enable-ec2-spot-hibernation
```

No aware encryption
no extra volumes attached
Ubuntu 18.04


Enable hibernation agent, 
[see](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/spot-interruptions.html#hibernate-spot-instances), 
and [see](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/spot-interruptions.html#prepare-for-instance-hibernation)
```bash
sudo /usr/bin/enable-ec2-spot-hibernation
```

C5.2XLARGE=ec2-34-243-35-250.eu-west-1.compute.amazonaws.com


Port Forwarding to see Dask dashboard locally:
```bash
ssh -i ~/.ssh/keys/fernando-polimi-aws.pem -N -L localhost:8787:localhost:8787 ubuntu@ec2-18-191-77-43.us-east-2.compute.amazonaws.com
ssh -i ~/.ssh/keys/fernando-polimi-aws.pem -N -L localhost:8888:localhost:8787 ubuntu@ec2-54-72-171-68.eu-west-1.compute.amazonaws.com
```

Connect to AWS:
```bash
ssh -i ~/.ssh/keys/fernando-polimi-aws.pem ubuntu@ec2-54-72-171-68.eu-west-1.compute.amazonaws.com
```

Original code execution
```bash
ssh -i ~/.ssh/keys/fernando-polimi-aws.pem ubuntu@ec2-54-72-171-68.eu-west-1.compute.amazonaws.com
tmux new -s original
cd /fbpm/cfgan_study
conda activate cfgan_tf_v1
python run_cfgan_original_implementation.py \
 --run_original_code \
 --NUMBER_OF_EXECUTIONS 30 \
 --run_wrapper_tf_v1 \
 &> result_experiments/20210512_code.txt
```

Hyper-parameter tuning
```bash
ssh -i ~/.ssh/keys/fernando-polimi-aws.pem ubuntu@ec2-54-72-171-68.eu-west-1.compute.amazonaws.com
tmux new -s hp
cd /fbpm/cfgan_study
conda activate cfgan
python run_cfgan_original_implementation.py \
 --run_hyper_parameter_tuning \
 --include_baselines \
 --include_cfgan \
 --include_guideline_cfgan \
 --include_cfgan_code_hyper_parameters \
 --include_cfgan_paper_hyper_parameters \
 &> result_experiments/20210506_hp_2.txt
```

Run extra experiments
```bash
ssh -i ~/.ssh/keys/fernando-polimi-aws.pem ubuntu@ec2-54-72-171-68.eu-west-1.compute.amazonaws.com
tmux new -s hp
cd /fbpm/cfgan_study
conda activate cfgan
python run_cfgan_original_implementation.py \
 --run_extra_experiments \
 --include_cfgan_cold_users_experiment \
 --include_cfgan_cold_users_random_noise_experiment \
 &> result_experiments/20210522_cold_users.txt
```

Run Paper and Code hyper-parameter tuning
```bash
python run_cikm_cfgan.py \
 --run_hyper_parameter_tuning \
 --include_cfgan_code_hyper_parameters \
 --include_cfgan_paper_hyper_parameters \
 &> result_experiments/out.txt
```

Results exporting (`--print_training_item_weights` requires 30 GB of memory for each worker.)
```bash
python run_cikm_cfgan.py \
 --print_accuracy_metrics \
 --include_baselines \
 --include_cfgan \
 --include_guideline_cfgan \
 &> result_experiments/20210506_results_latex.txt
--include_cfgan_code_hyper_parameters --include_cfgan_paper_hyper_parameters
--print_item_weights --print_training_item_weights --print_losses
```

Download console output every 30 seconds.
```bash
watch --interval 3600 -x rsync -avz -e "ssh -i ~/.ssh/keys/fernando-polimi-aws.pem" \
ubuntu@ec2-54-72-171-68.eu-west-1.compute.amazonaws.com:/fbpm/cfgan_study/result_experiments/20210506_hp_3.txt \
~/Development/polimi/polimi-recsys-collaborative-filtering-gan-evaluation/result_experiments/
```

Download results from AWS to local folder.
```bash
 "ssh -i ~/.ssh/keys/fernando-polimi-aws.pem" \
ubuntu@ec2-54-72-171-68.eu-west-1.compute.amazonaws.com:/fbpm/cfgan_study/result_experiments/latex \
~/Development/polimi/polimi-recsys-collaborative-filtering-gan-evaluation/

# ubuntu@ec2-18-188-184-162.us-east-2.compute.amazonaws.com:/fbpm/cfgan_study/result_experiments/plots \
# ubuntu@ec2-18-188-184-162.us-east-2.compute.amazonaws.com:/fbpm/cfgan_study/result_experiments/replication_original_code \
# ubuntu@ec2-54-72-171-68.eu-west-1.compute.amazonaws.com:/fbpm/cfgan_study/logs \
```

```bash
rsync -avz -e "ssh -i ~/.ssh/keys/fernando-polimi-aws.pem" \
ubuntu@ec2-54-72-171-68.eu-west-1.compute.amazonaws.com:/fbpm/cfgan_study/result_experiments/latex \
~/Development/polimi/polimi-recsys-collaborative-filtering-gan-evaluation/result_experiments/
```

Download backups
```bash
rsync -avz -e "ssh -i ~/.ssh/keys/fernando-polimi-aws.pem" \
ubuntu@ec2-54-72-171-68.eu-west-1.compute.amazonaws.com:/fbpm/cfgan_study/bk_20210412_result_experiments.zip \
~/Development/polimi/polimi-recsys-collaborative-filtering-gan-evaluation/
```

Upload local changes to remote


```bash
rsync -avz -e "ssh -i ~/.ssh/research.pem" \
--exclude={'**/bk','**/data_splits','**/data_split','**/dask-worker-space','*.c','*.so','*.html','.DS_Store','.idea','*__pycache__*','**/result_experiments','.git','**/logs','training_checkpoints','**/.mypy_cache','**/.pytest_cache','Archivio.zip','bk_20210208_result_experiments.zip'} \
~/Development/vionlabs-demo-evaluation/ ubuntu@ec2-54-74-121-216.eu-west-1.compute.amazonaws.com:/cw/vionlabs-evaluation/

rsync -avz -e "ssh -i ~/.ssh/research.pem" \
--exclude={'**/data_splits','**/data_split','**/dask-worker-space','*.c','*.so','*.html','.DS_Store','.idea','*__pycache__*','**/result_experiments','.git','**/logs','training_checkpoints','**/.mypy_cache','**/.pytest_cache','Archivio.zip','bk_20210208_result_experiments.zip'} \
~/Development/vionlabs-demo-evaluation/ ubuntu@ec2-52-211-32-254.eu-west-1.compute.amazonaws.com:/cw/vionlabs-evaluation/
```

Delete temporal/uncompleted hyper-parameter tuning results
```shell
find ./result_experiments -name ".temp_DataIO__metadata*" | xargs rm -r 
find ./result_experiments/hyper_parameter_tuning -name "*CFGANRecommenderEarlyStopping_*" | xargs rm -r
find ./result_experiments/single_run -name "*CFGANRecommender*" | xargs rm -r
find ./result_experiments/hyper_parameter_tuning -name "GuidelineCFGANRecommenderEarlyStopping_*" | xargs rm -r 
find ./result_experiments/hyper_parameter_tuning -name ".temp_DataIO__CFGAN*" | xargs rm -r 
find ./result_experiments/single_run -name ".temp_DataIO__CFGAN*" | xargs rm -r 
find ./result_experiments/hyper_parameter_tuning -name ".temp_DataIO__GuidelineCFGAN*" | xargs rm -r 
find ./result_experiments/single_run -name ".temp_DataIO__GuidelineCFGAN*" | xargs rm -r 

```


Delete plots
```bash
find ./result_experiments/plots -name "CFGANRecommenderEarlyStopping_*" | xargs rm -r
```



RUN
```bash
python run_cikm_cfgan.py \
--use_original_splits \
--run_hyper_parameter_tuning --include_baselines --include_cfgan --include_guideline_cfgan \
 &> result_experiments/20210320_hp.txt
```




