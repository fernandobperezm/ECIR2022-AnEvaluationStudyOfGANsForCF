[![arXiv](https://img.shields.io/badge/arXiv-2201.01815-b31b1b.svg)](https://arxiv.org/abs/2201.01815)
[![Repo DOI](https://zenodo.org/badge/419178547.svg)](https://zenodo.org/badge/latestdoi/419178547)

# An Evaluation Study of Generative Adversarial Networks for Collaborative Filtering.
This repository contains the source code of the following article: 
- An Evaluation Study of Generative Adversarial Networks for Collaborative Filtering. 
  Fernando Benjamín Pérez Maurera, Maurizio Ferrari Dacrema, and Paolo Cremonesi. ECIR 2022. DOI: [10.1007/978-3-030-99736-6_45](https://doi.org/10.1007/978-3-030-99736-6_45)
  - [Paper Pre-Print & Supplemental Material (arXiv)](https://arxiv.org/abs/2201.01815), 
  - [Paper Published Version (Springer Link)](https://link.springer.com/chapter/10.1007/978-3-030-99736-6_45), 
  - [GitHub](https://github.com/recsyspolimi/ecir-2022-an-evaluation-of-GAN-for-CF), 
  - [Zenodo](https://zenodo.org/badge/latestdoi/419178547)

If you use our work, please cite our work. You can click on the `Cite this repository` button or copy the following BibTeX snippet:
```bibtex
@inproceedings{conf/ecir/PerezMaureraFDC22/an-evaluation-study-of-generative-adversarial-networks-for-collaborative-filtering,
  author    = {Fernando Benjam{\'{i}}n {P{\'{e}}rez Maurera} and
               Maurizio {Ferrari Dacrema} and
               Paolo Cremonesi},
  title     = {An Evaluation Study of Generative Adversarial Networks for Collaborative Filtering},
  booktitle = {Advances in Information Retrieval - 44th European Conference on {IR} Research, {ECIR} 2022, Stavanger, Norway, April 10-14, 2022, Proceedings, Part {I}},
  series    = {Lecture Notes in Computer Science},
  volume    = {13185},
  pages     = {671--685},
  publisher = {Springer},
  year      = {2022},
  url       = {https://doi.org/10.1007/978-3-030-99736-6\_45},
  doi       = {10.1007/978-3-030-99736-6\_45}
}
```

See our [website](http://recsys.deib.polimi.it/) for more information on our research group. We are actively pursuing
this research direction in evaluation and reproducibility, and we are open to collaboration with other researchers. Follow
our project on [ResearchGate](https://www.researchgate.net/project/Recommender-systems-reproducibility-and-evaluation)

This repo is divided into three folders:
- [evaluation-cfgan](evaluation-cfgan/README.md): Contains the implementation of CFGAN and the experiments presented in 
  the article.
- [recsys-framework](recsys-framework/README.md): Contains the implementation of baselines, hyper-parameter 
  tuning, and evaluation used in the article.
- [pdf](pdf): Contains a copy of the [submitted article](pdf/article.pdf) and 
  [additional material](pdf/additional-material.pdf)

You'll find instructions to install this project and run the experiments in the  
[README inside evaluation-cfgan](evaluation-cfgan/README.md), in fact, all commands must be run inside 
the `evaluation-cfgan` folder.
