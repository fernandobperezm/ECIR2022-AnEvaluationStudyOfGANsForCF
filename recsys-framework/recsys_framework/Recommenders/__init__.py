from __future__ import annotations

from recsys_framework.Recommenders.Hybrid.HybridContentBasedRecommender import \
    LinearHybridSimilarityRecommender, HybridStackSimilarityRecommender
from recsys_framework.Recommenders.KNN.ItemKNNCFRecommender import \
    ItemKNNCFRecommender
from recsys_framework.Recommenders.KNN.ItemKNNCBFRecommender import \
    ItemKNNCBFRecommender, ItemKNNCBFRecommenderAllSimilarities
from recsys_framework.Recommenders.KNN.ItemKNN_CFCBF_Hybrid_Recommender import ItemKNN_CFCBF_Hybrid_Recommender
from recsys_framework.Recommenders.KNN.UserKNNCBFRecommender import UserKNNCBFRecommender
from recsys_framework.Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from recsys_framework.Recommenders.KNN.UserKNN_CFCBF_Hybrid_Recommender import UserKNN_CFCBF_Hybrid_Recommender
from recsys_framework.Recommenders.NonPersonalizedRecommender import Random, TopPop, GlobalEffects
from recsys_framework.Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
from recsys_framework.Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from recsys_framework.Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender
from recsys_framework.Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from recsys_framework.Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from recsys_framework.Recommenders.MatrixFactorization.Cython.MatrixFactorization_Cython import \
    MatrixFactorization_AsySVD_Cython, MatrixFactorization_BPR_Cython, MatrixFactorization_FunkSVD_Cython
from recsys_framework.Recommenders.MatrixFactorization.PureSVDRecommender import PureSVDRecommender
from recsys_framework.Recommenders.MatrixFactorization.NMFRecommender import NMFRecommender
from recsys_framework.Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender

__all__: list[str] = [
    # Non-personalized
    "Random",
    "TopPop",
    "GlobalEffects",

    # ItemKNN
    "ItemKNNCFRecommender",
    "ItemKNNCBFRecommender",
    "ItemKNNCBFRecommenderAllSimilarities",

    # UserKNN
    "UserKNNCBFRecommender",
    "UserKNNCFRecommender",

    # Matrix Factorization
    "MatrixFactorization_AsySVD_Cython",
    "MatrixFactorization_BPR_Cython",
    "MatrixFactorization_FunkSVD_Cython",

    "NMFRecommender",
    "IALSRecommender",
    "PureSVDRecommender",

    # SLIM
    "SLIMElasticNetRecommender",
    "SLIM_BPR_Cython",

    # EASE-R
    "EASE_R_Recommender",

    # Graph-Based
    "P3alphaRecommender",
    "RP3betaRecommender",

    # Hybrid
    "ItemKNN_CFCBF_Hybrid_Recommender",
    "UserKNN_CFCBF_Hybrid_Recommender",
    "LinearHybridSimilarityRecommender",
    "HybridStackSimilarityRecommender",
]
