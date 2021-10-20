#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/10/17

@author: Maurizio Ferrari Dacrema
"""
from typing import Optional

from recsys_framework.Recommenders.Recommender_utils import check_matrix
from recsys_framework.Recommenders.BaseCBFRecommender import BaseItemCBFRecommender
from recsys_framework.Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from recsys_framework.Recommenders.IR_feature_weighting import okapi_BM_25, TF_IDF
import numpy as np
import scipy.sparse as sp

from recsys_framework.Recommenders.Similarity.Compute_Similarity import Compute_Similarity, SimilarityFunction, FeatureWeightingFunction, \
    EuclideanSimilarityFromDistanceMode, SET_SIMILARITIES, NON_NORMALIZABLE_SIMILARITIES
from recsys_framework.Utils.conf_logging import get_logger

logger = get_logger(
    logger_name=__file__,
)


class ItemKNNCBFRecommender(BaseItemCBFRecommender, BaseItemSimilarityMatrixRecommender):  # type: ignore
    """ ItemKNN recommender"""

    RECOMMENDER_NAME = "ItemKNNCBFRecommender"

    FEATURE_WEIGHTING_VALUES = ["BM25", "TF-IDF", "none"]

    def __init__(self, URM_train, ICM_train, verbose=True):
        super(ItemKNNCBFRecommender, self).__init__(URM_train, ICM_train, verbose=verbose)

    def fit(self, topK=50, shrink=100, similarity='cosine', normalize=True, feature_weighting="none",
            **similarity_args):

        self.topK = topK
        self.shrink = shrink

        if feature_weighting not in self.FEATURE_WEIGHTING_VALUES:
            raise ValueError(
                "Value for 'feature_weighting' not recognized. Acceptable values are {}, provided was '{}'".format(
                    self.FEATURE_WEIGHTING_VALUES, feature_weighting))

        if feature_weighting == "BM25":
            self.ICM_train = self.ICM_train.astype(np.float32)
            self.ICM_train = okapi_BM_25(self.ICM_train)

        elif feature_weighting == "TF-IDF":
            self.ICM_train = self.ICM_train.astype(np.float32)
            self.ICM_train = TF_IDF(self.ICM_train)

        similarity = Compute_Similarity(self.ICM_train.T, shrink=shrink, topK=topK, normalize=normalize,
                                        similarity=similarity, **similarity_args)

        self.W_sparse = similarity.compute_similarity()
        self.W_sparse = check_matrix(self.W_sparse, format='csr')


class ItemKNNCBFRecommenderAllSimilarities(ItemKNNCBFRecommender):

    RECOMMENDER_NAME = "ItemKNNCBFRecommenderAllSimilarities"

    def __init__(
        self,
        urm_train: sp.csr_matrix,
        icm_train: sp.csr_matrix,
    ):
        super().__init__(
            URM_train=urm_train,
            ICM_train=icm_train,
            verbose=True,
        )

        self.W_sparse: Optional[sp.csr_matrix] = None

    def fit(
        self,
        top_k: int,
        shrink: int,
        similarity: SimilarityFunction,
        normalize: bool,
        feature_weighting: FeatureWeightingFunction,
        asymmetric_alpha: float,
        tversky_alpha: float,
        tversky_beta: float,
        euclidean_normalize_avg_row: bool,
        euclidean_similarity_from_distance_mode: EuclideanSimilarityFromDistanceMode,
        **kwargs
    ) -> None:
        if (similarity in NON_NORMALIZABLE_SIMILARITIES
                and normalize):
            logger.warning(
                f"Changed the value of 'normalize' from {True} to {False}. "
                f"Because received an invalid value for similarity={similarity} and normalize={normalize}. "
                f"This similarity must have 'normalize' as false given that it is a NON_NORMALIZABLE_SIMILARITIES"
                f"={NON_NORMALIZABLE_SIMILARITIES}"
            )
            normalize = False

        if (similarity in SET_SIMILARITIES
                and feature_weighting != FeatureWeightingFunction.NONE):
            logger.warning(
                f"Changed the value of 'feature_weighting' from {feature_weighting} to {FeatureWeightingFunction.NONE}. "
                f"Because received an invalid value for similarity={similarity} and feature_weighting"
                f"={feature_weighting}. "
                f"This similarity must have 'feature_weighting' as {FeatureWeightingFunction.NONE} given that it is "
                f"a SET_SIMILARITIES={SET_SIMILARITIES}"
            )
            feature_weighting = FeatureWeightingFunction.NONE

        if similarity == SimilarityFunction.EUCLIDEAN:
            super().fit(
                topK=top_k,
                shrink=shrink,
                normalize=normalize,
                similarity=similarity.value,
                feature_weighting=feature_weighting.value,
                asymmetric_alpha=asymmetric_alpha,
                tversky_alpha=tversky_alpha,
                tversky_beta=tversky_beta,
                normalize_avg_row=euclidean_normalize_avg_row,
                similarity_from_distance_mode=euclidean_similarity_from_distance_mode.value,
            )
        else:
            super().fit(
                topK=top_k,
                shrink=shrink,
                normalize=normalize,
                similarity=similarity.value,
                feature_weighting=feature_weighting.value,
                asymmetric_alpha=asymmetric_alpha,
                tversky_alpha=tversky_alpha,
                tversky_beta=tversky_beta,
            )
