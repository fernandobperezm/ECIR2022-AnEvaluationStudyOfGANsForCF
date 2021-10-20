from typing import Type, Optional, Any

import scipy.sparse as sp

from recsys_framework.Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from recsys_framework.Recommenders.Similarity.Compute_Similarity import SimilarityFunction, FeatureWeightingFunction, \
    EuclideanSimilarityFromDistanceMode, FeatureWeightingEnumToFunction
from recsys_framework.Recommenders.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender, ItemKNNCBFRecommenderAllSimilarities
from recsys_framework.Utils.conf_logging import get_logger
from recsys_framework.Utils.decorators import log_calling_args

logger = get_logger(
    logger_name=__file__,
)


class LinearHybridSimilarityRecommender(BaseItemSimilarityMatrixRecommender):  # type: ignore

    RECOMMENDER_NAME = "LinearHybridSimilarityRecommender"

    def __init__(
        self,
        urm_train: sp.csr_matrix,
        recommender_1: ItemKNNCBFRecommender,
        recommender_2: ItemKNNCBFRecommender,
    ):
        super().__init__(
            URM_train=urm_train,
            verbose=True,
        )

        self.recommender_1 = recommender_1
        self.recommender_2 = recommender_2
        self.W_sparse: Optional[sp.csr_matrix] = None

        if self.recommender_1.W_sparse is None:
            raise ValueError(
                f"Recommender 1 ({self.recommender_1}) is not trained."
            )

        if self.recommender_2.W_sparse is None:
            raise ValueError(
                f"Recommender 2 ({self.recommender_2}) is not trained."
            )

    @log_calling_args
    def fit(
        self,
        linear_alpha: float,
    ) -> None:
        if not 0 <= linear_alpha <= 1:
            raise ValueError(
                f"Invalid value for 'linear_alpha': '{linear_alpha}'. It must be between 0 and 1."
            )

        self.W_sparse = (
            (linear_alpha * self.recommender_1.W_sparse)
            + ((1 - linear_alpha) * self.recommender_2.W_sparse)
        )


class HybridLinearTrainingItemKNNCBFAllSimilaritiesRecommender(BaseItemSimilarityMatrixRecommender):  # type: ignore
    RECOMMENDER_NAME = "HybridLinearTrainingItemKNNCBFAllSimilaritiesRecommender"

    def __init__(
        self,
        urm_train: sp.csr_matrix,
        recommender_1_class: Type[ItemKNNCBFRecommenderAllSimilarities],
        recommender_1_kwargs: dict[str, Any],
        recommender_2_class: Type[ItemKNNCBFRecommenderAllSimilarities],
        recommender_2_kwargs: dict[str, Any],
    ):
        super().__init__(
            URM_train=urm_train,
            verbose=True,
        )

        self.recommender_1 = recommender_1_class(
            **recommender_1_kwargs
        )
        self.recommender_2 = recommender_2_class(
            **recommender_2_kwargs
        )

        self.W_sparse: Optional[sp.csr_matrix] = None

    def fit(
        self,
        linear_alpha: float,

        recommender_1_top_k: int,
        recommender_1_shrink: int,
        recommender_1_similarity: SimilarityFunction,
        recommender_1_normalize: bool,
        recommender_1_feature_weighting: FeatureWeightingFunction,
        recommender_1_asymmetric_alpha: float,
        recommender_1_tversky_alpha: float,
        recommender_1_tversky_beta: float,
        recommender_1_euclidean_normalize_avg_row: bool,
        recommender_1_euclidean_similarity_from_distance_mode: EuclideanSimilarityFromDistanceMode,

        recommender_2_top_k: int,
        recommender_2_shrink: int,
        recommender_2_similarity: SimilarityFunction,
        recommender_2_normalize: bool,
        recommender_2_feature_weighting: FeatureWeightingFunction,
        recommender_2_asymmetric_alpha: float,
        recommender_2_tversky_alpha: float,
        recommender_2_tversky_beta: float,
        recommender_2_euclidean_normalize_avg_row: bool,
        recommender_2_euclidean_similarity_from_distance_mode: EuclideanSimilarityFromDistanceMode,
    ) -> None:
        self.recommender_1.fit(
            top_k=recommender_1_top_k,
            shrink=recommender_1_shrink,
            similarity=recommender_1_similarity,
            normalize=recommender_1_normalize,
            feature_weighting=recommender_1_feature_weighting,
            asymmetric_alpha=recommender_1_asymmetric_alpha,
            tversky_alpha=recommender_1_tversky_alpha,
            tversky_beta=recommender_1_tversky_beta,
            euclidean_normalize_avg_row=recommender_1_euclidean_normalize_avg_row,
            euclidean_similarity_from_distance_mode=recommender_1_euclidean_similarity_from_distance_mode,
        )

        self.recommender_2.fit(
            top_k=recommender_2_top_k,
            shrink=recommender_2_shrink,
            similarity=recommender_2_similarity,
            normalize=recommender_2_normalize,
            feature_weighting=recommender_2_feature_weighting,
            asymmetric_alpha=recommender_2_asymmetric_alpha,
            tversky_alpha=recommender_2_tversky_alpha,
            tversky_beta=recommender_2_tversky_beta,
            euclidean_normalize_avg_row=recommender_2_euclidean_normalize_avg_row,
            euclidean_similarity_from_distance_mode=recommender_2_euclidean_similarity_from_distance_mode,
        )

        logger.debug(
            f"{self.RECOMMENDER_NAME=}"
            f"\n* {self.recommender_1.RECOMMENDER_NAME=}-{self.recommender_2.RECOMMENDER_NAME=}"
            f"\n* {recommender_1_similarity=}-{recommender_2_similarity=}"
            f"\n* {self.recommender_1.W_sparse.min()=}-{self.recommender_1.W_sparse.max()=}"
            f"\n* {self.recommender_2.W_sparse.min()=}-{self.recommender_2.W_sparse.max()=}"
        )

        self.W_sparse = (
            (linear_alpha * self.recommender_1.W_sparse)
            + ((1 - linear_alpha) * self.recommender_2.W_sparse)
        )


class HybridStackSimilarityRecommender(ItemKNNCBFRecommenderAllSimilarities):  # type: ignore
    RECOMMENDER_NAME = "HybridStackSimilarityRecommender"

    def __init__(
        self,
        urm_train: sp.csr_matrix,
        icm_1: sp.csr_matrix,
        icm_2: sp.csr_matrix,
    ):
        super().__init__(
            urm_train=urm_train,
            icm_train=sp.hstack(
                [icm_1, icm_2],
                format="csr",
            ),
        )

        self.icm_1 = icm_1
        self.icm_2 = icm_2

        self.W_sparse: Optional[sp.csr_matrix] = None

    def fit(
        self,
        linear_alpha: float,
        feature_weighting_icm_1: FeatureWeightingFunction,
        feature_weighting_icm_2: FeatureWeightingFunction,
        top_k: int,
        shrink: int,
        similarity: SimilarityFunction,
        normalize: bool,

        asymmetric_alpha: float,
        tversky_alpha: float,
        tversky_beta: float,
        euclidean_normalize_avg_row: bool,
        euclidean_similarity_from_distance_mode: EuclideanSimilarityFromDistanceMode,
    ) -> None:

        feature_weighting_function_icm_1 = FeatureWeightingEnumToFunction[feature_weighting_icm_1]
        feature_weighting_function_icm_2 = FeatureWeightingEnumToFunction[feature_weighting_icm_2]

        self.ICM_train = sp.hstack(
            [
                linear_alpha * feature_weighting_function_icm_1(self.icm_1),
                (1 - linear_alpha) * feature_weighting_function_icm_2(self.icm_2),
            ],
            format="csr",
        )

        super().fit(
            top_k=top_k,
            shrink=shrink,
            normalize=normalize,
            similarity=similarity,
            feature_weighting=FeatureWeightingFunction.NONE,
            asymmetric_alpha=asymmetric_alpha,
            tversky_alpha=tversky_alpha,
            tversky_beta=tversky_beta,
            euclidean_normalize_avg_row=euclidean_normalize_avg_row,
            euclidean_similarity_from_distance_mode=euclidean_similarity_from_distance_mode,
        )
