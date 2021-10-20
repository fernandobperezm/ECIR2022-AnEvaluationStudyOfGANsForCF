import os
import random
from pathlib import Path

import attr
import numpy as np
import pytest
import scipy.sparse as sp
import tensorflow as tf

from conferences.cikm.cfgan.our_implementation.constants import CFGANBenchmarks, CFGANMode, CFGANOptimizer, \
    CFGANActivation, CFGANMaskType
from conferences.cikm.cfgan.our_implementation.parameters import RandomNoiseCFGANHyperParameters
from conferences.cikm.cfgan.our_implementation.recommenders.RandomNoiseCFGANRecommender import \
    RandomNoiseCFGANRecommender


class TestRandomNoiseCFGANRecommender:
    TEST_NUM_ROWS = 5
    TEST_NUM_COLS = 10
    TEST_INTERACTIONS = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int32)
    TEST_USERS_INDICES = np.array([1, 1, 2, 4, 0, 1, 2, 4, 2, 3], dtype=np.int32)
    TEST_ITEMS_INDICES = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int32)
    TEST_URM_TRAIN = sp.csr_matrix(
        (TEST_INTERACTIONS,
         (TEST_USERS_INDICES, TEST_ITEMS_INDICES)),
        shape=(TEST_NUM_ROWS, TEST_NUM_COLS),
        dtype=np.int32
    )
    TEST_HYPER_PARAMETERS = RandomNoiseCFGANHyperParameters(
        benchmark=CFGANBenchmarks.CIAO,
        epochs=10,
        mode=CFGANMode.USER_BASED,
        generator_hidden_features=2,
        generator_regularization=0.0001,
        generator_learning_rate=0.00001,
        generator_batch_size=2,
        generator_hidden_layers=1,
        generator_steps=1,
        generator_optimizer=CFGANOptimizer.ADAM,
        generator_activation=CFGANActivation.SIGMOID,
        discriminator_hidden_features=2,
        discriminator_regularization=0.0002,
        discriminator_learning_rate=0.00002,
        discriminator_batch_size=3,
        discriminator_hidden_layers=1,
        discriminator_steps=1,
        discriminator_optimizer=CFGANOptimizer.ADAM,
        discriminator_activation=CFGANActivation.SIGMOID,
        mask_type=CFGANMaskType.ZERO_RECONSTRUCTION_AND_PARTIAL_MASKING,
        coefficient_zero_reconstruction=0.5,
        ratio_zero_reconstruction=40,
        ratio_partial_masking=60,
        reproducibility_seed=2306,
        noise_size=5,
    )
    TEST_EXPECTED_ITEM_WEIGHTS = np.array([
        [-0.00837438, 0.11428935, 0.10524099, 0.11820737, -0.05635303,
         0.26041737, -0.21597566, -0.02632179, 0.02793963, -0.16456515],
        [-0.0704511, 0.04129399, 0.14212565, 0.19986424, -0.16963093,
         0.333523, -0.28723386, -0.17897633, 0.10315005, -0.09698798],
        [0.01388724, 0.29223552, 0.23458824, 0.25383765, -0.09255937,
         0.58423424, -0.4872336, -0.00807887, 0.04196675, -0.40107158],
        [-0.04343034, 0.13748571, 0.18658626, 0.23431818, -0.15293357,
         0.45027328, -0.38219845, -0.12802179, 0.08781433, -0.2165019],
        [0.08358474, 0.36665222, 0.1860924, 0.15396345, 0.03844177,
         0.48476437, -0.392482, 0.16512947, -0.04451356, -0.4663991]],
        dtype=np.float32
    )
    TEST_NUM_TRAINING_ITEM_WEIGHTS_TO_SAVE = 0

    @pytest.mark.skipif(
        int(tf.__version__.split(".")[0]) == 1,
        reason="Test requires Tensorflow 2."
    )
    def test_fit(
        self,
        tmp_path: Path,
    ) -> None:
        # Arrange
        recommender = RandomNoiseCFGANRecommender(
            urm_train=self.TEST_URM_TRAIN,
            num_training_item_weights_to_save=self.TEST_NUM_TRAINING_ITEM_WEIGHTS_TO_SAVE,
        )

        # Assert
        assert recommender.model is None
        assert recommender.item_weights is None
        assert recommender.hyper_parameters is None

        # Act
        recommender.fit(
            **attr.asdict(self.TEST_HYPER_PARAMETERS)
        )

        # Assert
        assert recommender.model is not None
        assert recommender.item_weights is not None and np.allclose(
            recommender.item_weights,
            self.TEST_EXPECTED_ITEM_WEIGHTS
        )
        assert recommender.hyper_parameters is not None and recommender.hyper_parameters == self.TEST_HYPER_PARAMETERS

    @pytest.mark.skipif(
        int(tf.__version__.split(".")[0]) == 1,
        reason="Test requires Tensorflow 2."
    )
    def test_save_after_fit(
        self,
        tmp_path: Path,
    ) -> None:
        # Arrange
        folder_path = str(tmp_path.resolve())
        file_name = "test_cfgan_recommender"
        expected_model_dir: str = os.path.join(
            folder_path,
            f"{file_name}_models"
        )
        expected_metadata_file: str = os.path.join(
            expected_model_dir,
            "metadata.zip"
        )
        expected_generator_prefix = "GEN_MODEL"
        expected_discriminator_prefix = "DISC_MODEL"

        recommender = RandomNoiseCFGANRecommender(
            urm_train=self.TEST_URM_TRAIN,
            num_training_item_weights_to_save=self.TEST_NUM_TRAINING_ITEM_WEIGHTS_TO_SAVE,
        )
        recommender.fit(
            **attr.asdict(self.TEST_HYPER_PARAMETERS)
        )

        # Act
        recommender.save_model(
            folder_path=folder_path,
            file_name=file_name,
        )

        # Assert
        assert os.path.exists(expected_model_dir) and os.path.isdir(expected_model_dir)
        assert os.path.exists(expected_metadata_file) and os.path.isfile(expected_metadata_file)
        assert any(
            file.startswith(expected_generator_prefix)
            for file in os.listdir(expected_model_dir)
        )
        assert any(
            file.startswith(expected_discriminator_prefix)
            for file in os.listdir(expected_model_dir)
        )

    @pytest.mark.skipif(
        int(tf.__version__.split(".")[0]) == 1,
        reason="Test requires Tensorflow 2."
    )
    def test_load_after_fit(
        self,
        tmp_path: Path,
    ) -> None:
        # Arrange
        folder_path = str(tmp_path.resolve())
        file_name = "test_cfgan_recommender"

        recommender = RandomNoiseCFGANRecommender(
            urm_train=self.TEST_URM_TRAIN,
            num_training_item_weights_to_save=self.TEST_NUM_TRAINING_ITEM_WEIGHTS_TO_SAVE,
        )
        recommender.fit(
            **attr.asdict(self.TEST_HYPER_PARAMETERS)
        )
        recommender.save_model(
            folder_path=folder_path,
            file_name=file_name,
        )

        new_recommender = RandomNoiseCFGANRecommender(
            urm_train=self.TEST_URM_TRAIN,
            num_training_item_weights_to_save=self.TEST_NUM_TRAINING_ITEM_WEIGHTS_TO_SAVE,
        )

        # Act
        new_recommender.load_model(
            folder_path=folder_path,
            file_name=file_name,
        )

        # Assert
        assert recommender.model is not None and new_recommender.model is not None
        assert recommender.item_weights is not None and new_recommender.item_weights is not None
        assert recommender.hyper_parameters is not None and new_recommender.hyper_parameters is not None

        assert recommender.hyper_parameters == new_recommender.hyper_parameters
        assert np.array_equal(
            recommender.item_weights,
            new_recommender.item_weights,
        )
        assert recommender.model.hyper_parameters == new_recommender.model.hyper_parameters

        # Given the nature of of the Random Noise and the seeds, it is necessary to set the seeds before calling each
        # `get_item_weights` method, this way we can ensure that both item weight arrays are the same. Without a
        # reproducibility seed, these arrays will highly likely be different.
        tf.compat.v1.set_random_seed(self.TEST_HYPER_PARAMETERS.reproducibility_seed)
        np.random.seed(self.TEST_HYPER_PARAMETERS.reproducibility_seed)
        random.seed(self.TEST_HYPER_PARAMETERS.reproducibility_seed)
        recommender_item_weights = recommender.model.get_item_weights()

        tf.compat.v1.set_random_seed(self.TEST_HYPER_PARAMETERS.reproducibility_seed)
        np.random.seed(self.TEST_HYPER_PARAMETERS.reproducibility_seed)
        random.seed(self.TEST_HYPER_PARAMETERS.reproducibility_seed)
        new_recommender_item_weights = new_recommender.model.get_item_weights()

        assert np.array_equal(
            recommender_item_weights,
            new_recommender_item_weights,
        )
