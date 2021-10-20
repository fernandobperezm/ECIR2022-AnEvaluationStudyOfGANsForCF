import os
from pathlib import Path

import attr
import numpy as np
import pytest
import scipy.sparse as sp
import tensorflow as tf

from conferences.cikm.cfgan.our_implementation.constants import CFGANBenchmarks, CFGANMode, CFGANOptimizer, \
    CFGANActivation, CFGANMaskType
from conferences.cikm.cfgan.our_implementation.parameters import CFGANHyperParameters
from conferences.cikm.cfgan.our_implementation.recommenders.ClassConditionCFGANRecommender import \
    ClassConditionCFGANRecommender


class TestClassConditionClassConditionCFGANRecommender:
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
    TEST_HYPER_PARAMETERS = CFGANHyperParameters(
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
    )
    TEST_EXPECTED_ITEM_WEIGHTS = np.array([
        [-0.01583029, 0.1063571, 0.11067197, 0.12925172, -0.070421,
         0.27168292, -0.22656481, -0.04539329, 0.03677309, -0.15784194],
        [-0.01428045, 0.16486458, 0.16304855, 0.18888001, -0.09630407,
         0.40071112, -0.33572692, -0.05531691, 0.0501275, -0.23885669],
        [-0.02720447, 0.19842376, 0.21657462, 0.25892857, -0.144584,
         0.52849257, -0.44599196, -0.09892098, 0.07887896, -0.29302388],
        [0.03010221, 0.20078425, 0.12136664, 0.11277816, -0.00707125,
         0.3108527, -0.25291947, 0.05779455, -0.00799037, -0.26438975],
        [0.0074986, 0.16483523, 0.12599427, 0.13232857, -0.04357394,
         0.3158598, -0.2605377, 0.00446633, 0.01686442, -0.22667795]],
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
        recommender = ClassConditionCFGANRecommender(
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

        recommender = ClassConditionCFGANRecommender(
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

        recommender = ClassConditionCFGANRecommender(
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

        new_recommender = ClassConditionCFGANRecommender(
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
        assert np.array_equal(
            recommender.model.get_item_weights(),
            new_recommender.model.get_item_weights(),
        )
