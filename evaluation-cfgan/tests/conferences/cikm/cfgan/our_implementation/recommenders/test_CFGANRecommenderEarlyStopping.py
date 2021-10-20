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
from conferences.cikm.cfgan.our_implementation.recommenders.CFGANRecommenderEarlyStopping import CFGANRecommenderEarlyStopping


class TestCFGANRecommenderEarlyStopping:
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
        [0.00722052, 0.17413032, 0.13507938, 0.14303291, -0.04907871,
         0.3380974, -0.27956042, 0.00185272, 0.02050101, -0.2396721],
        [-0.05753589, 0.08479333, 0.16117114, 0.21389103, -0.16057262,
         0.38395023, -0.32810602, -0.15422547, 0.09540772, -0.15073305],
        [-0.00581824, 0.25337082, 0.23170353, 0.2630232, -0.12080385,
         0.5716689, -0.47957754, -0.05301622, 0.06191684, -0.35777652],
        [-0.05149199, 0.10331406, 0.16830505, 0.21834987, -0.15532717,
         0.40316543, -0.34349898, -0.14211333, 0.09117432, -0.17327842],
        [0.06766552, 0.29215604, 0.1432746, 0.11455517, 0.03743808,
         0.3752159, -0.30201885, 0.13918824, -0.04003751, -0.37114266]],
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
        recommender = CFGANRecommenderEarlyStopping(
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
        assert recommender.item_weights is not None
        assert recommender.item_weights is not None and np.allclose(
            recommender.item_weights,
            self.TEST_EXPECTED_ITEM_WEIGHTS
        )
        assert np.allclose(recommender.item_weights, self.TEST_EXPECTED_ITEM_WEIGHTS)

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

        recommender = CFGANRecommenderEarlyStopping(
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
        expected_model_dir: str = os.path.join(
            folder_path,
            f"{file_name}_models"
        )

        recommender = CFGANRecommenderEarlyStopping(
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

        new_recommender = CFGANRecommenderEarlyStopping(
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
