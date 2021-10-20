from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import tensorflow as tf

from conferences.cikm.cfgan.our_implementation.constants import CFGANMode, CFGANMaskType, \
    CFGANBenchmarks, CFGANOptimizer, CFGANActivation
from conferences.cikm.cfgan.our_implementation.models.v1_compat.ClassConditionCFGANModel import ClassConditionCFGANModel
from conferences.cikm.cfgan.our_implementation.parameters import CFGANHyperParameters


class TestRandomNoiseCFGANModel:
    TEST_FILENAME = "tmp_fixed_cfgan_model_save_data_metadata.zip"
    TEST_NUM_INTERACTIONS = 10
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
    TEST_EXPECTED_ITEM_WEIGHTS = {
        0: np.array([
            [-0.01627225, 0.10673425, 0.1110329, 0.12961322, -0.07058235,
             0.27203685, -0.2268979, -0.04543582, 0.03718384, -0.15812258],
            [-0.01480698, 0.16531101, 0.16347389, 0.18930084, -0.09649464,
             0.40112084, -0.3361145, -0.05537012, 0.05061404, -0.23918508],
            [-0.02780626, 0.1989485, 0.21706025, 0.25939977, -0.14479114,
             0.528953, -0.4464286, -0.09896842, 0.07943124, -0.29340962],
            [0.02963718, 0.20119736, 0.12175111, 0.11315742, -0.00723248,
             0.3112287, -0.25327238, 0.05776695, -0.00755981, -0.2647006],
            [0.00702955, 0.16524507, 0.12637967, 0.1327104, -0.04374018,
             0.3162366, -0.26089203, 0.00443044, 0.01729905, -0.22698456]],
            dtype=np.float32
        ),
        10: np.array([
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
    }
    TEST_NUM_TRAINING_ITEM_WEIGHTS_TO_SAVE = 0

    @pytest.mark.skipif(
        int(tf.__version__.split(".")[0]) == 1,
        reason="Test requires Tensorflow 2."
    )
    def test_execution(
        self,
        tmp_path: Path,
    ) -> None:
        # Arrange
        pass

        # Act
        model = ClassConditionCFGANModel(
            urm_train=self.TEST_URM_TRAIN,
            hyper_parameters=self.TEST_HYPER_PARAMETERS,
            num_training_item_weights_to_save=self.TEST_NUM_TRAINING_ITEM_WEIGHTS_TO_SAVE,
            initialize_model=True,
        )
        obtained_item_weights = dict()
        obtained_item_weights[0] = model.get_item_weights(
            urm=None
        )
        model.run_all_epochs()
        obtained_item_weights[self.TEST_HYPER_PARAMETERS.epochs] = model.get_item_weights(
            urm=None
        )

        # Assert none.
        for epoch, expected_item_weight in self.TEST_EXPECTED_ITEM_WEIGHTS.items():
            assert epoch in obtained_item_weights
            assert np.allclose(expected_item_weight, obtained_item_weights[epoch])
            print(f"Epoch={epoch} passed!")

