from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import tensorflow as tf

from conferences.cikm.cfgan.our_implementation.constants import CFGANMode, CFGANMaskType, \
    CFGANBenchmarks, CFGANOptimizer, CFGANActivation
from conferences.cikm.cfgan.our_implementation.models.v1_compat.RandomNoiseCFGANModel import RandomNoiseCFGANModel
from conferences.cikm.cfgan.our_implementation.parameters import RandomNoiseCFGANHyperParameters


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
    TEST_EXPECTED_ITEM_WEIGHTS = {
        0: np.array([
            [-0.00771603, 0.11801331, 0.10709855, 0.11969054, -0.05559515,
             0.26478615, -0.21947391, -0.02469558, 0.02715313, -0.16914333],
            [-0.06844258, 0.03905341, 0.13604553, 0.1912981, -0.16251332,
             0.31875816, -0.27429357, -0.17215621, 0.09863645, -0.09245089],
            [0.00525498, 0.3107912, 0.265864, 0.29503313, -0.12119924,
             0.6579858, -0.55088633, -0.0350797, 0.05892677, -0.43140405],
            [-0.06358397, 0.0391831, 0.12792745, 0.17886323, -0.15084806,
             0.30020362, -0.2578382, -0.158885, 0.09126249, -0.08993806],
            [0.01590512, 0.16253312, 0.10878471, 0.10679898, -0.02094785,
             0.27555707, -0.22508, 0.02937599, 0.00303993, -0.21827246]],
            dtype=np.float32
        ),
        10: np.array([
            [-0.01481019, 0.10655445, 0.10883806, 0.12641507, -0.06799196,
             0.2674457, -0.22282334, -0.04211024, 0.03563209, -0.15737641],
            [-0.04525711, 0.10315618, 0.15735564, 0.20165858, -0.13994882,
             0.37800035, -0.3210994, -0.12477341, 0.08126589, -0.16955695],
            [-0.00175916, 0.26122308, 0.23192894, 0.26058728, -0.11468922,
             0.57330155, -0.48019516, -0.0435133, 0.05741607, -0.36647072],
            [-0.06131838, 0.02203693, 0.10845051, 0.15518296, -0.13775966,
             0.2534247, -0.21787463, -0.14930728, 0.08393192, -0.06501901],
            [0.06625363, 0.3169627, 0.16876791, 0.14480805, 0.02168092,
             0.4373539, -0.35478485, 0.1295647, -0.03152839, -0.40661258]],
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
        model = RandomNoiseCFGANModel(
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

