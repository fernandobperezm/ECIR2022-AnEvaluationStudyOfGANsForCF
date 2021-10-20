from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import tensorflow as tf

from conferences.cikm.cfgan.our_implementation.constants import CFGANMode, CFGANMaskType, \
    CFGANBenchmarks, CFGANOptimizer, CFGANActivation
from conferences.cikm.cfgan.our_implementation.models.v1_compat.CFGANModel import CFGANModel
from conferences.cikm.cfgan.our_implementation.parameters import CFGANHyperParameters


class TestCFGANModel:
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
            [0.00674605, 0.1744552, 0.13549654, 0.14342526, -0.04905511,
             0.3385267, -0.27983156, 0.00146153, 0.0201943, -0.24001512],
            [-0.05806057, 0.08510344, 0.1616187, 0.21433142, -0.16059944,
             0.3844217, -0.3284126, -0.15469557, 0.09512012, -0.15105996],
            [-0.00643594, 0.25380704, 0.23225796, 0.26353818, -0.12077607,
             0.57224464, -0.4799411, -0.05352073, 0.06151499, -0.35823765],
            [-0.05200896, 0.10365342, 0.16874965, 0.21877092, -0.1553148,
             0.4036182, -0.34378386, -0.14254361, 0.09085295, -0.17363235],
            [0.06717069, 0.29251426, 0.14372388, 0.11497127, 0.03747664,
             0.37568316, -0.30231416, 0.13878682, -0.04037279, -0.37152544]],
            dtype=np.float32
        ),
        10: np.array([
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
    }
    TEST_NUM_TRAINING_ITEM_WEIGHTS_TO_SAVE = 0

    @pytest.mark.skipif(
        int(tf.__version__.split(".")[0]) == 1,
        reason="Test requires Tensorflow 2."
    )
    def test_execution_tensorflow_v2(
        self,
        tmp_path: Path,
    ) -> None:
        # Arrange
        pass

        # Act
        model = CFGANModel(
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

