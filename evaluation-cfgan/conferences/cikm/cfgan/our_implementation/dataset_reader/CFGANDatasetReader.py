import os
from typing import Dict, List

import numpy as np
import scipy.sparse as sps

from recsys_framework.Data_manager.Utility import print_stat_datareader
from conferences.cikm.cfgan.our_implementation.constants import CFGANDatasetMode, CFGANBenchmarks
from conferences.cikm.cfgan.our_implementation.original.data import \
    loadTestData as OriginalTestDataReader
from conferences.cikm.cfgan.our_implementation.original.data import \
    loadTrainingData as OriginalTrainingDataReader
from recsys_framework.Data_manager.data_consistency_check import assert_implicit_data, assert_disjoint_matrices
from recsys_framework.Data_manager.load_and_save_data import save_data_dict_zip, load_data_dict_zip
from recsys_framework.Data_manager.split_functions.split_train_validation import \
    split_train_validation_percentage_random_holdout, split_train_validation_cold_start_user_wise
from recsys_framework.Utils.conf_logging import get_logger

logger = get_logger(__name__)


def dict_to_sparse_csr(data: Dict[int, List[int]], num_users: int, num_items: int) -> \
        sps.csr_matrix:
    mat = sps.dok_matrix((num_users, num_items), dtype=np.int32)
    for user_id, item_ids in data.items():
        for item_id in item_ids:
            mat[user_id, item_id] = 1

    mat = mat.tocsr()
    new_num_users, new_num_items = mat.shape
    if new_num_users != num_users or new_num_items != num_items:
        raise ValueError(
            f"New dimensions do not match. "
            f"Previous #Users: {num_users} - Actual #Users: {new_num_users}. "
            f"Previous #Items: {num_items} - Actual #Items: {new_num_items}."
        )

    return mat


class CFGANDatasetReader:
    CONFERENCE = "CIKM"
    MODEL = "CFGAN"
    ORIGINAL_DATA_PATH = os.path.join("", "conferences", "cikm", "cfgan", "original_source_code", "datasets")

    NAME_URM_TRAIN = "URM_train"
    NAME_URM_VALIDATION = "URM_validation"
    NAME_URM_VALIDATION_CONDITION = "URM_validation_condition"
    NAME_URM_VALIDATION_PROFILE = "URM_validation_profile"
    NAME_URM_TEST = "URM_test"
    NAME_URM_TEST_CONDITION = "URM_test_condition"
    NAME_URM_TEST_PROFILE = "URM_test_profile"

    def __init__(
        self,
        dataset: CFGANBenchmarks,
        mode: CFGANDatasetMode
    ):
        if not isinstance(dataset, CFGANBenchmarks):
            raise ValueError("Variable <dataset_name> must be an instance of <CFGANBenchmarks>.")

        if not isinstance(mode, CFGANDatasetMode):
            raise ValueError("Variable <mode> must be an instance of <CFGANDatasetMode>.")

        self.mode = mode
        self.dataset_name: str = dataset.value
        self.URM_DICT: dict[str, sps.csr_matrix] = {}
        self.ICM_DICT: dict[str, sps.csr_matrix] = {}

    def load_data(self) -> None:
        if self.mode != CFGANDatasetMode.ORIGINAL:
            raise ValueError(f"Dataset can only be loaded if it's the original split.")

        pre_split_path: str = os.path.join(
            ".",
            "data_split",
            f"{self.dataset_name}/"
        )
        pre_split_filename = f"{self.mode.value}_split_data_"

        # If directory does not exist, create
        os.makedirs(pre_split_path, exist_ok=True)

        try:
            logger.info(f"Dataset_{self.dataset_name}: Attempting to load pre-split data")

            for attrib_name, attrib_object in load_data_dict_zip(pre_split_path, pre_split_filename).items():
                self.__setattr__(attrib_name, attrib_object)

        except FileNotFoundError:

            logger.info(f"Dataset_{self.dataset_name}: Pre-split data not found, building new one")

            train_split: Dict[int, List[int]]
            test_split: Dict[int, List[int]]
            num_users: int
            num_items: int

            train_split, num_users, num_items = OriginalTrainingDataReader(
                benchmark=self.dataset_name,
                path=self.ORIGINAL_DATA_PATH
            )

            test_split, _ = OriginalTestDataReader(
                benchmark=self.dataset_name,
                path=self.ORIGINAL_DATA_PATH
            )

            urm_train = dict_to_sparse_csr(train_split, num_users, num_items)
            urm_test = dict_to_sparse_csr(test_split, num_users, num_items)

            original_urm_train = urm_train.copy()

            logger.info(
                "Loading dataset using the original splits. Still need to create a validation set."
            )
            urm_test = urm_test.copy()

            urm_train, urm_validation = split_train_validation_percentage_random_holdout(
                URM_train=urm_train.copy(),
                train_percentage=0.8
            )

            urm_validation_condition, urm_validation_profile = split_train_validation_percentage_random_holdout(
                URM_train=urm_validation.copy(),
                train_percentage=0.8,
            )
            urm_test_condition, urm_test_profile = split_train_validation_percentage_random_holdout(
                URM_train=urm_test.copy(),
                train_percentage=0.8,
            )

            assert_implicit_data([
                urm_train,
                urm_validation_condition, urm_validation_profile, urm_validation,
                urm_test_condition, urm_test_profile, urm_test,
            ])
            assert_disjoint_matrices([
                urm_train,
                urm_validation_condition, urm_validation_profile,
                urm_test_condition, urm_test_profile,
            ])
            assert_disjoint_matrices([
                urm_train,
                urm_validation,
                urm_test,
            ])

            assert np.array_equal(
                urm_validation.toarray(),
                (urm_validation_profile + urm_validation_condition).toarray()
            )
            assert np.array_equal(
                urm_test.toarray(),
                (urm_test_profile + urm_test_condition).toarray()
            )
            assert np.array_equal(
                original_urm_train.toarray(),
                (urm_train + urm_validation).toarray(),
            )

            self.URM_DICT = {
                self.NAME_URM_TRAIN: urm_train,
                self.NAME_URM_VALIDATION: urm_validation,
                self.NAME_URM_VALIDATION_CONDITION: urm_validation_condition,
                self.NAME_URM_VALIDATION_PROFILE: urm_validation_profile,
                self.NAME_URM_TEST: urm_test,
                self.NAME_URM_TEST_CONDITION: urm_test_condition,
                self.NAME_URM_TEST_PROFILE: urm_test_profile,
            }

            self.ICM_DICT = {}

            save_data_dict_zip(self.URM_DICT, self.ICM_DICT, pre_split_path, pre_split_filename)

        logger.info(f"{self.dataset_name}: Dataset loaded")

        print_stat_datareader(self)

    def load_data_cold_users(self) -> None:
        if self.mode != CFGANDatasetMode.COLD_USERS:
            raise ValueError(
                f"Cold users split can only be generated using mode={CFGANDatasetMode.COLD_USERS}"
            )

        pre_split_path: str = os.path.join(
            ".",
            "data_split",
            f"{self.dataset_name}/"
        )
        pre_split_filename = f"{self.mode.value}_cold_user_split_data_"

        # If directory does not exist, create
        os.makedirs(pre_split_path, exist_ok=True)

        try:
            logger.info(
                f"Dataset_{self.dataset_name}: Attempting to load cold-users pre-split data"
            )

            for attrib_name, attrib_object in load_data_dict_zip(pre_split_path, pre_split_filename).items():
                self.__setattr__(attrib_name, attrib_object)

        except FileNotFoundError:
            logger.info(
                f"Dataset_{self.dataset_name}: Pre-split data not found, building new one"
            )

            train_split: Dict[int, List[int]]
            test_split: Dict[int, List[int]]
            num_users: int
            num_items: int

            train_split, num_users, num_items = OriginalTrainingDataReader(
                benchmark=self.dataset_name,
                path=self.ORIGINAL_DATA_PATH
            )

            test_split, _ = OriginalTestDataReader(
                benchmark=self.dataset_name,
                path=self.ORIGINAL_DATA_PATH
            )

            urm_train = dict_to_sparse_csr(train_split, num_users, num_items)
            urm_test = dict_to_sparse_csr(test_split, num_users, num_items)

            urm_full: sps.csr_matrix = urm_train + urm_test
            urm_train, urm_test = split_train_validation_cold_start_user_wise(
                URM_train=urm_full.copy(),
                full_train_percentage=0.8,
                cold_items=0,
                verbose=True,
            )
            urm_train, urm_validation = split_train_validation_cold_start_user_wise(
                URM_train=urm_train.copy(),
                full_train_percentage=0.8,
                cold_items=0,
                verbose=True,
            )
            urm_validation_condition, urm_validation_profile = split_train_validation_percentage_random_holdout(
                URM_train=urm_validation.copy(),
                train_percentage=0.8,
            )
            urm_test_condition, urm_test_profile = split_train_validation_percentage_random_holdout(
                URM_train=urm_test.copy(),
                train_percentage=0.8,
            )

            self.URM_DICT = {
                "URM_train": urm_train,
                "URM_validation": urm_validation,
                "URM_validation_condition": urm_validation_condition,
                "URM_validation_profile": urm_validation_profile,
                "URM_test": urm_test,
                "URM_test_condition": urm_test_condition,
                "URM_test_profile": urm_test_profile,
            }

            self.ICM_DICT = {}

            save_data_dict_zip(self.URM_DICT, self.ICM_DICT, pre_split_path, pre_split_filename)

        logger.info(f"{self.dataset_name}: Cold User Dataset loaded")

        print_stat_datareader(self)
