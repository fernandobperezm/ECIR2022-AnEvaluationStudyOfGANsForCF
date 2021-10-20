from typing import Optional, Any

import scipy.sparse as sp
from recsys_framework.Recommenders.DataIO import DataIO
from recsys_framework.Utils.conf_logging import get_logger

from conferences.cikm.cfgan.our_implementation.constants import CFGANMode, CFGANMaskType
from conferences.cikm.cfgan.our_implementation.models.v1_compat.RandomNoiseCFGANModel import RandomNoiseCFGANModel
from conferences.cikm.cfgan.our_implementation.parameters import RandomNoiseCFGANHyperParameters
from conferences.cikm.cfgan.our_implementation.recommenders.CFGANRecommender import \
    CFGANRecommender

logger = get_logger(__name__)


class RandomNoiseCFGANRecommender(CFGANRecommender):
    RECOMMENDER_NAME = "RandomNoiseCFGANRecommender"
    RANDOM_NOISE_SIZE_KWARG_NAME = "cfgan_random_noise_size"

    def __init__(
        self,
        urm_train: sp.csr_matrix,
        num_training_item_weights_to_save: int = 0,
    ):
        super().__init__(
            urm_train=urm_train,
            num_training_item_weights_to_save=num_training_item_weights_to_save,
        )
        self.model: Optional[RandomNoiseCFGANModel] = None

    @staticmethod
    def get_recommender_name(
        cfgan_mode: CFGANMode,
        cfgan_mask_type: CFGANMaskType,
        **kwargs,
    ) -> str:
        if RandomNoiseCFGANRecommender.RANDOM_NOISE_SIZE_KWARG_NAME in kwargs:
            cfgan_random_noise_size = int(kwargs[RandomNoiseCFGANRecommender.RANDOM_NOISE_SIZE_KWARG_NAME])
            return (
                f"{RandomNoiseCFGANRecommender.RECOMMENDER_NAME}_"
                f"{cfgan_mode.value}_"
                f"{cfgan_mask_type.value}"
                f"noise_size_{cfgan_random_noise_size}"
            )

        return (
            f"{RandomNoiseCFGANRecommender.RECOMMENDER_NAME}_"
            f"{cfgan_mode.value}_"
            f"{cfgan_mask_type.value}"
        )

    def fit(
        self,
        **kwargs: dict[str, Any]
    ) -> None:

        try:
            self.hyper_parameters = RandomNoiseCFGANHyperParameters(**kwargs)
        except TypeError as e:
            logger.exception(f"Kwargs {kwargs} could not be converted into CFGANHyperParameters")
            raise e

        self.RECOMMENDER_NAME = self.get_recommender_name(
            cfgan_mode=self.hyper_parameters.mode,
            cfgan_mask_type=self.hyper_parameters.mask_type,
            cfgan_random_noise_size=self.hyper_parameters.noise_size,
        )

        self.model = RandomNoiseCFGANModel(
            urm_train=self.URM_train,
            hyper_parameters=self.hyper_parameters,
            num_training_item_weights_to_save=self.num_training_item_weights_to_save,
            initialize_model=True,
        )

        self.model.run_all_epochs()
        self.item_weights = self.model.get_item_weights()

    def load_model(
        self,
        folder_path: str,
        file_name: Optional[str] = None,
    ) -> None:
        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        data_io = DataIO(
            folder_path=folder_path
        )
        data_dict = data_io.load_data(
            file_name=file_name
        )

        self.hyper_parameters = RandomNoiseCFGANHyperParameters(
            **data_dict["hyper_parameters"]
        )
        self.item_weights = data_dict["item_weights"]
        self.num_training_item_weights_to_save = data_dict.get(
            "save_training_item_weights_every",
            0,
        )

        self.model = RandomNoiseCFGANModel(
            urm_train=self.URM_train,
            hyper_parameters=self.hyper_parameters,
            num_training_item_weights_to_save=self.num_training_item_weights_to_save,
            initialize_model=False,
        )
        self.model.load_model(
            folder_path=folder_path,
            file_name=file_name
        )
