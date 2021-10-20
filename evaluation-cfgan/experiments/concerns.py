import math
import os
import uuid
from typing import Any, Optional, Type

import attr
import numpy as np
import pandas as pd
import scipy.sparse as sp

import experiments.commons as commons
import experiments.reproducibility as reproducibility
from conferences.cikm.cfgan.our_implementation.constants import CFGANMode, CFGANMaskType, \
    CFGANBenchmarks
from conferences.cikm.cfgan.our_implementation.parameters import get_cfgan_code_hyper_parameters, \
    get_cfgan_paper_hyper_parameters, CFGANHyperParameters, RandomNoiseCFGANHyperParameters
from conferences.cikm.cfgan.our_implementation.recommenders.CFGANRecommender import CFGANRecommender
from conferences.cikm.cfgan.our_implementation.recommenders.CFGANRecommenderEarlyStopping import \
    CFGANRecommenderEarlyStopping
from conferences.cikm.cfgan.our_implementation.recommenders.ClassConditionCFGANRecommender import \
    ClassConditionCFGANRecommender
from conferences.cikm.cfgan.our_implementation.recommenders.RandomNoiseCFGANRecommender import \
    RandomNoiseCFGANRecommender
from recsys_framework.Evaluation.Evaluator import EvaluatorHoldout
from recsys_framework.HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from recsys_framework.HyperparameterTuning.SearchSingleCase import SearchSingleCase
from recsys_framework.Recommenders.BaseRecommender import BaseRecommender
from recsys_framework.Recommenders.DataIO import DataIO
from recsys_framework.Utils.conf_dask import DaskInterface
from recsys_framework.Utils.conf_logging import get_logger
from recsys_framework.Utils.plotting import generate_accuracy_and_beyond_metrics_latex

logger = get_logger(__name__)


####################################################################################################
####################################################################################################
#                                FOLDERS VARIABLES                            #
####################################################################################################
####################################################################################################
BASE_FOLDER = os.path.join(
    commons.RESULTS_EXPERIMENTS_DIR,
    "concerns",
    "{benchmark}",
)
LATEX_RESULTS_FOLDER = os.path.join(
    BASE_FOLDER,
    "latex",
    "",
)
ARTICLE_ACCURACY_METRICS_BASELINES_LATEX_DIR = os.path.join(
    LATEX_RESULTS_FOLDER,
    "article",
    "accuracy_and_beyond_accuracy",
)
ACCURACY_METRICS_BASELINES_LATEX_DIR = os.path.join(
    LATEX_RESULTS_FOLDER,
    "accuracy_and_beyond_accuracy",
)
SINGLE_EXECUTION_EXPERIMENTS_CONCERNS_DIR = os.path.join(
    BASE_FOLDER,
    "experiments",
    "single_execution",
    ""
)

commons.FOLDERS.add(BASE_FOLDER)
commons.FOLDERS.add(LATEX_RESULTS_FOLDER)
commons.FOLDERS.add(ACCURACY_METRICS_BASELINES_LATEX_DIR)
commons.FOLDERS.add(ARTICLE_ACCURACY_METRICS_BASELINES_LATEX_DIR)
commons.FOLDERS.add(SINGLE_EXECUTION_EXPERIMENTS_CONCERNS_DIR)

EPOCHS_COMPARISON_RESULTS_FILE = os.path.join(
    LATEX_RESULTS_FOLDER,
    "epochs_comparison_results_file.tex",
)


####################################################################################################
####################################################################################################
#                                REPRODUCIBILITY VARIABLES                            #
####################################################################################################
####################################################################################################
CONCERNS_SEED = 1234567890


####################################################################################################
####################################################################################################
#               EXPERIMENTS OF CFGAN CONCERNS (SINGLE RUN RANDOM NOISE &CONDITION AS CLASS)        #
####################################################################################################
####################################################################################################
def get_best_hyper_parameters_of_tuned_cfgan_model(
    benchmark: CFGANBenchmarks,
    cfgan_mode: CFGANMode,
    cfgan_mask_type: CFGANMaskType,
) -> dict[str, Any]:
    tuned_recommenders_folder_path = reproducibility.HYPER_PARAMETER_TUNING_EXPERIMENTS_DIR.format(
        benchmark=benchmark.value,
    )
    tuned_recommender_name = CFGANRecommenderEarlyStopping.get_recommender_name(
        cfgan_mode=cfgan_mode,
        cfgan_mask_type=cfgan_mask_type,
    )

    tuned_recommender_metadata_filename = f"{tuned_recommender_name}_metadata"

    data_io = DataIO(
        folder_path=tuned_recommenders_folder_path,
    )
    tuned_recommender_metadata = data_io.load_data(
        file_name=tuned_recommender_metadata_filename
    )

    return tuned_recommender_metadata["hyperparameters_best"]


def calculate_random_noise_sizes(
    cfgan_mode: CFGANMode,
    urm: sp.csr_matrix,
) -> list[int]:
    if cfgan_mode == CFGANMode.ITEM_BASED:
        num_cols, num_rows = urm.shape
    else:
        num_rows, num_cols = urm.shape

    return [
        math.floor(num_cols / 2),
        num_cols,
        math.ceil(num_cols * 2)
    ]


def _run_experiment_cfgan_with_random_noise(
    benchmark: CFGANBenchmarks,
    cfgan_mode: CFGANMode,
    cfgan_mask_type: CFGANMaskType,
    cfgan_noise_size: int,
    urm_train: sp.csr_matrix,
    urm_validation: sp.csr_matrix,
    urm_test: sp.csr_matrix,
) -> None:
    import random
    import tensorflow as tf
    import numpy as np

    random.seed(CONCERNS_SEED)
    np.random.seed(CONCERNS_SEED)
    tf.random.set_seed(CONCERNS_SEED)

    tuned_recommender_best_hyper_parameters = get_best_hyper_parameters_of_tuned_cfgan_model(
        benchmark=benchmark,
        cfgan_mode=cfgan_mode,
        cfgan_mask_type=cfgan_mask_type,
    )

    experiments_recommender_folder_path = SINGLE_EXECUTION_EXPERIMENTS_CONCERNS_DIR.format(
        benchmark=benchmark.value
    )
    experiment_recommender_class = RandomNoiseCFGANRecommender
    experiment_recommender_name = experiment_recommender_class.get_recommender_name(
        cfgan_mode=cfgan_mode,
        cfgan_mask_type=cfgan_mask_type,
        cfgan_random_noise_size=cfgan_noise_size,
    )
    experiment_recommender_hyper_parameters = RandomNoiseCFGANHyperParameters(
        **tuned_recommender_best_hyper_parameters,
        noise_size=cfgan_noise_size,
    )

    evaluator_validation = EvaluatorHoldout(
        URM_test_list=urm_validation,
        cutoff_list=reproducibility.VALIDATION_CUTOFFS,
    )

    evaluator_test = EvaluatorHoldout(
        URM_test_list=urm_test,
        cutoff_list=reproducibility.TEST_CUTOFFS,
    )

    parameter_search = SearchSingleCase(
        experiment_recommender_class,
        evaluator_validation=evaluator_validation,
        evaluator_test=evaluator_test
    )

    recommender_input_args = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS=[],
        CONSTRUCTOR_KEYWORD_ARGS={
            "urm_train": urm_train,
            "num_training_item_weights_to_save": 0,
        },
        FIT_POSITIONAL_ARGS=[],
        FIT_KEYWORD_ARGS={}
    )

    recommender_input_args_last_test = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS=[],
        CONSTRUCTOR_KEYWORD_ARGS={
            "urm_train": urm_train + urm_validation,
            "num_training_item_weights_to_save": 0,
        },
        FIT_POSITIONAL_ARGS=[],
        FIT_KEYWORD_ARGS={}
    )

    parameter_search.search(
        recommender_input_args=recommender_input_args,
        recommender_input_args_last_test=recommender_input_args_last_test,
        fit_hyperparameters_values=attr.asdict(experiment_recommender_hyper_parameters),
        output_folder_path=experiments_recommender_folder_path,
        output_file_name_root=experiment_recommender_name,
        resume_from_saved=True,
        save_model="best",
        evaluate_on_test="best",
    )

    logger.info(
        f"Scheduled the experiment for {experiment_recommender_name} "
        f"\n{experiment_recommender_hyper_parameters}"
    )


def _run_experiment_cfgan_with_class_condition(
    benchmark: CFGANBenchmarks,
    cfgan_mode: CFGANMode,
    cfgan_mask_type: CFGANMaskType,
    urm_train: sp.csr_matrix,
    urm_validation: sp.csr_matrix,
    urm_test: sp.csr_matrix,
) -> None:
    import random
    import tensorflow as tf
    import numpy as np

    random.seed(CONCERNS_SEED)
    np.random.seed(CONCERNS_SEED)
    tf.random.set_seed(CONCERNS_SEED)

    tuned_recommender_best_hyper_parameters = get_best_hyper_parameters_of_tuned_cfgan_model(
        benchmark=benchmark,
        cfgan_mode=cfgan_mode,
        cfgan_mask_type=cfgan_mask_type,
    )

    experiments_recommender_folder_path = SINGLE_EXECUTION_EXPERIMENTS_CONCERNS_DIR.format(
        benchmark=benchmark.value
    )
    experiment_recommender_class = ClassConditionCFGANRecommender
    experiment_recommender_name = experiment_recommender_class.get_recommender_name(
        cfgan_mode=cfgan_mode,
        cfgan_mask_type=cfgan_mask_type,
    )
    experiment_recommender_hyper_parameters = CFGANHyperParameters(
        **tuned_recommender_best_hyper_parameters,
    )

    evaluator_validation = EvaluatorHoldout(
        URM_test_list=urm_validation,
        cutoff_list=reproducibility.VALIDATION_CUTOFFS,
    )

    evaluator_test = EvaluatorHoldout(
        URM_test_list=urm_test,
        cutoff_list=reproducibility.TEST_CUTOFFS,
    )

    parameter_search = SearchSingleCase(
        experiment_recommender_class,
        evaluator_validation=evaluator_validation,
        evaluator_test=evaluator_test,
    )

    recommender_input_args = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS=[],
        CONSTRUCTOR_KEYWORD_ARGS={
            "urm_train": urm_train,
            "num_training_item_weights_to_save": 0,
        },
        FIT_POSITIONAL_ARGS=[],
        FIT_KEYWORD_ARGS={}
    )

    recommender_input_args_last_test = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS=[],
        CONSTRUCTOR_KEYWORD_ARGS={
            "urm_train": urm_train + urm_validation,
            "num_training_item_weights_to_save": 0,
        },
        FIT_POSITIONAL_ARGS=[],
        FIT_KEYWORD_ARGS={}
    )

    parameter_search.search(
        recommender_input_args=recommender_input_args,
        recommender_input_args_last_test=recommender_input_args_last_test,
        fit_hyperparameters_values=attr.asdict(experiment_recommender_hyper_parameters),
        output_folder_path=experiments_recommender_folder_path,
        output_file_name_root=experiment_recommender_name,
        resume_from_saved=True,
        save_model="best",
        evaluate_on_test="best",
    )

    logger.info(
        f"Scheduled the experiment for {experiment_recommender_name} "
        f"\n{experiment_recommender_hyper_parameters}"
    )


def _run_experiment_cfgan_without_early_stopping(
    benchmark: CFGANBenchmarks,
    cfgan_mode: CFGANMode,
    cfgan_mask_type: CFGANMaskType,
    urm_train: sp.csr_matrix,
    urm_validation: sp.csr_matrix,
    urm_test: sp.csr_matrix,
) -> None:
    import random
    import tensorflow as tf
    import numpy as np

    random.seed(CONCERNS_SEED)
    np.random.seed(CONCERNS_SEED)
    tf.random.set_seed(CONCERNS_SEED)

    tuned_recommender_best_hyper_parameters = get_best_hyper_parameters_of_tuned_cfgan_model(
        benchmark=benchmark,
        cfgan_mode=cfgan_mode,
        cfgan_mask_type=cfgan_mask_type,
    )

    experiments_recommender_folder_path = SINGLE_EXECUTION_EXPERIMENTS_CONCERNS_DIR.format(
        benchmark=benchmark.value
    )
    experiment_recommender_class = CFGANRecommender
    experiment_recommender_name = experiment_recommender_class.get_recommender_name(
        cfgan_mode=cfgan_mode,
        cfgan_mask_type=cfgan_mask_type,
    )
    experiment_recommender_hyper_parameters = CFGANHyperParameters(
        **{
            **tuned_recommender_best_hyper_parameters,
            "epochs": reproducibility.CFGAN_NUM_EPOCHS,
        },
    )

    evaluator_validation = EvaluatorHoldout(
        URM_test_list=urm_validation,
        cutoff_list=reproducibility.VALIDATION_CUTOFFS,
    )

    evaluator_test = EvaluatorHoldout(
        URM_test_list=urm_test,
        cutoff_list=reproducibility.TEST_CUTOFFS,
    )

    parameter_search = SearchSingleCase(
        experiment_recommender_class,
        evaluator_validation=evaluator_validation,
        evaluator_test=evaluator_test,
    )

    recommender_input_args = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS=[],
        CONSTRUCTOR_KEYWORD_ARGS={
            "urm_train": urm_train,
            "num_training_item_weights_to_save": 0,
        },
        FIT_POSITIONAL_ARGS=[],
        FIT_KEYWORD_ARGS={}
    )

    recommender_input_args_last_test = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS=[],
        CONSTRUCTOR_KEYWORD_ARGS={
            "urm_train": urm_train + urm_validation,
            "num_training_item_weights_to_save": 0,
        },
        FIT_POSITIONAL_ARGS=[],
        FIT_KEYWORD_ARGS={}
    )

    parameter_search.search(
        recommender_input_args=recommender_input_args,
        recommender_input_args_last_test=recommender_input_args_last_test,
        fit_hyperparameters_values=attr.asdict(experiment_recommender_hyper_parameters),
        output_folder_path=experiments_recommender_folder_path,
        output_file_name_root=experiment_recommender_name,
        resume_from_saved=True,
        save_model="best",
        evaluate_on_test="best",
    )


def run_concerns_experiments(
    include_cfgan_with_random_noise: bool,
    include_cfgan_with_class_condition: bool,
    include_cfgan_without_early_stopping: bool,
    dask_interface: DaskInterface,
) -> None:
    for dataset in commons.datasets():
        future_urm_train = dask_interface.scatter_data(
            data=dataset.urm_train,
        )
        future_urm_validation = dask_interface.scatter_data(
            data=dataset.urm_validation,
        )
        future_urm_test = dask_interface.scatter_data(
            data=dataset.urm_test,
        )

        if include_cfgan_with_random_noise:
            for cfgan_mode, cfgan_mask_type in reproducibility.cfgan_hyper_parameter_search_settings():
                for noise_size in calculate_random_noise_sizes(
                    cfgan_mode=cfgan_mode,
                    urm=dataset.urm_train
                ):
                    dask_interface.submit_job(
                        job_key=(
                            f"_run_experiment_cfgan_with_random_noise"
                            f"|{dataset.benchmark.value}"
                            f"|RandomNoiseCFGANRecommender"
                            f"|{cfgan_mode.value}"
                            f"|{cfgan_mask_type.value}"
                            f"|{noise_size}"
                            f"|{uuid.uuid4()}"
                        ),
                        job_priority=dataset.priority,
                        job_info={
                            "recommender": "RandomNoiseCFGANRecommender",
                            "benchmark": dataset.benchmark.value,
                            "cfgan_mode": cfgan_mode.value,
                            "cfgan_mask_type": cfgan_mask_type.value,
                            "cfgan_noise_size": noise_size,
                        },
                        method=_run_experiment_cfgan_with_random_noise,
                        method_kwargs={
                            "benchmark": dataset.benchmark,
                            "urm_train": future_urm_train,
                            "urm_validation": future_urm_validation,
                            "urm_test": future_urm_test,
                            "cfgan_mode": cfgan_mode,
                            "cfgan_mask_type": cfgan_mask_type,
                            "cfgan_noise_size": noise_size,
                        }
                    )

        if include_cfgan_with_class_condition:
            for cfgan_mode, cfgan_mask_type in reproducibility.cfgan_hyper_parameter_search_settings():
                dask_interface.submit_job(
                    job_key=(
                        f"_run_experiment_cfgan_with_class_condition"
                        f"|{dataset.benchmark.value}"
                        f"|ClassConditionCFGANRecommender"
                        f"|{cfgan_mode.value}"
                        f"|{cfgan_mask_type.value}"
                        f"|{uuid.uuid4()}"
                    ),
                    job_priority=dataset.priority,
                    job_info={
                        "recommender": "ClassConditionCFGANRecommender",
                        "benchmark": dataset.benchmark.value,
                        "cfgan_mode": cfgan_mode.value,
                        "cfgan_mask_type": cfgan_mask_type.value,
                    },
                    method=_run_experiment_cfgan_with_class_condition,
                    method_kwargs={
                        "benchmark": dataset.benchmark,
                        "urm_train": future_urm_train,
                        "urm_validation": future_urm_validation,
                        "urm_test": future_urm_test,
                        "cfgan_mode": cfgan_mode,
                        "cfgan_mask_type": cfgan_mask_type,
                    }
                )

        if include_cfgan_without_early_stopping:
            for cfgan_mode, cfgan_mask_type in reproducibility.cfgan_hyper_parameter_search_settings():
                dask_interface.submit_job(
                    job_key=(
                        f"_run_experiment_cfgan_without_early_stopping"
                        f"|{dataset.benchmark.value}"
                        f"|CFGANRecommender"
                        f"|{cfgan_mode.value}"
                        f"|{cfgan_mask_type.value}"
                        f"|{uuid.uuid4()}"
                    ),
                    job_priority=dataset.priority,
                    job_info={
                        "recommender": "CFGANRecommender",
                        "benchmark": dataset.benchmark.value,
                        "cfgan_mode": cfgan_mode.value,
                        "cfgan_mask_type": cfgan_mask_type.value,
                    },
                    method=_run_experiment_cfgan_without_early_stopping,
                    method_kwargs={
                        "benchmark": dataset.benchmark,
                        "urm_train": future_urm_train,
                        "urm_validation": future_urm_validation,
                        "urm_test": future_urm_test,
                        "cfgan_mode": cfgan_mode,
                        "cfgan_mask_type": cfgan_mask_type,
                    }
                )


####################################################################################################
####################################################################################################
#             Experiments Concerns: Results exporting          #
####################################################################################################
####################################################################################################
def _print_article_accuracy_and_beyond_accuracy_metrics(
    urm: sp.csr_matrix,
    benchmark: CFGANBenchmarks,
    num_test_users: int,
) -> None:
    other_algorithm_list: list[Optional[BaseRecommender]] = []

    export_experiments_folder_path = ARTICLE_ACCURACY_METRICS_BASELINES_LATEX_DIR.format(
        benchmark=benchmark.value,
    )

    experiments_folder_path = SINGLE_EXECUTION_EXPERIMENTS_CONCERNS_DIR.format(
        benchmark=benchmark.value,
    )

    recommenders: list[Type[CFGANRecommender]] = [
        CFGANRecommender,
        ClassConditionCFGANRecommender,
        RandomNoiseCFGANRecommender,
    ]

    modes = [
        CFGANMode.ITEM_BASED,
        CFGANMode.USER_BASED,
    ]

    mask_types = [
        CFGANMaskType.ZERO_RECONSTRUCTION_AND_PARTIAL_MASKING,
    ]

    for cfgan_mode in modes:
        for cfgan_mask_type in mask_types:
            for recommender_class in recommenders:
                if recommender_class == RandomNoiseCFGANRecommender:
                    for random_noise_size in calculate_random_noise_sizes(
                        cfgan_mode=cfgan_mode,
                        urm=urm,
                    ):
                        recommender_instance = recommender_class(
                            urm_train=urm,
                            num_training_item_weights_to_save=0,
                        )
                        recommender_instance.RECOMMENDER_NAME = recommender_class.get_recommender_name(
                            cfgan_mode=cfgan_mode,
                            cfgan_mask_type=cfgan_mask_type,
                            cfgan_random_noise_size=random_noise_size,
                        )
                        other_algorithm_list.append(recommender_instance)
                else:
                    recommender_instance = recommender_class(
                        urm_train=urm,
                        num_training_item_weights_to_save=0,
                    )
                    recommender_instance.RECOMMENDER_NAME = recommender_class.get_recommender_name(
                        cfgan_mode=cfgan_mode,
                        cfgan_mask_type=cfgan_mask_type,
                    )
                    other_algorithm_list.append(recommender_instance)
        other_algorithm_list.append(None)

    generate_accuracy_and_beyond_metrics_latex(
        experiments_folder_path=experiments_folder_path,
        export_experiments_folder_path=export_experiments_folder_path,
        num_test_users=num_test_users,
        base_algorithm_list=[],
        knn_similarity_list=[],
        other_algorithm_list=other_algorithm_list,
        accuracy_metrics_list=["PRECISION", "NDCG"],
        beyond_accuracy_metrics_list=["COVERAGE_ITEM"],
        all_metrics_list=["PRECISION", "NDCG", "COVERAGE_ITEM"],
        cutoffs_list=[20],
        icm_names=None,
    )


def _print_accuracy_and_beyond_accuracy_metrics(
    urm: sp.csr_matrix,
    benchmark: CFGANBenchmarks,
    num_test_users: int,
) -> None:
    export_experiments_folder_path = ACCURACY_METRICS_BASELINES_LATEX_DIR.format(
        benchmark=benchmark.value,
    )

    other_algorithm_list: list[Optional[BaseRecommender]] = []

    experiments_folder_path = SINGLE_EXECUTION_EXPERIMENTS_CONCERNS_DIR.format(
        benchmark=benchmark.value,
    )

    recommenders: list[Type[CFGANRecommender]] = [
        CFGANRecommender,
        ClassConditionCFGANRecommender,
        RandomNoiseCFGANRecommender,
    ]

    for cfgan_mode, cfgan_mask_type in reproducibility.cfgan_hyper_parameter_search_settings():
        for recommender_class in recommenders:
            if recommender_class == RandomNoiseCFGANRecommender:
                for random_noise_size in calculate_random_noise_sizes(
                    cfgan_mode=cfgan_mode,
                    urm=urm,
                ):
                    recommender_instance = recommender_class(
                        urm_train=urm,
                        num_training_item_weights_to_save=0,
                    )
                    recommender_instance.RECOMMENDER_NAME = recommender_class.get_recommender_name(
                        cfgan_mode=cfgan_mode,
                        cfgan_mask_type=cfgan_mask_type,
                        cfgan_random_noise_size=random_noise_size,
                    )
                    other_algorithm_list.append(recommender_instance)
            else:
                recommender_instance = recommender_class(
                    urm_train=urm,
                    num_training_item_weights_to_save=0,
                )
                recommender_instance.RECOMMENDER_NAME = recommender_class.get_recommender_name(
                    cfgan_mode=cfgan_mode,
                    cfgan_mask_type=cfgan_mask_type,
                )
                other_algorithm_list.append(recommender_instance)
        other_algorithm_list.append(None)

    generate_accuracy_and_beyond_metrics_latex(
        experiments_folder_path=experiments_folder_path,
        export_experiments_folder_path=export_experiments_folder_path,
        num_test_users=num_test_users,
        base_algorithm_list=[],
        knn_similarity_list=[],
        other_algorithm_list=other_algorithm_list,
        accuracy_metrics_list=reproducibility.ACCURACY_METRICS_LIST,
        beyond_accuracy_metrics_list=reproducibility.BEYOND_ACCURACY_METRICS_LIST,
        all_metrics_list=reproducibility.ALL_METRICS_LIST,
        cutoffs_list=reproducibility.TEST_CUTOFFS,
        icm_names=None,
    )


def print_accuracy_and_beyond_accuracy_metrics():
    for dataset in commons.datasets():
        urm = dataset.urm_train + dataset.urm_validation

        num_test_users: int = np.sum(np.ediff1d(dataset.urm_test.indptr) >= 1)

        # Print all baselines, cfgan, g-cfgan, similarities, and accuracy and beyond accuracy metrics
        _print_accuracy_and_beyond_accuracy_metrics(
            urm=urm,
            benchmark=dataset.benchmark,
            num_test_users=num_test_users,
        )

        # Print article baselines, cfgan, similarities, and accuracy and beyond-accuracy metrics.
        _print_article_accuracy_and_beyond_accuracy_metrics(
            urm=urm,
            benchmark=dataset.benchmark,
            num_test_users=num_test_users,
        )

        logger.info(
            f"Successfully finished exporting accuracy and beyond-accuracy results to LaTeX"
        )


def print_epochs():
    """
    This creates a pandas table with the following structure:

    Dataset|Variant|# of Epochs
    """

    naming_dicts = {
        CFGANMode.ITEM_BASED: "i",
        CFGANMode.USER_BASED: "u",
        CFGANMaskType.ZERO_RECONSTRUCTION_AND_PARTIAL_MASKING: "ZP",
        CFGANMaskType.PARTIAL_MASKING: "PM",
        CFGANMaskType.ZERO_RECONSTRUCTION: "ZR",
    }

    column_dataset = "Dataset"
    column_variant = "Variant"
    column_num_epochs = "# of Epochs"
    column_adjusted_num_epochs = "Adjusted # of Epochs"

    columns = [
        column_dataset,
        column_variant,
        column_num_epochs,
        column_adjusted_num_epochs,
    ]

    for benchmark in commons.BENCHMARKS:
        dataframe_data: dict[int, dict[str, Any]] = {}
        data_index = 0

        # Original number of epochs in Code:
        hyper_parameters_for_benchmark = get_cfgan_code_hyper_parameters(
            benchmark=benchmark,
        )

        mode = naming_dicts[hyper_parameters_for_benchmark.mode]
        mask = naming_dicts[hyper_parameters_for_benchmark.mask_type]

        dataset = benchmark.value
        model = f"{mode}{mask} Original Code"
        epochs = hyper_parameters_for_benchmark.epochs
        adjusted_epochs = hyper_parameters_for_benchmark.epochs * hyper_parameters_for_benchmark.generator_steps

        dataframe_data[data_index] = {
            column_dataset: dataset,
            column_variant: model,
            column_num_epochs: epochs,
            column_adjusted_num_epochs: adjusted_epochs,
        }
        data_index += 1

        # Original number of epochs in Paper:
        hyper_parameters_for_benchmark = get_cfgan_paper_hyper_parameters(
            benchmark=benchmark,
        )

        mode = naming_dicts[hyper_parameters_for_benchmark.mode]
        mask = naming_dicts[hyper_parameters_for_benchmark.mask_type]

        dataset = benchmark.value
        model = f"{mode}{mask} Reference" + r"~\cite{cfgan}"
        epochs = hyper_parameters_for_benchmark.epochs
        adjusted_epochs = hyper_parameters_for_benchmark.epochs * hyper_parameters_for_benchmark.generator_steps

        dataframe_data[data_index] = {
            column_dataset: dataset,
            column_variant: model,
            column_num_epochs: epochs,
            column_adjusted_num_epochs: adjusted_epochs,
        }

        data_index += 1

        # Num of Epochs selected by the early-stopping and without early-stopping.
        for cfgan_mode, cfgan_mask_type in reproducibility.cfgan_hyper_parameter_search_settings():
            hyper_parameters_for_recommender = get_best_hyper_parameters_of_tuned_cfgan_model(
                benchmark=benchmark,
                cfgan_mode=cfgan_mode,
                cfgan_mask_type=cfgan_mask_type,
            )

            mode = naming_dicts[cfgan_mode]
            mask = naming_dicts[cfgan_mask_type]

            dataset = benchmark.value
            model = f"{mode}{mask}"
            epochs = hyper_parameters_for_recommender["epochs"]
            adjusted_epochs = (
                hyper_parameters_for_recommender["epochs"]
                * hyper_parameters_for_recommender["generator_steps"]
            )

            dataframe_data[data_index] = {
                column_dataset: dataset,
                column_variant: model,
                column_num_epochs: epochs,
                column_adjusted_num_epochs: adjusted_epochs,
            }

            data_index += 1

            model = f"{mode}{mask} NO-ES"
            epochs = reproducibility.CFGAN_NUM_EPOCHS
            adjusted_epochs = (
                reproducibility.CFGAN_NUM_EPOCHS
                * hyper_parameters_for_recommender["generator_steps"]
            )

            dataframe_data[data_index] = {
                column_dataset: dataset,
                column_variant: model,
                column_num_epochs: epochs,
                column_adjusted_num_epochs: adjusted_epochs,
            }

            data_index += 1

        # To avoid ellipsis (...) if the values in cells are too big.
        with pd.option_context("max_colwidth", 1000):
            pd.DataFrame.from_records(
                data=list(dataframe_data.values()),
                columns=columns,
            ).astype(
                dtype={
                    column_dataset: pd.StringDtype(),
                    column_variant: pd.StringDtype(),
                    column_num_epochs: np.int32,
                    column_adjusted_num_epochs: np.int32,
                }
            ).set_index(
                column_dataset,
                drop=True,
                inplace=False,
            ).to_latex(
                buf=EPOCHS_COMPARISON_RESULTS_FILE.format(
                    benchmark=benchmark.value
                ),
                index=False,
                escape=False,
                header=True,
                float_format="{:.4f}".format,
                encoding="utf-8",
            )


def print_concerns_results() -> None:
    print_accuracy_and_beyond_accuracy_metrics()
    print_epochs()
