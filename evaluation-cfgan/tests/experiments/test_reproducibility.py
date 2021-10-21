import os
from pathlib import Path

import pytest

import experiments.commons as commons
import experiments.reproducibility as reproducibility
from conferences.cikm.cfgan.our_implementation.constants import CFGANBenchmarks
from recsys_framework.Utils.conf_dask import configure_dask_cluster, DaskInterface, _DASK_CONF
import recsys_framework.Recommenders as recommenders


@pytest.fixture
def dask_interface() -> DaskInterface:
    return configure_dask_cluster()


@pytest.fixture
def dataset_interface(dask_interface: DaskInterface) -> commons.DatasetInterface:
    return commons.DatasetInterface(
        dask_interface=dask_interface,
        benchmarks=[CFGANBenchmarks.CIAO],
        priorities=[1],
    )


def patch_cfgan_recommender_early_stopping(
    test_folder: str,
    benchmarks: list[CFGANBenchmarks],
) -> None:
    from unittest.mock import MagicMock
    import experiments.reproducibility
    import experiments.commons

    experiments.reproducibility.BASE_FOLDER = test_folder
    experiments.reproducibility.HYPER_PARAMETER_TUNING_EXPERIMENTS_DIR = os.path.join(
        experiments.reproducibility.BASE_FOLDER,
        "experiments",
        "",
    )
    experiments.reproducibility.ARTICLE_ACCURACY_METRICS_BASELINES_LATEX_DIR = os.path.join(
        experiments.reproducibility.BASE_FOLDER,
        "article-latex",
        ""
    )
    experiments.reproducibility.ACCURACY_METRICS_BASELINES_LATEX_DIR = os.path.join(
        experiments.reproducibility.BASE_FOLDER,
        "latex",
        ""
    )

    experiments.commons.FOLDERS.add(reproducibility.HYPER_PARAMETER_TUNING_EXPERIMENTS_DIR)
    experiments.commons.FOLDERS.add(reproducibility.ARTICLE_ACCURACY_METRICS_BASELINES_LATEX_DIR)
    experiments.commons.FOLDERS.add(reproducibility.ACCURACY_METRICS_BASELINES_LATEX_DIR)
    experiments.commons.create_necessary_folders(
        benchmarks=benchmarks,
    )

    experiments.reproducibility.ALGORITHM_NAME = "CFGAN"
    experiments.reproducibility.CONFERENCE_NAME = "CIKM"
    experiments.reproducibility.VALIDATION_CUTOFFS = [10]
    experiments.reproducibility.TEST_CUTOFFS = [5, 20]
    experiments.reproducibility.METRIC_TO_OPTIMIZE = "NDCG"
    experiments.reproducibility.NUM_CASES = 3
    experiments.reproducibility.NUM_RANDOM_STARTS = 1
    experiments.reproducibility.REPRODUCIBILITY_SEED = 1234567890
    experiments.reproducibility.CFGAN_NUM_EPOCHS = 10
    experiments.reproducibility.KNN_SIMILARITY_LIST = [
        "asymmetric",
    ]
    experiments.reproducibility.ACCURACY_METRICS_LIST = [
        "PRECISION",
        "RECALL",
        "MRR",
        "NDCG",
    ]
    experiments.reproducibility.BEYOND_ACCURACY_METRICS_LIST = [
        "NOVELTY",
        "COVERAGE_ITEM",
        "DIVERSITY_MEAN_INTER_LIST",
        "DIVERSITY_GINI",
    ]
    experiments.reproducibility.ALL_METRICS_LIST = [
        *experiments.reproducibility.ACCURACY_METRICS_LIST,
        *experiments.reproducibility.BEYOND_ACCURACY_METRICS_LIST,
    ]
    experiments.reproducibility.ARTICLE_BASELINES = [
        recommenders.Random,
        recommenders.TopPop,
        recommenders.RP3betaRecommender,
        recommenders.PureSVDRecommender,
        recommenders.SLIMElasticNetRecommender,
        recommenders.MatrixFactorization_BPR_Cython,
        recommenders.EASE_R_Recommender,
    ]

    experiments.reproducibility.cfgan_hyper_parameter_search_settings = MagicMock(
        "cfgan_hyper_parameter_search_settings",
        return_value=([
            (
                experiments.reproducibility.CFGANMode.ITEM_BASED,
                experiments.reproducibility.CFGANMaskType.PARTIAL_MASKING,
            ),
            (
                experiments.reproducibility.CFGANMode.USER_BASED,
                experiments.reproducibility.CFGANMaskType.ZERO_RECONSTRUCTION,
            ),
            (
                experiments.reproducibility.CFGANMode.ITEM_BASED,
                experiments.reproducibility.CFGANMaskType.ZERO_RECONSTRUCTION_AND_PARTIAL_MASKING,
            ),
        ])
    )

    experiments.reproducibility.CFGANRecommenderEarlyStopping.MIN_NUM_EPOCHS = 10
    experiments.reproducibility.CFGANRecommenderEarlyStopping.MAX_NUM_EPOCHS = 20


class TestReproducibilityExperiments:
    def test_run_reproducibility_experiments_and_print_results(
        self,
        dask_interface: DaskInterface,
        dataset_interface: commons.DatasetInterface,
        tmp_path: Path,
    ):
        # Arrange
        test_folder = str(tmp_path.resolve())
        patch_cfgan_recommender_early_stopping(
            test_folder=test_folder,
            benchmarks=dataset_interface.benchmarks,
        )

        # Act

        # Need to patch the call that receives the hyper-parameters for the experiment.
        # This must be done on each worker, therefore, we submit a job that patches each worker.
        # This is done twice the number of workers to *quasi-ensure* that all workers will
        # run the patched version.
        for worker_number in range(_DASK_CONF.num_workers * 2):
            dask_interface.submit_job(
                job_key=f"patch_worker_{worker_number}",
                job_info={},
                job_priority=50,
                method=patch_cfgan_recommender_early_stopping,
                method_kwargs={
                    "test_folder": test_folder,
                    "benchmarks": dataset_interface.benchmarks,
                }
            )
        # Wait to ensure all workers are patched.
        dask_interface.wait_for_jobs()

        # Now run the experiments for real.
        reproducibility.run_reproducibility_experiments(
            include_baselines=True,
            include_cfgan=True,
            dask_interface=dask_interface,
            dataset_interface=dataset_interface,
        )
        dask_interface.wait_for_jobs()
        dask_interface.close()

        reproducibility.print_reproducibility_results(
            dataset_interface=dataset_interface,
        )

        # Assert
        expected_results_set = {
            'time_latex_results.tex',
            'all_metrics_latex_results.tex',
            'accuracy_metrics_latex_results.tex',
            'beyond_accuracy_metrics_latex_results.tex'
        }
        expected_models_set = set()
        for recommender in reproducibility.ARTICLE_BASELINES:

            if recommender in [recommenders.ItemKNNCFRecommender, recommenders.UserKNNCFRecommender]:
                for similarity in reproducibility.KNN_SIMILARITY_LIST:
                    expected_models_set.add(f"{recommender.RECOMMENDER_NAME}_{similarity}_best_model_last.zip")
                    expected_models_set.add(f"{recommender.RECOMMENDER_NAME}_{similarity}_best_model.zip")
                    expected_models_set.add(f"{recommender.RECOMMENDER_NAME}_{similarity}_SearchSingleCase.txt")
                    expected_models_set.add(f"{recommender.RECOMMENDER_NAME}_{similarity}_metadata.zip")

            elif recommender in [recommenders.Random, recommenders.TopPop]:
                expected_models_set.add(f"{recommender.RECOMMENDER_NAME}_best_model_last.zip")
                expected_models_set.add(f"{recommender.RECOMMENDER_NAME}_best_model.zip")
                expected_models_set.add(f"{recommender.RECOMMENDER_NAME}_SearchSingleCase.txt")
                expected_models_set.add(f"{recommender.RECOMMENDER_NAME}_metadata.zip")
            else:
                expected_models_set.add(f"{recommender.RECOMMENDER_NAME}_best_model_last.zip")
                expected_models_set.add(f"{recommender.RECOMMENDER_NAME}_best_model.zip")
                expected_models_set.add(f"{recommender.RECOMMENDER_NAME}_SearchBayesianSkopt.txt")
                expected_models_set.add(f"{recommender.RECOMMENDER_NAME}_metadata.zip")

        for cfgan_mode, cfgan_mask in reproducibility.cfgan_hyper_parameter_search_settings():
            recommender_name = reproducibility.CFGANRecommenderEarlyStopping.get_recommender_name(
                cfgan_mask_type=cfgan_mask,
                cfgan_mode=cfgan_mode,
            )
            expected_models_set.add(f"{recommender_name}_best_model_last.zip")
            expected_models_set.add(f"{recommender_name}_best_model_last_models")
            expected_models_set.add(f"{recommender_name}_best_model.zip")
            expected_models_set.add(f"{recommender_name}_best_model_models")
            expected_models_set.add(f"{recommender_name}_SearchBayesianSkopt.txt")
            expected_models_set.add(f"{recommender_name}_metadata.zip")

        assert set(os.listdir(reproducibility.HYPER_PARAMETER_TUNING_EXPERIMENTS_DIR)) == expected_models_set
        assert set(os.listdir(reproducibility.ARTICLE_ACCURACY_METRICS_BASELINES_LATEX_DIR)) == expected_results_set
        assert set(os.listdir(reproducibility.ACCURACY_METRICS_BASELINES_LATEX_DIR)) == expected_results_set
