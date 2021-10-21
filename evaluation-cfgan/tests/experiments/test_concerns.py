import os
from pathlib import Path

import pytest

import experiments.concerns as concerns
import experiments.reproducibility as reproducibility
from conferences.cikm.cfgan.our_implementation.constants import CFGANBenchmarks
from experiments.commons import DatasetInterface
from recsys_framework.Utils.conf_dask import configure_dask_cluster, DaskInterface, _DASK_CONF


@pytest.fixture
def dask_interface() -> DaskInterface:
    return configure_dask_cluster()


@pytest.fixture
def dataset_interface(dask_interface: DaskInterface) -> DatasetInterface:
    return DatasetInterface(
        dask_interface=dask_interface,
        benchmarks=[CFGANBenchmarks.CIAO],
        priorities=[1],
    )


def patch_concerns(
    test_folder: str,
    benchmarks: list[CFGANBenchmarks],
) -> None:
    import experiments.concerns
    import experiments.commons

    experiments.concerns.BASE_FOLDER = test_folder
    experiments.concerns.SINGLE_EXECUTION_EXPERIMENTS_CONCERNS_DIR = os.path.join(
        experiments.concerns.BASE_FOLDER,
        "concerns",
        "experiments",
        "",
    )
    experiments.concerns.ARTICLE_ACCURACY_METRICS_BASELINES_LATEX_DIR = os.path.join(
        experiments.concerns.BASE_FOLDER,
        "concerns",
        "article-latex",
        ""
    )
    experiments.concerns.ACCURACY_METRICS_BASELINES_LATEX_DIR = os.path.join(
        experiments.concerns.BASE_FOLDER,
        "concerns",
        "latex",
        ""
    )

    experiments.commons.FOLDERS.add(experiments.concerns.BASE_FOLDER)
    experiments.commons.FOLDERS.add(experiments.concerns.SINGLE_EXECUTION_EXPERIMENTS_CONCERNS_DIR)
    experiments.commons.FOLDERS.add(experiments.concerns.ARTICLE_ACCURACY_METRICS_BASELINES_LATEX_DIR)
    experiments.commons.FOLDERS.add(experiments.concerns.ACCURACY_METRICS_BASELINES_LATEX_DIR)
    experiments.commons.create_necessary_folders(
        benchmarks=benchmarks,
    )

    experiments.concerns.EPOCHS_COMPARISON_RESULTS_FILE = os.path.join(
        experiments.concerns.ACCURACY_METRICS_BASELINES_LATEX_DIR,
        "epochs_comparison_results_file.tex",
    )


def patch_reproducibility(
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

    experiments.commons.FOLDERS.add(experiments.reproducibility.HYPER_PARAMETER_TUNING_EXPERIMENTS_DIR)
    experiments.commons.FOLDERS.add(experiments.reproducibility.ARTICLE_ACCURACY_METRICS_BASELINES_LATEX_DIR)
    experiments.commons.FOLDERS.add(experiments.reproducibility.ACCURACY_METRICS_BASELINES_LATEX_DIR)
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
        ])
    )

    experiments.reproducibility.CFGANRecommenderEarlyStopping.MIN_NUM_EPOCHS = 10
    experiments.reproducibility.CFGANRecommenderEarlyStopping.MAX_NUM_EPOCHS = 20


class TestConcernsExperiments:
    def test_run_reproducibility_experiments_and_print_results(
        self,
        dask_interface: DaskInterface,
        dataset_interface: DatasetInterface,
        tmp_path: Path,
    ):
        # Arrange
        test_folder = str(tmp_path.resolve())
        patch_reproducibility(
            test_folder=test_folder,
            benchmarks=dataset_interface.benchmarks,
        )
        patch_concerns(
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
                job_key=f"patch_reproducibility_worker_{worker_number}",
                job_info={},
                job_priority=50,
                method=patch_reproducibility,
                method_kwargs={
                    "test_folder": test_folder,
                    "benchmarks": dataset_interface.benchmarks,
                }
            )

        # Wait to ensure all workers are patched.
        dask_interface.wait_for_jobs()

        for worker_number in range(_DASK_CONF.num_workers * 2):
            dask_interface.submit_job(
                job_key=f"patch_concerns_worker_{worker_number}",
                job_info={},
                job_priority=50,
                method=patch_concerns,
                method_kwargs={
                    "test_folder": test_folder,
                    "benchmarks": dataset_interface.benchmarks,
                }
            )

        # Wait to ensure all workers are patched.
        dask_interface.wait_for_jobs()

        reproducibility.run_reproducibility_experiments(
            include_baselines=False,
            include_cfgan=True,
            dask_interface=dask_interface,
            dataset_interface=dataset_interface,
        )
        dask_interface.wait_for_jobs()

        # Now run the experiments for real.
        concerns.run_concerns_experiments(
            include_cfgan_with_random_noise=True,
            include_cfgan_with_class_condition=True,
            include_cfgan_without_early_stopping=True,
            dask_interface=dask_interface,
            dataset_interface=dataset_interface,
        )
        dask_interface.wait_for_jobs()
        dask_interface.close()

        concerns.print_concerns_results(
            dataset_interface=dataset_interface,
        )

        # Assert
        expected_results_set = {
            'time_latex_results.tex',
            'all_metrics_latex_results.tex',
            'accuracy_metrics_latex_results.tex',
            'beyond_accuracy_metrics_latex_results.tex',
            'epochs_comparison_results_file.tex',
        }
        expected_article_results_set = {
            'time_latex_results.tex',
            'all_metrics_latex_results.tex',
            'accuracy_metrics_latex_results.tex',
            'beyond_accuracy_metrics_latex_results.tex'
        }
        expected_models_set = set()
        for cfgan_mode, cfgan_mask in reproducibility.cfgan_hyper_parameter_search_settings():
            for recommender in [
                concerns.CFGANRecommender,
                concerns.RandomNoiseCFGANRecommender,
                concerns.ClassConditionCFGANRecommender
            ]:
                if recommender == concerns.RandomNoiseCFGANRecommender:
                    noise_sizes = [331, 662, 1324, ] if cfgan_mode == cfgan_mode.ITEM_BASED else [673, 1347, 2694]
                    for noise_size in noise_sizes:
                        recommender_name = recommender.get_recommender_name(
                            cfgan_mask_type=cfgan_mask,
                            cfgan_mode=cfgan_mode,
                        )
                        expected_models_set.add(f"{recommender_name}noise_size_{noise_size}_best_model_last.zip")
                        expected_models_set.add(f"{recommender_name}noise_size_{noise_size}_best_model_last_models")
                        expected_models_set.add(f"{recommender_name}noise_size_{noise_size}_best_model.zip")
                        expected_models_set.add(f"{recommender_name}noise_size_{noise_size}_best_model_models")
                        expected_models_set.add(f"{recommender_name}noise_size_{noise_size}_SearchSingleCase.txt")
                        expected_models_set.add(f"{recommender_name}noise_size_{noise_size}_metadata.zip")
                else:
                    recommender_name = recommender.get_recommender_name(
                        cfgan_mask_type=cfgan_mask,
                        cfgan_mode=cfgan_mode,
                    )
                    expected_models_set.add(f"{recommender_name}_best_model_last.zip")
                    expected_models_set.add(f"{recommender_name}_best_model_last_models")
                    expected_models_set.add(f"{recommender_name}_best_model.zip")
                    expected_models_set.add(f"{recommender_name}_best_model_models")
                    expected_models_set.add(f"{recommender_name}_SearchSingleCase.txt")
                    expected_models_set.add(f"{recommender_name}_metadata.zip")

        assert set(os.listdir(concerns.SINGLE_EXECUTION_EXPERIMENTS_CONCERNS_DIR)) == expected_models_set
        assert set(os.listdir(concerns.ARTICLE_ACCURACY_METRICS_BASELINES_LATEX_DIR)) == expected_article_results_set
        assert set(os.listdir(concerns.ACCURACY_METRICS_BASELINES_LATEX_DIR)) == expected_results_set
