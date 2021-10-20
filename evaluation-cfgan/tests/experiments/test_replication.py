import os
from pathlib import Path

import pytest

import experiments.commons as commons
import experiments.replication as replication
from conferences.cikm.cfgan.our_implementation.constants import CFGANBenchmarks
from recsys_framework.Utils.conf_dask import configure_dask_cluster, DaskInterface, _DASK_CONF


@pytest.fixture
def dask_interface() -> DaskInterface:
    return configure_dask_cluster()


def patch_original_hyper_parameters_call() -> None:
    from unittest.mock import MagicMock
    import experiments.replication

    experiments.replication.original_cfgan.parameters.getHyperParams = MagicMock(
        "getHyperParams",
        return_value={
            "epochs": 10,
            "mode": "itemBased",
            "hiddenDim_G": 250,
            "hiddenDim_D": 50,
            "reg_G": 0.001,

            "reg_D": 0.001,
            "lr_G": 0.0001,
            "lr_D": 0.0001,
            "batchSize_G": 128,
            "batchSize_D": 128,

            "opt_G": "adam",
            "opt_D": "adam",
            "hiddenLayer_G": 1,
            "hiddenLayer_D": 4,
            "step_G": 1,
            "step_D": 1,

            "scheme": "ZR",
            "ZR_ratio": 40,
            "ZP_ratio": "None",
            "ZR_coefficient": 0.1
        }
    )


class TestReplicationExperiment:
    TEST_NUM_REPLICATIONS = 2

    def test_run_replicability_experiments_and_print_results(
        self,
        dask_interface: DaskInterface,
        tmp_path: Path,
    ):
        # Arrange
        replication.BASE_FOLDER = str(tmp_path.resolve())
        replication.EXPERIMENTS_REPLICATION_DIR = os.path.join(
            replication.BASE_FOLDER,
            "experiments",
            "",
        )
        replication.EXPERIMENTS_REPLICATION_RESULTS_DIR = os.path.join(
            replication.BASE_FOLDER,
            "latex",
            ""
        )
        replication.ORIGINAL_CODE_EXECUTIONS_ALL_STATISTICS_FILE = os.path.join(
            replication.EXPERIMENTS_REPLICATION_RESULTS_DIR,
            "original_code_executions_all_statistics.tex",
        )
        replication.ORIGINAL_CODE_EXECUTIONS_ONLY_MEAN_AND_STD_FILE = os.path.join(
            replication.EXPERIMENTS_REPLICATION_RESULTS_DIR,
            "original_code_executions_only_mean_and_std.tex",
        )
        replication.ORIGINAL_CODE_EXECUTIONS_ONLY_MIN_AND_MAX_FILE = os.path.join(
            replication.EXPERIMENTS_REPLICATION_RESULTS_DIR,
            "original_code_executions_only_min_and_max.tex",
        )
        replication.ORIGINAL_PUBLISHED_RESULTS_FILE = os.path.join(
            replication.EXPERIMENTS_REPLICATION_RESULTS_DIR,
            "original_published_results.tex",
        )
        replication.ORIGINAL_PUBLISHED_RESULTS_AND_CODE_EXECUTIONS_FILE = os.path.join(
            replication.EXPERIMENTS_REPLICATION_RESULTS_DIR,
            "original_published_results_and_code_executions.tex",
        )
        replication.ORIGINAL_PUBLISHED_RESULTS_AND_CODE_EXECUTIONS_AT_X_FILE = os.path.join(
            replication.EXPERIMENTS_REPLICATION_RESULTS_DIR,
            "original_published_results_and_code_executions_{recommendation_length}.tex",
        )
        replication.ARTICLE_REPLICATION_RESULTS_FILE = os.path.join(
            replication.EXPERIMENTS_REPLICATION_RESULTS_DIR,
            "article_replication_results_table.tex",
        )
        replication.NUMBER_OF_EXECUTIONS = self.TEST_NUM_REPLICATIONS

        commons.BENCHMARKS = [CFGANBenchmarks.CIAO]
        commons.DATASET_PRIORITIES = [1]
        commons.FOLDERS.add(replication.EXPERIMENTS_REPLICATION_DIR)
        commons.FOLDERS.add(replication.EXPERIMENTS_REPLICATION_RESULTS_DIR)
        commons.create_necessary_folders()

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
                method=patch_original_hyper_parameters_call,
                method_kwargs={}
            )
        # Wait to ensure all workers are patched.
        dask_interface.wait_for_jobs()

        # Now run the experiments for real.
        replication.run_replicability_experiments(
            dask_interface=dask_interface,
        )
        dask_interface.wait_for_jobs()
        dask_interface.close()

        replication.print_replicability_results()

        # Assert
        assert set(os.listdir(replication.EXPERIMENTS_REPLICATION_DIR)) == set(
            f"CFGAN_original_implementation_execution_{execution_number}.zip"
            for execution_number in range(self.TEST_NUM_REPLICATIONS)
        )
        assert set(os.listdir(replication.EXPERIMENTS_REPLICATION_RESULTS_DIR)) == {
            "original_code_executions_all_statistics.tex",
            "original_code_executions_only_mean_and_std.tex",
            "original_code_executions_only_min_and_max.tex",
            "original_published_results.tex",
            "original_published_results_and_code_executions.tex",
            "article_replication_results_table.tex",
            *[
                f"original_published_results_and_code_executions_{recommendation_length}.tex"
                for recommendation_length in replication.RECOMMENDATION_LENGTHS
            ]
        }


