#!/usr/bin/env python3
from __future__ import annotations

from typing import Any

from distributed import Future, as_completed
from recsys_framework.Utils.conf_dask import configure_dask_cluster, close_dask_client
from recsys_framework.Utils.conf_logging import get_logger
from tap import Tap

from experiments.commons import create_necessary_folders
from experiments.concerns import run_concerns_experiments, print_concerns_results
from experiments.replication import run_replicability_experiments, print_replicability_results
from experiments.reproducibility import run_reproducibility_experiments, print_reproducibility_results


class ConsoleArguments(Tap):
    run_replicability: bool = False
    """Run a fixed number of executions of the original CFGAN implementation."""

    run_reproducibility: bool = False
    """Run Hyper-parameter tuning of recommenders on the Ciao, ML100K, and ML1M datasets. Which recommenders are 
    tuned depend on the presence of the options --include_baselines and --include_cfgan.
    """

    include_baselines: bool = False
    """Include baselines in the hyper-parameter tuning"""

    include_cfgan: bool = False
    """Include CFGAN with early-stopping in the hyper-parameter tuning"""

    run_concerns: bool = False
    """Runs modified CFGAN models (one for each variant) using the best hyper-parameter set found in the 
    reproducibility study. Which modifications are included depend on the presence of the options 
    --include_cfgan_with_random_noise and --include_cfgan_with_class_condition --include_cfgan_without_early_stopping"""

    include_cfgan_without_early_stopping: bool = False
    """Include CFGAN without early-stopping in the concerns experiments."""

    include_cfgan_with_random_noise: bool = False
    """Include CFGAN with random noise in the concerns experiments."""

    include_cfgan_with_class_condition: bool = False
    """Include CFGAN using user/item classes as condition vectors in the concerns experiments."""

    print_replicability_results: bool = False
    """Print LaTeX tables containing the accuracy metrics of 'number_of_executions_original_code' number of 
    executions of the original source code."""

    print_reproducibility_results: bool = False
    """Print LaTeX tables containing the accuracy and beyond accuracy metrics of the hyper-parameter tuned 
    recommenders."""

    print_concerns_results: bool = False
    """Print LaTeX tables containing the accuracy and beyond accuracy metrics of the modified CFGAN models."""


####################################################################################################
####################################################################################################
#                                            MAIN                                                  #
####################################################################################################
####################################################################################################
def wait_for_experiments_to_finish():
    future: Future
    for future in as_completed(dask_experiments_futures):
        experiment_info = dask_experiments_futures_info[future.key]

        try:
            future.result()
            logger.info(
                f"Successfully finished this experiment: {experiment_info}"
            )
        except:
            logger.exception(
                f"The following experiment failed: {experiment_info}"
            )


if __name__ == '__main__':
    input_flags = ConsoleArguments().parse_args()

    logger = get_logger(__name__)

    create_necessary_folders()

    DASK_CLIENT = configure_dask_cluster()

    dask_experiments_futures: list[Future] = []
    dask_experiments_futures_info: dict[str, dict[str, Any]] = dict()

    if input_flags.run_replicability:
        run_replicability_experiments(
            dask_client=DASK_CLIENT,
            dask_experiments_futures=dask_experiments_futures,
            dask_experiments_futures_info=dask_experiments_futures_info,
        )

    if input_flags.run_reproducibility:
        run_reproducibility_experiments(
            include_baselines=input_flags.include_baselines,
            include_cfgan=input_flags.include_cfgan,
            dask_client=DASK_CLIENT,
            dask_experiments_futures=dask_experiments_futures,
            dask_experiments_futures_info=dask_experiments_futures_info,
        )

    if input_flags.run_concerns:
        run_concerns_experiments(
            include_cfgan_with_random_noise=input_flags.include_cfgan_with_random_noise,
            include_cfgan_with_class_condition=input_flags.include_cfgan_with_class_condition,
            include_cfgan_without_early_stopping=input_flags.include_cfgan_without_early_stopping,
            dask_client=DASK_CLIENT,
            dask_experiments_futures=dask_experiments_futures,
            dask_experiments_futures_info=dask_experiments_futures_info,
        )

    wait_for_experiments_to_finish()

    if input_flags.print_replicability_results:
        print_replicability_results()

    if input_flags.print_reproducibility_results:
        print_reproducibility_results()

    if input_flags.print_concerns_results:
        print_concerns_results()

    close_dask_client(
        client=DASK_CLIENT,
    )
