#!/usr/bin/env python3
from __future__ import annotations

from tap import Tap

from conferences.cikm.cfgan.our_implementation.constants import CFGANBenchmarks
from experiments.commons import create_necessary_folders, DatasetInterface
from experiments.concerns import run_concerns_experiments, print_concerns_results
from experiments.replication import run_replicability_experiments, print_replicability_results
from experiments.reproducibility import run_reproducibility_experiments, print_reproducibility_results
from recsys_framework.Utils.conf_dask import configure_dask_cluster
from recsys_framework.Utils.conf_logging import get_logger


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
if __name__ == '__main__':
    input_flags = ConsoleArguments().parse_args()

    logger = get_logger(__name__)

    dask_interface = configure_dask_cluster()

    dataset_interface = DatasetInterface(
        dask_interface=dask_interface,
        priorities=[
            30,
            20,
            10,
        ],
        benchmarks=[
            CFGANBenchmarks.ML1M,
            CFGANBenchmarks.ML100K,
            CFGANBenchmarks.CIAO,
        ],
    )

    create_necessary_folders(
        benchmarks=dataset_interface.benchmarks
    )

    if input_flags.run_replicability:
        run_replicability_experiments(
            dask_interface=dask_interface,
            dataset_interface=dataset_interface,
        )

    if input_flags.run_reproducibility:
        run_reproducibility_experiments(
            include_baselines=input_flags.include_baselines,
            include_cfgan=input_flags.include_cfgan,
            dask_interface=dask_interface,
            dataset_interface=dataset_interface,
        )

    if input_flags.run_concerns:
        run_concerns_experiments(
            include_cfgan_with_random_noise=input_flags.include_cfgan_with_random_noise,
            include_cfgan_with_class_condition=input_flags.include_cfgan_with_class_condition,
            include_cfgan_without_early_stopping=input_flags.include_cfgan_without_early_stopping,
            dask_interface=dask_interface,
            dataset_interface=dataset_interface,
        )

    dask_interface.wait_for_jobs()

    if input_flags.print_replicability_results:
        print_replicability_results(
            dataset_interface=dataset_interface,
        )

    if input_flags.print_reproducibility_results:
        print_reproducibility_results(
            dataset_interface=dataset_interface,
        )

    if input_flags.print_concerns_results:
        print_concerns_results(
            dataset_interface=dataset_interface,
        )

    dask_interface.close()
