import os
from collections import Generator

import attr
import scipy.sparse as sp

from conferences.cikm.cfgan.our_implementation.constants import CFGANBenchmarks, CFGANDatasetMode
from conferences.cikm.cfgan.our_implementation.data.CFGANDatasetReader import CFGANDatasetReader


@attr.s(frozen=True, kw_only=True)
class Dataset:
    benchmark: CFGANBenchmarks = attr.ib()
    urm_train: sp.csr_matrix = attr.ib()
    urm_validation: sp.csr_matrix = attr.ib()
    urm_test: sp.csr_matrix = attr.ib()
    priority: int = attr.ib()


def set_seed(
    seed: int
) -> None:
    # This should be called by each dask process, not outside.
    import random
    import tensorflow as tf
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def datasets() -> Generator[
    Dataset,
    None,
    None
]:
    """ Generator that returns the dataset and train, validation, and test splits.

    Returns
    -------
    A Dataset instance.
    """
    for priority, benchmark in zip(
        DATASET_PRIORITIES,
        BENCHMARKS
    ):
        dataset_reader = CFGANDatasetReader(
            dataset=benchmark,
            mode=CFGANDatasetMode.ORIGINAL,
        )
        dataset_reader.load_data()

        yield Dataset(
            benchmark=benchmark,
            urm_train=dataset_reader.URM_DICT[dataset_reader.NAME_URM_TRAIN],
            urm_validation=dataset_reader.URM_DICT[dataset_reader.NAME_URM_VALIDATION],
            urm_test=dataset_reader.URM_DICT[dataset_reader.NAME_URM_TEST],
            priority=priority,
        )


####
DATASET_PRIORITIES = [
    30,
    20,
    10,
]
BENCHMARKS = [
    CFGANBenchmarks.ML1M,
    CFGANBenchmarks.ML100K,
    CFGANBenchmarks.CIAO,
]

RESULTS_EXPERIMENTS_DIR = os.path.join(
    ".",
    "result_experiments",
    ""
)

# Each module calls common.FOLDERS.add(<folder_name>) on this variable so they make aware the folder-creator function
# that their folders need to be created.
FOLDERS: set[str] = {
    RESULTS_EXPERIMENTS_DIR,
}


# Should be called from main.py
def create_necessary_folders():
    for benchmark in BENCHMARKS:
        for folder in FOLDERS:
            os.makedirs(
                name=folder.format(
                    benchmark=benchmark.value,
                ),
                exist_ok=True,
            )
