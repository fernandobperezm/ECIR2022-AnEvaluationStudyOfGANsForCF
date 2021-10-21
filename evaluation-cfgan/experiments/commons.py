import os

import attr
import scipy.sparse as sp
from distributed import Future

from conferences.cikm.cfgan.our_implementation.constants import CFGANBenchmarks, CFGANDatasetMode
from conferences.cikm.cfgan.our_implementation.dataset_reader.CFGANDatasetReader import CFGANDatasetReader
from recsys_framework.Utils.conf_dask import DaskInterface


@attr.s(frozen=True, kw_only=True)
class Dataset:
    benchmark: CFGANBenchmarks = attr.ib()
    priority: int = attr.ib()
    num_rows: int = attr.ib()
    num_cols: int = attr.ib()
    urm_train: sp.csr_matrix = attr.ib()
    urm_validation: sp.csr_matrix = attr.ib()
    urm_test: sp.csr_matrix = attr.ib()


@attr.s(frozen=True, kw_only=True)
class ScatteredDataset:
    benchmark: CFGANBenchmarks = attr.ib()
    priority: int = attr.ib()
    num_rows: int = attr.ib()
    num_cols: int = attr.ib()
    urm_train: Future = attr.ib()
    urm_validation: Future = attr.ib()
    urm_test: Future = attr.ib()


class DatasetInterface:
    NAME_URM_TRAIN = "URM_TRAIN"
    NAME_URM_VALIDATION = "URM_VALIDATION"
    NAME_URM_TEST = "URM_TEST"

    def __init__(
        self,
        dask_interface: DaskInterface,
        priorities: list[int],
        benchmarks: list[CFGANBenchmarks]
    ):
        self.priorities = priorities
        self.benchmarks = benchmarks
        self._dask_interface = dask_interface
        self._datasets: list[Dataset] = []
        self._scattered_datasets: list[ScatteredDataset] = []

    @property
    def datasets(self) -> list[Dataset]:
        if len(self._datasets) == 0:
            self._load_datasets()

        return self._datasets

    @property
    def scattered_datasets(self):
        if len(self._scattered_datasets) == 0:
            self._load_datasets()

        return self._scattered_datasets

    def _load_datasets(self) -> None:
        for priority, benchmark in zip(
            self.priorities,
            self.benchmarks
        ):
            dataset_reader = CFGANDatasetReader(
                dataset=benchmark,
                mode=CFGANDatasetMode.ORIGINAL,
            )
            dataset_reader.load_data()

            self._datasets.append(
                Dataset(
                    benchmark=benchmark,
                    priority=priority,
                    urm_train=dataset_reader.URM_DICT[dataset_reader.NAME_URM_TRAIN].copy(),
                    urm_validation=dataset_reader.URM_DICT[dataset_reader.NAME_URM_VALIDATION].copy(),
                    urm_test=dataset_reader.URM_DICT[dataset_reader.NAME_URM_TEST].copy(),
                    num_rows=dataset_reader.URM_DICT[dataset_reader.NAME_URM_TRAIN].shape[0],
                    num_cols=dataset_reader.URM_DICT[dataset_reader.NAME_URM_TRAIN].shape[1],
                )
            )

            self._scattered_datasets.append(
                ScatteredDataset(
                    benchmark=benchmark,
                    priority=priority,
                    urm_train=self._dask_interface.scatter_data(
                        data=dataset_reader.URM_DICT[dataset_reader.NAME_URM_TRAIN].copy(),
                    ),
                    urm_validation=self._dask_interface.scatter_data(
                        data=dataset_reader.URM_DICT[dataset_reader.NAME_URM_VALIDATION].copy(),
                    ),
                    urm_test=self._dask_interface.scatter_data(
                        data=dataset_reader.URM_DICT[dataset_reader.NAME_URM_TEST].copy(),
                    ),
                    num_rows=dataset_reader.URM_DICT[dataset_reader.NAME_URM_TRAIN].shape[0],
                    num_cols=dataset_reader.URM_DICT[dataset_reader.NAME_URM_TRAIN].shape[1],
                )
            )


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
def create_necessary_folders(
    benchmarks: list[CFGANBenchmarks],
):
    for benchmark in benchmarks:
        for folder in FOLDERS:
            os.makedirs(
                name=folder.format(
                    benchmark=benchmark.value,
                ),
                exist_ok=True,
            )
