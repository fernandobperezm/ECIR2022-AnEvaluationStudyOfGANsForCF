import os
import uuid
from typing import Any

import pandas as pd
from distributed import Future, Client
from recsys_framework.Recommenders.DataIO import DataIO
from recsys_framework.Utils.conf_logging import get_logger

import experiments.commons as commons
from conferences.cikm.cfgan.our_implementation.constants import CFGANBenchmarks

logger = get_logger(__name__)


####################################################################################################
####################################################################################################
#                                FOLDERS VARIABLES                            #
####################################################################################################
####################################################################################################
BASE_FOLDER = os.path.join(
    commons.RESULTS_EXPERIMENTS_DIR,
    "replicability",
)
EXPERIMENTS_REPLICATION_RESULTS_DIR = os.path.join(
    BASE_FOLDER,
    "latex",
    "",
)
EXPERIMENTS_REPLICATION_DIR = os.path.join(
    BASE_FOLDER,
    "experiments",
    "{benchmark}",
    ""
)

commons.FOLDERS.add(BASE_FOLDER)
commons.FOLDERS.add(EXPERIMENTS_REPLICATION_RESULTS_DIR)
commons.FOLDERS.add(EXPERIMENTS_REPLICATION_DIR)

ORIGINAL_CODE_EXECUTIONS_ALL_STATISTICS_FILE = os.path.join(
    EXPERIMENTS_REPLICATION_RESULTS_DIR,
    "original_code_executions_all_statistics.tex",
)
ORIGINAL_CODE_EXECUTIONS_ONLY_MEAN_AND_STD_FILE = os.path.join(
    EXPERIMENTS_REPLICATION_RESULTS_DIR,
    "original_code_executions_only_mean_and_std.tex",
)
ORIGINAL_CODE_EXECUTIONS_ONLY_MIN_AND_MAX_FILE = os.path.join(
    EXPERIMENTS_REPLICATION_RESULTS_DIR,
    "original_code_executions_only_min_and_max.tex",
)
ORIGINAL_PUBLISHED_RESULTS_FILE = os.path.join(
    EXPERIMENTS_REPLICATION_RESULTS_DIR,
    "original_published_results.tex",
)
ORIGINAL_PUBLISHED_RESULTS_AND_CODE_EXECUTIONS_FILE = os.path.join(
    EXPERIMENTS_REPLICATION_RESULTS_DIR,
    "original_published_results_and_code_executions.tex",
)
ORIGINAL_PUBLISHED_RESULTS_AND_CODE_EXECUTIONS_AT_X_FILE = os.path.join(
    EXPERIMENTS_REPLICATION_RESULTS_DIR,
    "original_published_results_and_code_executions_{recommendation_length}.tex",
)
ARTICLE_REPLICATION_RESULTS_FILE = os.path.join(
    EXPERIMENTS_REPLICATION_RESULTS_DIR,
    "article_replication_results_table.tex",
)
####################################################################################################
####################################################################################################
#                                RESULTS EXPORTING (LATEX) VARIABLES                            #
####################################################################################################
####################################################################################################
DATASET_COLUMN = "Dataset"
EXECUTION_NUMBER_COLUMN = "Execution"
EPOCHS_COLUMN = "Epochs"
STAT_COLUMN = "Stats"

RECOMMENDATION_LENGTHS = ["@5", "@20"]
ARTICLE_RECOMMENDATION_LENGTH = "@20"

METRICS = ["PREC", "REC", "MRR", "NDCG"]
ARTICLE_METRICS = ["PREC", "NDCG"]

NON_METRICS_COLUMNS = [
    (column, "")
    for column in [DATASET_COLUMN, EXECUTION_NUMBER_COLUMN, EPOCHS_COLUMN]
]

ACCURACY_METRICS_COLUMNS = [
    (n, metric)
    for n in RECOMMENDATION_LENGTHS
    for metric in METRICS
]
ARTICLE_ACCURACY_METRICS_COLUMNS = [
    (n, metric)
    for n in RECOMMENDATION_LENGTHS
    for metric in ARTICLE_METRICS
]

COLUMNS: pd.MultiIndex = pd.MultiIndex.from_tuples(
    tuples=[*NON_METRICS_COLUMNS, *ACCURACY_METRICS_COLUMNS]
)
ARTICLE_COLUMNS: pd.MultiIndex = pd.MultiIndex.from_tuples(
    tuples=[*NON_METRICS_COLUMNS, *ARTICLE_METRICS]
)
PUBLISHED_RESULTS_COLUMNS: pd.MultiIndex = pd.MultiIndex.from_tuples(
    tuples=[(DATASET_COLUMN, ""), *ACCURACY_METRICS_COLUMNS]
)
ORIGINALLY_PUBLISHED_RESULTS: pd.DataFrame = pd.DataFrame.from_dict(
    data={
        1: {
            (DATASET_COLUMN, ""): CFGANBenchmarks.CIAO.value,
            ("@5", "PREC"): 0.072,
            ("@20", "PREC"): 0.045,
            ("@5", "REC"): 0.081,
            ("@20", "REC"): 0.194,
            ("@5", "NDCG"): 0.092,
            ("@20", "NDCG"): 0.124,
            ("@5", "MRR"): 0.154,
            ("@20", "MRR"): 0.167,
        },
        2: {
            (DATASET_COLUMN, ""): CFGANBenchmarks.ML100K.value,
            ("@5", "PREC"): 0.444,
            ("@20", "PREC"): 0.294,
            ("@5", "REC"): 0.152,
            ("@20", "REC"): 0.360,
            ("@5", "NDCG"): 0.476,
            ("@20", "NDCG"): 0.433,
            ("@5", "MRR"): 0.681,
            ("@20", "MRR"): 0.693,
        },
        3: {
            (DATASET_COLUMN, ""): CFGANBenchmarks.ML1M.value,
            ("@5", "PREC"): 0.432,
            ("@20", "PREC"): 0.309,
            ("@5", "REC"): 0.108,
            ("@20", "REC"): 0.272,
            ("@5", "NDCG"): 0.455,
            ("@20", "NDCG"): 0.406,
            ("@5", "MRR"): 0.647,
            ("@20", "MRR"): 0.660,
        },
    },
    orient="index",
    columns=PUBLISHED_RESULTS_COLUMNS
).set_index(
    keys=([DATASET_COLUMN]),
    inplace=False,
).sort_index(
    inplace=False,
    ascending=True,
)


####################################################################################################
####################################################################################################
#                                Experiment 1. Replicability & Numerical Stability                 #
####################################################################################################
####################################################################################################
NUMBER_OF_EXECUTIONS: int = 30


def _run_iteration_original_cfgan_code(
    benchmark: CFGANBenchmarks,
    execution_number: int,
    results_folder: str
) -> None:
    from conferences.cikm.cfgan.our_implementation.original.cfgan import run_original

    all_results_filename = (
        f"CFGAN_original_implementation_execution_{execution_number}.zip"
    )

    if os.path.exists(os.path.join(results_folder, all_results_filename)):
        logger.warning(
            f"Execution number {execution_number} for benchmark {benchmark.value} already exists."
            f"If you wish to redo the execution number, delete the .zip file and try again."
        )
        return

    (
        hyper_parameters, results_dict, _, _,
        _, _, _, _, _, _,
    ) = run_original(
        benchmark=benchmark.value
    )

    data_io = DataIO(
        folder_path=results_folder
    )
    data_io.save_data(
        file_name=all_results_filename,
        data_dict_to_save={
            "hyper_parameters": hyper_parameters,
            "results_dict": results_dict,
        }
    )


def run_replicability_experiments(
    dask_client: Client,
    dask_experiments_futures: list[Future],
    dask_experiments_futures_info: dict[str, dict[str, Any]],
) -> None:
    for priority, benchmark in zip(commons.DATASET_PRIORITIES, commons.BENCHMARKS):
        results_folder = EXPERIMENTS_REPLICATION_DIR.format(
            benchmark=benchmark.value
        )

        for execution_number in range(NUMBER_OF_EXECUTIONS):
            future_execution = dask_client.submit(
                _run_iteration_original_cfgan_code,
                key=(
                    f"_run_iteration_original_cfgan_code"
                    f"|{benchmark.value}"
                    f"|{execution_number}"
                    f"|{uuid.uuid4()}"
                ),
                pure=False,
                priority=priority,
                benchmark=benchmark,
                execution_number=execution_number,
                results_folder=results_folder,
            )
            dask_experiments_futures.append(future_execution)
            dask_experiments_futures_info[future_execution.key] = {
                "benchmark": benchmark.value,
                "execution_number": execution_number,
            }


####################################################################################################
####################################################################################################
#                                Experiment 1. Original Code Execution                             #
####################################################################################################
####################################################################################################
def _print_results_original_cfgan_code() -> None:
    results_dict = dict()
    result_number = 0

    for benchmark in commons.BENCHMARKS:
        results_folder = EXPERIMENTS_REPLICATION_DIR.format(
            benchmark=benchmark.value
        )

        data_io = DataIO(
            folder_path=results_folder
        )

        for execution_number in range(NUMBER_OF_EXECUTIONS):
            results_filename = (
                f"CFGAN_original_implementation_execution_{execution_number}.zip"
            )

            execution_results = data_io.load_data(
                file_name=results_filename
            )
            execution_results_hyper_parameters = execution_results["hyper_parameters"]
            execution_results_dict: dict[Any, Any] = execution_results["results_dict"]

            results_dict[result_number] = {
                (DATASET_COLUMN, ""): benchmark.value,
                (EXECUTION_NUMBER_COLUMN, ""): execution_number,
                (EPOCHS_COLUMN, ""): execution_results_hyper_parameters["epochs"],
                **{
                    (f"@{n}", metric): value
                    for n, metrics in execution_results_dict.items()
                    for metric, value in metrics.items()
                }
            }

            result_number += 1

    # To avoid ellipsis (...) if the values in cells are too big.
    with pd.option_context("max_colwidth", 1000):
        results_dataframe = pd.DataFrame.from_dict(
            data=results_dict,
            orient="index",
            columns=COLUMNS
        )

        described_results_dataframe = results_dataframe.groupby(DATASET_COLUMN).describe()
        described_results_dataframe.to_latex(
            buf=ORIGINAL_CODE_EXECUTIONS_ALL_STATISTICS_FILE,
            index=True,
            header=True,
            float_format="{:.4f}".format,
            encoding="utf-8",
        )

        results_dataframe_with_mean_and_std = ORIGINALLY_PUBLISHED_RESULTS.copy()
        results_dataframe_with_min_and_max = ORIGINALLY_PUBLISHED_RESULTS.copy()
        for n, metric in ACCURACY_METRICS_COLUMNS:
            results_dataframe_with_mean_and_std[(n, metric)] = \
                described_results_dataframe.apply(
                    func=(
                        lambda row, n, metric:
                        fr"${row[(n, metric)]['mean']:.4f} \pm {row[(n, metric)]['std']:.4f}$"
                    ),
                    args=(n, metric),
                    axis=1,
                    result_type="expand",
                )
            results_dataframe_with_min_and_max[(n, metric)] = \
                described_results_dataframe.apply(
                    func=(
                        lambda row, n, metric:
                        fr"${row[(n, metric)]['min']:.4f} - {row[(n, metric)]['max']:.4f}$"
                    ),
                    args=(n, metric),
                    axis=1,
                    result_type="expand",
                )

        results_dataframe_with_mean_and_std.to_latex(
            buf=ORIGINAL_CODE_EXECUTIONS_ONLY_MEAN_AND_STD_FILE,
            index=True,
            header=True,
            escape=False,
            float_format="{:.4f}".format,
            encoding="utf-8",
        )
        results_dataframe_with_min_and_max.to_latex(
            buf=ORIGINAL_CODE_EXECUTIONS_ONLY_MIN_AND_MAX_FILE,
            index=True,
            header=True,
            escape=False,
            float_format="{:.4f}".format,
            encoding="utf-8",
        )

        copy_original_results = ORIGINALLY_PUBLISHED_RESULTS.copy()
        copy_original_results[(STAT_COLUMN, "")] = r"Reported \cite{cfgan}"

        replicated_results_mean_and_std = results_dataframe_with_mean_and_std.copy()
        replicated_results_mean_and_std[(STAT_COLUMN, "")] = r"$\text{Mean} \pm \text{Std}$"

        replicated_results_min_and_max = results_dataframe_with_min_and_max.copy()
        replicated_results_min_and_max[(STAT_COLUMN, "")] = "Range"

        original_results_and_replicated_results = pd.concat(
            objs=[
                copy_original_results,
                replicated_results_mean_and_std,
                replicated_results_min_and_max,
            ],
            axis=0
        )
        original_results_and_replicated_results = original_results_and_replicated_results.reset_index(
            drop=False,
            inplace=False,
        ).set_index(
            keys=([DATASET_COLUMN, STAT_COLUMN]),
            inplace=False,
        ).sort_index(
            inplace=False,
            ascending=True,
        )

        original_results_and_replicated_results.to_latex(
            buf=ORIGINAL_PUBLISHED_RESULTS_AND_CODE_EXECUTIONS_FILE,
            index=True,
            header=True,
            escape=False,
            float_format="{:.4f}".format,
            encoding="utf-8",
        )

        for recommendation_length in RECOMMENDATION_LENGTHS:
            filename = ORIGINAL_PUBLISHED_RESULTS_AND_CODE_EXECUTIONS_AT_X_FILE.format(
                recommendation_length=recommendation_length
            )
            original_results_and_replicated_results[recommendation_length].to_latex(
                buf=filename,
                index=True,
                header=True,
                escape=False,
                float_format="{:.4f}".format,
                encoding="utf-8",
            )

        # Prints the table presented in the article
        original_results_and_replicated_results[ARTICLE_RECOMMENDATION_LENGTH].to_latex(
            buf=ARTICLE_REPLICATION_RESULTS_FILE,
            columns=["PREC", "NDCG"],
            index=True,
            header=True,
            escape=False,
            float_format="{:.4f}".format,
            encoding="utf-8",
        )


def _print_originally_published_results():
    ORIGINALLY_PUBLISHED_RESULTS.to_latex(
        buf=ORIGINAL_PUBLISHED_RESULTS_FILE,
        index=True,
        header=True,
        float_format="{:.4f}".format,
        encoding="utf-8",
    )


def print_replicability_results():
    _print_results_original_cfgan_code()
    _print_originally_published_results()
