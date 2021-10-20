import os
import socket
from typing import Optional

import distributed
import psutil
import toml
import attr
from dask.distributed import Client, LocalCluster
from distributed import Scheduler

from recsys_framework.Utils.conf_logging import get_logger

logger = get_logger(__name__)


@attr.s(frozen=True, kw_only=True)
class DaskConfig:
    use_processes: bool = attr.ib()
    dashboard_address: str = attr.ib()
    scheduler_port: int = attr.ib()
    num_workers: int = attr.ib()
    threads_per_worker: int = attr.ib()
    memory_limit: int = attr.ib()


def _load_logger_config() -> DaskConfig:
    with open(os.path.join(os.getcwd(), "pyproject.toml"), "r") as project_file:
        config = toml.load(
            f=project_file
        )
    dask_config = DaskConfig(**config["dask"])
    return dask_config


_DASK_CONF = _load_logger_config()


def _is_scheduler_alive(
    scheduler_address: str,
    scheduler_port: int,
) -> bool:
    with socket.socket(
        family=socket.AF_INET,
        type=socket.SOCK_STREAM
    ) as a_socket:
        try:
            # Avoids exception on binding if the socket was destroyed just before creating it again.
            # See: https://docs.python.org/3/library/socket.html#example
            a_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            a_socket.bind(
                (scheduler_address, scheduler_port)
            )
            return False
        except OSError:
            return True


def configure_dask_cluster() -> Client:
    use_processes = _DASK_CONF.use_processes
    dashboard_address = _DASK_CONF.dashboard_address
    scheduler_port = _DASK_CONF.scheduler_port
    num_workers = _DASK_CONF.num_workers
    threads_per_worker = _DASK_CONF.threads_per_worker
    memory_limit = _DASK_CONF.memory_limit

    machine_memory = psutil.virtual_memory().total
    # Reserve 1GB (2 ** 30) so the machine does not hang or the process gets killed because it
    #  consumes all the available memory.
    machine_memory -= 2 ** 30
    cpu_count = psutil.cpu_count()

    # Recommended by Dask docs:
    #  https://docs.dask.org/en/latest/dataframe-best-practices.html#repartition-to-reduce-overhead
    partition_memory = 100 * 2 ** 20

    # Used to detect if the scheduler is on and to automatically connect to it.
    scheduler_address = "127.0.0.1"

    if use_processes:
        # 4  # Default value for a 16vCPU machine.
        n_workers = num_workers

        # 4  # Default value for a 16vCPU machine. Cores:= n_workers * threads_per_worker
        threads_per_worker = threads_per_worker

        # 0 for no limit.  14 * 2 ** 30  # machine_memory / n_workers
        # Each worker will have this memory limit.
        memory_limit = memory_limit
    else:
        # Default value in Dask's source code
        n_workers = 1

        # cpu_count  # Default value in Dask's source code
        threads_per_worker = cpu_count

        # machine_memory  # Default value in Dask's source code
        memory_limit = machine_memory

    if _is_scheduler_alive(
        scheduler_address=scheduler_address,
        scheduler_port=scheduler_port
    ):
        logger.info(
            f"Connecting to already-created scheduler at {scheduler_address}:{scheduler_port}"
        )
        return Client(
            address=f"{scheduler_address}:{scheduler_port}"
        )
    else:
        logger.info(
            f"Dask Client and Cluster information"
            f"\n* Scheduler Port={scheduler_port}"
            f"\n* Dashboard Address={dashboard_address}"
            f"\n* CPU Count={cpu_count}"
            f"\n* Workers Count={n_workers}"
            f"\n* Threads per Worker={threads_per_worker}"
            f"\n* Partition Memory={partition_memory / 2 ** 20} MiB"
            f"\n* Installed memory={(machine_memory + 2 ** 30) / 2 ** 30:.2f} GB"
            f"\n* Whole Cluster Usable Memory={machine_memory / 2 ** 30:.2f} GB"
            f"\n* Worker Usable Memory={memory_limit / 2 ** 30:.2f} GB"
            f"\n* Use Processes={use_processes}"
        )

        logger.info(
            f"Creating new Dask Local Cluster Client"
        )
        client = Client(
            LocalCluster(
                n_workers=n_workers,
                threads_per_worker=threads_per_worker,
                memory_limit=memory_limit,
                processes=use_processes,
                dashboard_address=dashboard_address,
                scheduler_port=scheduler_port,
            )
        )
        return client


def _number_of_tasks_in_scheduler(dask_scheduler: Scheduler) -> int:
    num_ready_tasks = 0
    ready_task_state = "memory"
    v: distributed.scheduler.TaskState
    for v in dask_scheduler.tasks.values():
        if v.state == ready_task_state:
            num_ready_tasks += 1

    logger.debug(
        f"len(dask_scheduler.tasks)={len(dask_scheduler.tasks)}"
        f"\n* dask_scheduler.n_tasks={dask_scheduler.n_tasks}"
        f"\n* num_ready_tasks={num_ready_tasks}"
    )
    return len(dask_scheduler.tasks)


def close_dask_client(client: Client) -> None:
    num_tasks = client.run_on_scheduler(
        _number_of_tasks_in_scheduler
    )

    if num_tasks > 0:
        logger.info(
            f"Will not shutdown Dask's scheduler due to {num_tasks} running at the moment."
        )
        return

    logger.info(
        f"Shutting down dask client. No more pending tasks."
    )
    client.close()
