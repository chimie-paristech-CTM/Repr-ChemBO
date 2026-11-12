import argparse
import os

from config_grid import ConfigGrid

from bo.BOrunner import BOrunner
from utils.context_manager import ContextManager
from utils.merger import Merger

import subprocess
import sys

def run_one(context: dict, continue_on_error=False) -> None:
    '''
    Only current context: a point in the config_grid, not care about global config
    '''
    
    context_manager = ContextManager(context)
    try:
        bo_runner = BOrunner(config=context, context_manager=context_manager)
        bo_runner.run()
    except BaseException as error:
        context_manager.save_failed_context(error)
        if not continue_on_error:
            raise

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--job-id",
        type=int,
        default=None,
        help="Run only one configuration locally.",
    )

    parser.add_argument(
        "--count",
        action="store_true",
        help="Print number of configurations.",
    )
    
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge all runs and analyze.",
    )
    
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run configurations across Slurm worker processes.",
    )
    
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=None)

    args = parser.parse_args()

    config_grid = ConfigGrid.from_json(args.config)

    if args.count:
        print(len(config_grid))
        return
    
    if args.merge:
        m = Merger()
        m.merge()
        m.analyze([
            "dataset.name",
            "fp.name",
            "hyperprior",
            "normalize_feature",
            "fp_level",
        ])
        return
    
    if args.parallel:
        rank = int(os.environ["SLURM_PROCID"])
        workers = int(os.environ["SLURM_NTASKS"])
        failed = False

        start = args.start
        stop = len(config_grid) if args.stop is None else args.stop

        if not 0 <= start < stop <= len(config_grid):
            parser.error("Require 0 <= start < stop <= number of configurations")

        for job_id in range(start + rank, stop, workers):
            print(f"Worker {rank}: running configuration {job_id}", flush=True)

            result = subprocess.run([
                sys.executable,
                "-u",
                os.path.abspath(__file__),
                "--config", args.config,
                "--job-id", str(job_id),
            ])

            if result.returncode != 0:
                print(f"Configuration {job_id} failed", flush=True)
                failed = True

        sys.exit(1 if failed else 0)

    # name job_id manually for local test
    if args.job_id is not None:
        context = config_grid[args.job_id]
        run_one(context)
        return

    # Slurm array for current job
    slurm_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")

    if slurm_task_id is not None:
        context = config_grid[int(slurm_task_id)]
        run_one(context)
        return

    # local test: sequential test by enumerating
    for job_id, context in enumerate(config_grid):
        print(
            f"Running configuration "
            f"{job_id + 1}/{len(config_grid)}"
        )

        run_one(context)
        


if __name__ == "__main__":
    main()
