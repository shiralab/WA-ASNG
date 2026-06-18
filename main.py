import argparse
import json
import logging
import os
import platform
import time

import numpy as np

from src.algorithms import get_algorithm
from src.problems import get_problem
from src.utils import Logger


def setup_logging(log_path):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename=log_path,
        filemode="w",
    )
    console = logging.StreamHandler()
    console.setLevel(logging.ERROR)
    logging.getLogger("").addHandler(console)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", type=str)
    args = parser.parse_args()

    with open(args.json_path, "r", encoding="utf-8") as f:
        task_config = json.load(f)

    # Setting of problem
    problem_config = task_config["problem"]

    cat_dim = problem_config["dim"]
    cat_num = np.full((cat_dim,), 2)
    problem_name = problem_config["name"]
    seed = problem_config["seed"]
    max_evals = problem_config.get("max_evals", 100000)

    with_noise = problem_name.lower().startswith("noisy")
    if with_noise:
        noisevar = problem_config.get("noisevar", 0.0)
        problem = get_problem(
            problem_name, cat_dim=cat_dim, cat_num=cat_num, noisevar=noisevar, seed=seed
        )
        problem_name += f"_{noisevar}"
    else:
        problem = get_problem(problem_name, cat_dim=cat_dim, cat_num=cat_num)

    # Setting of algorithm
    algorithm_config = task_config["algorithm"]
    algo_name = algorithm_config["name"]
    param = algorithm_config.get("param", {})

    # Save execution information
    uname = platform.uname()
    save_config = task_config.copy()
    save_config["execution_info"] = {
        "os": f"{uname.system} {uname.release}",
        "node_name": uname.node,
        "machine": uname.machine,
        "processor": uname.processor,
    }

    # Run the optimization
    optimizer = get_algorithm(algo_name, cat_num=cat_num, seed=seed, **param)
    logger = Logger(optimizer)

    # Create save directory
    params_str = "_".join(
        [f"{k}_{v}" for k, v in param.items() if not isinstance(v, list)]
    )
    save_dir = f"results/{optimizer.__class__.__name__.lower()}/{params_str}/{problem_name.lower()}/dim_{cat_dim}/seed_{seed:02}"
    os.makedirs(save_dir, exist_ok=True)

    setup_logging(os.path.join(save_dir, "execution.log"))

    save_config["execution_info"]["start_time"] = time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime()
    )

    logging.info("Starting optimization.")
    n_evals = 0

    best_fvalue = -np.inf
    best_true_fvalue = -np.inf
    best_true_c = None
    try:
        while n_evals < max_evals:
            solutions = []
            for _ in range(optimizer.population_size):
                c = optimizer.ask()
                if with_noise:
                    true_value, value = problem._evaluate(c)
                else:
                    value = problem._evaluate(c)
                    true_value = value

                if value > best_fvalue:
                    best_fvalue = value
                if true_value > best_true_fvalue:
                    best_true_fvalue = true_value
                    best_true_cat = np.argmax(c, axis=1)

                solutions.append((c, value))
                n_evals += 1
            optimizer.tell(solutions)

            if best_true_fvalue == problem.best_fvalue:
                contents = {
                    "evals": n_evals,
                    "best_fvalue": best_fvalue,
                    "best_true_fvalue": best_true_fvalue,
                    "is_success": True,
                }
                columns = [f"best_true_cat_{i}" for i in range(len(best_true_cat))]
                for i, v in enumerate(best_true_cat):
                    contents[columns[i]] = v
                logger(contents)
                break

            else:
                contents = {
                    "evals": n_evals,
                    "best_fvalue": best_fvalue,
                    "best_true_fvalue": best_true_fvalue,
                    "is_success": False,
                }
                columns = [f"best_true_cat_{i}" for i in range(len(best_true_cat))]
                for i, v in enumerate(best_true_cat):
                    contents[columns[i]] = v
                logger(contents)
        logging.info("Optimization completed successfully.")
    except KeyboardInterrupt:
        logging.warning("Optimization interrupted by user.")
    except Exception as e:
        logging.error(e)
        logging.info("Optimization terminated with an error.")

    save_config["execution_info"]["end_time"] = time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime()
    )
    logger.save(save_dir, save_config)
