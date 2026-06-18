import os

import numpy as np
import pandas as pd


class Logger:
    def __init__(self, optimizer: any):
        self.optimizer = optimizer
        self.df = None

    def __call__(self, contents, *args, **kwds):
        status = self.get_status()
        if self.df is None:
            columns = []
            for key, value in contents.items():
                columns.append(key)
            for key, value in status.items():
                if isinstance(value, (list, np.ndarray)):
                    columns.extend([f"{key}_{i}" for i in range(len(value))])
                else:
                    columns.append(key)
            self.df = pd.DataFrame(columns=columns)

        row = []
        for key, value in contents.items():
            row.append(value)
        for key, value in status.items():
            if isinstance(value, (list, np.ndarray)):
                row.extend(value)
            else:
                row.append(value)
        self.df.loc[len(self.df)] = row

    def get_status(self):
        """Get the status of the optimizer."""
        if (
            self.optimizer.__class__.__name__ == "PBIL"
            or self.optimizer.__class__.__name__ == "PBIL_MOD"
        ):
            return {
                "generation": getattr(self.optimizer, "generation", None),
                "current_best_fvalue": getattr(
                    self.optimizer, "current_best_fvalue", None
                ),
                "current_avg_fvalue": getattr(self.optimizer, "avg_fvalue", None),
                "weights_q": getattr(self.optimizer, "_weights", None),
                # "best_indv_q": np.argmax(getattr(self.optimizer, 'best_solution_q', np.array([None])), axis=1),
                "opt_cat_prob": getattr(self.optimizer, "_q", np.zeros((1, 1)))[:, 0],
                "hat_SNRq": getattr(self.optimizer, "hat_SNRq", None),
                "hat_signal": getattr(self.optimizer, "hat_signal", None),
                "ngrad_norm": getattr(self.optimizer, "ngrad_norm", None),
                "delta": getattr(self.optimizer, "_delta", None),
                "delta_exp": getattr(self.optimizer, "delta_exp", None),
                "delta_change_ratio": getattr(
                    self.optimizer, "delta_change_ratio", None
                ),
                "eps": getattr(self.optimizer, "_eps", None),
                "pnorm": getattr(self.optimizer, "pnorm", None),
                "beta": getattr(self.optimizer, "beta", None),
                "entropy": getattr(self.optimizer, "entropy", None),
                "_update_interval": getattr(self.optimizer, "_update_interval", None),
            }
        else:
            return {
                "generation": getattr(self.optimizer, "generation", None),
                "current_best_fvalue": getattr(
                    self.optimizer, "current_best_fvalue", None
                ),
                "current_avg_fvalue": getattr(self.optimizer, "avg_fvalue", None),
                "weights_q": getattr(self.optimizer, "_weights", None),
                # "best_indv_q": np.argmax(getattr(self.optimizer, 'best_solution_q', np.array([None])), axis=1),
                "opt_cat_prob": getattr(self.optimizer, "_q", np.zeros((1, 1)))[:, 1],
                "hat_SNRq": getattr(self.optimizer, "hat_SNRq", None),
                "hat_signal": getattr(self.optimizer, "hat_signal", None),
                "ngrad_norm": getattr(self.optimizer, "ngrad_norm", None),
                "delta": getattr(self.optimizer, "_delta", None),
                "delta_exp": getattr(self.optimizer, "delta_exp", None),
                "delta_change_ratio": getattr(
                    self.optimizer, "delta_change_ratio", None
                ),
                "eps": getattr(self.optimizer, "_eps", None),
                "pnorm": getattr(self.optimizer, "pnorm", None),
                "beta": getattr(self.optimizer, "beta", None),
                "entropy": getattr(self.optimizer, "entropy", None),
                "_update_interval": getattr(self.optimizer, "_update_interval", None),
            }

    def save(self, save_dir: str, config: dict = None):

        os.makedirs(save_dir, exist_ok=True)
        self.df.to_csv(os.path.join(save_dir, "log.csv"), index=False)

        if config is not None:
            with open(
                os.path.join(save_dir, "config.json"), "w", encoding="utf-8"
            ) as f:
                import json

                json.dump(config, f, indent=4, ensure_ascii=False)
