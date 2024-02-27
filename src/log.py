import json
from pathlib import Path
import logging


class Log:
    """Class to log the results of the experiment in human readable format"""

    def __init__(self, exp_log_lib, fname):
        self.exp_log_lib = Path(exp_log_lib)
        self.fname = fname

    def create_json(self, params, record):
        logging.info(f"Logging results to {self.exp_log_lib / f'{self.fname}.json'}")
        exp_results = {
            "params": params,
            "record": record,
        }
        with open(self.exp_log_lib / f"{self.fname}.json", "w") as f:
            json.dump(exp_results, f)

        return exp_results

    def create_log_pyFile(self, exp_results):
        logging.info(f"Logging results to {self.exp_log_lib / f'{self.fname}.log'}")
        with open(self.exp_log_lib / f"{self.fname}.log", "w", encoding="utf-8") as f:
            f.write(
                f"Results for experiment with parameters -\n{exp_results['params']}\n"
            )
            for i, (prompt, result, data) in enumerate(
                exp_results["record"]["Results"]
            ):
                f.write(
                    f"\n\n\n======================== Prompt {i} ========================\n"
                )
                f.write(prompt)
                f.write(f"\n------------ Result for Prompt {i} ------------\n\n")
                f.write(str(result.text))
                f.write(f"\n\n------------ End Result for Prompt {i} ------------\n\n")
                f.write(f"Finish Reason: {result.finish_reason}\n\n")
                f.write(
                    f"======================== End Prompt {i} ========================\n"
                )

    def create_logs(self, record, params):
        exp_results = self.create_json(params, record)
        self.create_log_pyFile(exp_results)
