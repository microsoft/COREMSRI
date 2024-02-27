import os
import subprocess
import pandas as pd
from datetime import datetime
import json
from pathlib import Path


_FILE_PATH = Path(__file__).resolve()
_FILE_DIR = _FILE_PATH.parent


CODEQL_METADATA_FILE_NAME = "codeql_meta.json"
RUN_CODEQL_SCRIPT_FILE_NAME = "run_oracle_codeql_simple.sh"


class LocalRunCodeQL:
    """Class to implement different oracles"""

    def __init__(self, exp_log_lib, codeql_pack_path):
        self.exp_log_lib = exp_log_lib  # path to the experiment folder
        self.codeql_pack_path = codeql_pack_path

        query_data: dict = json.load(open(_FILE_DIR / CODEQL_METADATA_FILE_NAME, "r"))
        self.codeql_queries_dict: dict = {
            q["name"]: q["codeql_query"] for q in query_data
        }

    def run_codeql_single_query(
        self,
        query_name,
        folder_name,
        exp_folder,
        codeql_db_path,
        codeql_result_path,
        simple=False,
    ):
        """
        codeql_query: query name
        folder_name: folder containing files for evaluation
        exp_folder: path to the experiment folder
        """
        codeql_query = self.codeql_queries_dict[query_name]
        # invoke bash script
        bash_script = (
            f"bash "
            + str(_FILE_DIR / RUN_CODEQL_SCRIPT_FILE_NAME)
            + " "
            + codeql_query
            + ' "'
            + folder_name
            + '" '
            + exp_folder
            + " "
            + codeql_db_path
            + " "
            + codeql_result_path
            + " "
            + self.codeql_pack_path
        )
        os.system(bash_script)
