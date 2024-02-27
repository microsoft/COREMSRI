# %%
import pickle
from pathlib import Path
import os
import argparse
import logging
import sys
import csv
import re
import json
from tqdm import tqdm
import shutil

from CodeQLOracle.run_oracle import LocalRunCodeQL

logging.basicConfig(level=getattr(logging, "INFO"))


def add_uncompiled_files(query, query_folderName, out, csv_file):
    out = out.decode("UTF-8")
    out_lines = [line.strip().split(" ") for line in out.split("\n")]
    for line in out_lines:
        if len(line) > 4 and line[3] == "[WARN]":
            msg = " ".join(line[5:])
            res = re.search(f"{query_folderName}\/.*\.py", msg)
            if res:
                file_path = res.group(0)
                with open(csv_file, "a") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            query,
                            msg,
                            "uncompiled",
                            "error",
                            file_path.removeprefix(f"{query_folderName}"),
                            "0",
                            "0",
                            "0",
                            "0",
                        ]
                    )


def run_codeql(args):
    Queries = args.Queries

    queries_meta_file = Path(args.queries_meta_file)
    assert os.path.exists(queries_meta_file)
    with open(queries_meta_file, 'r', encoding='utf-8') as f:
        query_metadata = json.load(f)

    LocalVerifier = LocalRunCodeQL(None, args.codeql_pack_path)

    if Queries is not None:
        query_list = Queries
    else:
        query_list = query_metadata.keys()

    for query in tqdm(query_list):
        query_data = query_metadata[query]
        query_folderName = query_data['folder_name']
        if Path(f"{args.codeql_db_path}/{query_folderName}").exists():
            shutil.rmtree(Path(f"{args.codeql_db_path}/{query_folderName}"))
        if Path(f"{args.codeql_result_path}/{query_folderName}").exists():
            shutil.rmtree(Path(f"{args.codeql_result_path}/{query_folderName}"))

        os.makedirs(Path(f"{args.codeql_db_path}/{query_folderName}"))
        os.makedirs(Path(f"{args.codeql_result_path}/{query_folderName}"))
        LocalVerifier.exp_log_lib = args.folder_path + f"/{query_folderName}"
        LocalVerifier.run_codeql_single_query(
            query,
            f"{query_folderName}",
            args.folder_path,
            args.codeql_db_path,
            args.codeql_result_path,
            simple=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_path", "-i", type=str, required=True)
    parser.add_argument("--codeql_pack_path", "-c", type=str, default="../codeql")
    parser.add_argument("--codeql_db_path", "-d", type=str, default="../db")
    parser.add_argument("--codeql_result_path", "-o", type=str, default="../codeqlout")
    parser.add_argument("--log", type=str, default="WARNING")
    parser.add_argument(
        "--Queries",  # name on the CLI - drop the `--` for positional/required parameters
        nargs="*",  # 0 or more values expected => creates a list
        type=str,
    )
    parser.add_argument(
        "--queries_meta_file", type=str, default="metadata/python/metadata.json"
    )

    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log.upper()))

    if not Path(args.codeql_db_path).exists():
        os.makedirs(Path(args.codeql_db_path))
    if not Path(args.codeql_result_path).exists():
        os.makedirs(Path(args.codeql_result_path))

    run_codeql(args)
