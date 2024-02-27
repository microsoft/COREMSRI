# %%
import pickle
import copy
import os
import re
from pathlib import Path
import argparse
import logging
import pandas as pd
import sys
import json

from CodeQLOracle.run_oracle import LocalRunCodeQL

logging.basicConfig(level=getattr(logging, "INFO"))

def get_unfixed_files(query_metadata, files_subset, args):
    """
    This function takes in queries, a mapping of queries to folder names, a subset of files, and an
    experiment configuration, and returns a dictionary of the number of unfixed files, the unfixed files
    themselves, the total number of files, and a dictionary of the unfixed files for each query.

    Fixed/unfixed status is marked as per marked/not marked by CodeQL.
    """
    top = args.n
    Queries = args.Queries
    # all unfixed files in 4 categories
    query_unfixed_files = {}
    for query in Queries:
        query_folderName = query_metadata[query]['folder_name']
        unfixed_in_set = 0
        try:
            csv_result_file = (
                args.codeql_result_path
                + "/"
                + f"{query_folderName}/results_{query_folderName}.csv"
            )
            query_result_file_new = pd.read_csv(
                csv_result_file,
                names=[
                    "CodeQL Vulnerability",
                    "Vulnerability Desc",
                    "Warning/Error",
                    "CodeQL Output",
                    "File",
                    "StartLine",
                    "StartChar",
                    "EndLine",
                    "EndChar",
                ],
            )
        except FileNotFoundError:
            raise FileNotFoundError(f"File {csv_result_file} not found")
        set_unique_files_names = query_result_file_new["File"].unique()
        # for only full file and method: include only indices 0 and 1
        list_of_files = copy.deepcopy(
            files_subset[query][0]
            + files_subset[query][1]
            + files_subset[query][2]
            + files_subset[query][3]
            + files_subset[query][4]
        )

        unfixed_files = set()
        unfixed_files_dict = {}
        for file_unfixed in set_unique_files_names:
            unformatted_file_check = copy.deepcopy(file_unfixed[1:])
            if top == 1:
                # only check first file
                if unformatted_file_check.split("_")[1].split(".py")[0] != "0":
                    continue

            file_check = unformatted_file_check.split("_")[0] + ".py"
            if file_check in list_of_files:
                if file_check in unfixed_files_dict:
                    unfixed_files_dict[file_check].append(unformatted_file_check)
                else:
                    unfixed_files_dict[file_check] = [unformatted_file_check]

                top_effective = get_file_fixes(
                    query, file_check, query_metadata, args.generated_folder_path
                )
                if len(unfixed_files_dict[file_check]) == min(top, top_effective):
                    unfixed_files.add(file_check)
                    unfixed_in_set += 1

        query_unfixed_files[query] = (
            unfixed_in_set,
            unfixed_files,
            len(list_of_files),
            unfixed_files_dict,
        )
    return query_unfixed_files


def update_aggregate_values(aggregate_data, types, validity, fixed, value):
    key = types + "::_::" + validity + "::_::" + fixed + "::_::"
    if key not in aggregate_data:
        aggregate_data[key] = value
    else:
        aggregate_data[key] += value
    return aggregate_data


def get_aggregated_plot_data(aggregate_data):
    aggregated_values = {"types": [], "validity": [], "fixed": [], "values": []}
    for key in aggregate_data:
        aggregated_values["types"].append(key.split("::_::")[0])
        aggregated_values["validity"].append(key.split("::_::")[1])
        fixed = None if key.split("::_::")[2] == "None" else key.split("::_::")[2]
        aggregated_values["fixed"].append(fixed)
        aggregated_values["values"].append(aggregate_data[key])

    return aggregated_values


def get_files_with_syntax_errors(
    query, files_subset, legends, query_metadata, args
):
    top = args.n
    error_pattern = r"\[WARN\] \[\d*\] Failed to analyse imports of ([a-zA-Z0-9\\/.:_\-\(\)\']*) : Syntax Error \(line \d*\)"
    error_pattern_expr = re.compile(error_pattern)
    parent_path = os.path.abspath(args.generated_folder_path + "/")

    log_dir = (
        args.codeql_db_path
        + "/"
        + f"{query_metadata[query]['folder_name']}/python-database/log/"
    )

    if not os.path.exists(log_dir):
        raise FileNotFoundError(log_dir)
    for p in os.listdir(log_dir):
        if p.startswith("database-create"):
            log_path = p
    with open(log_dir + log_path, "r") as f:
        logs = f.read()

    files_with_syntax_error = error_pattern_expr.findall(logs)
    edited_files_with_syntax_error = {}
    files_subset_with_syntax_error = {}
    for file_with_error in files_with_syntax_error:
        if not os.path.exists(file_with_error):
            print("danger: ", file_with_error)
            sys.exit(-1)

        child_path = os.path.abspath(file_with_error)
        t1 = os.path.commonpath([parent_path, child_path])
        t2 = os.path.commonpath([parent_path])
        if os.path.commonpath([parent_path]) == os.path.commonpath(
            [parent_path, child_path]
        ):
            fname = Path(file_with_error).name
            if top == 1:
                if fname.split("_")[1].split(".py")[0] != "0":
                    continue

            org_file_name = fname.split("_")[0] + ".py"
            if org_file_name in edited_files_with_syntax_error:
                edited_files_with_syntax_error[org_file_name].append(fname)
            else:
                edited_files_with_syntax_error[org_file_name] = [fname]

    for fn in edited_files_with_syntax_error.keys():
        assert len(edited_files_with_syntax_error[fn]) <= top

    # save the splits
    files_subset_with_syntax_error[query] = (
        {}
    )  # all unfixed <editede> files in 4 categories
    for i in range(len(files_subset[query])):
        temp_q = {}
        for file_name in files_subset[query][i]:
            if file_name in edited_files_with_syntax_error:
                temp_q[file_name] = edited_files_with_syntax_error[file_name]
        files_subset_with_syntax_error[query][legends[i]] = temp_q

    return edited_files_with_syntax_error, files_subset_with_syntax_error


def get_considered_files(
    Queries, query_metadata, root_folder, generated_files_folder
):
    considered_files = {}
    files_with_no_generated_files = {}
    for query in Queries:
        folder_name = query_metadata[query]['folder_name']
        files_in_query = set(os.listdir(f"{root_folder}/{folder_name}"))
        py_files_in_query = set([x for x in files_in_query if x.endswith(".py")])

        gff = f"{generated_files_folder}/{folder_name}"

        generated_files_in_query = set(os.listdir(gff))
        generated_py_files_in_query = set(
            [
                (x.split("_")[0] + ".py")
                for x in generated_files_in_query
                if x.endswith(".py")
            ]
        )

        considered_files[query] = py_files_in_query.intersection(
            generated_py_files_in_query
        )
        files_with_no_generated_files[query] = (
            py_files_in_query - generated_py_files_in_query
        )

    return considered_files, files_with_no_generated_files


def get_file_fixes(query, org_file, query_metadata, generated_files_folder):
    folder_name = query_metadata[query]['folder_name']
    gff = f"{generated_files_folder}/{folder_name}"

    generated_files_in_query = set(os.listdir(gff))
    file_generated_py_files_in_query = set(
        [
            x
            for x in generated_files_in_query
            if ((x.endswith(".py")) and (org_file == x.split("_")[0] + ".py"))
        ]
    )

    return len(file_generated_py_files_in_query)


def get_plot_data(query_metadata, files_subset, query_unfixed_files, args):
    top = args.n
    Queries = args.Queries
    legends = {0: "File", 1: "Method", 2: "Window", 3: "Method+Window", 4: "Class"}

    edited_file_lists = {}  # to know which files are fixed/not fixed, for debug purpose
    query_stats = {}  # for plot
    aggregate_data = {}
    considered_files, files_with_no_generated_files = get_considered_files(
        Queries, query_metadata, args.folder_path, args.generated_folder_path
    )

    for query in Queries:
        assert len(legends) == len(files_subset[query])
        query_stats[query] = {"types": [], "validity": [], "fixed": [], "values": []}
        folder_name = query_metadata[query]['folder_name']

        # get files with syntax errors
        edited_files_with_syntax_error, files_subset_with_syntax_error = (
            get_files_with_syntax_errors(
                query, files_subset, legends, query_metadata, args
            )
        )
        # get stats
        valid = 0
        not_valid = 0
        total = 0
        edited_file_lists[query] = {
            "invalid": set(),
            "valid_fixed": set(),
            "valid_unfixed": set(),
        }
        query_invalid_files = set()
        query_valid_files = set()
        query_valid_unfixed_files = set()
        query_valid_fixed_files = {}
        for i in range(len(files_subset[query])):
            query_bucket_specific_considered_files = set(
                files_subset[query][i]
            ).intersection(considered_files[query])
            # get numbers of syntacticallly corret or incorrect files
            temp_valid = 0
            temp_not_valid = 0
            for file_name in query_bucket_specific_considered_files:
                top_effective = get_file_fixes(
                    query, file_name, query_metadata, args.generated_folder_path
                )
                if file_name in edited_files_with_syntax_error and len(
                    edited_files_with_syntax_error[file_name]
                ) == min(top, top_effective):
                    temp_not_valid += 1
                    query_invalid_files.add((file_name, i))
                else:
                    temp_valid += 1
                    query_valid_files.add((file_name, i))
                total += 1
            valid += temp_valid
            not_valid += temp_not_valid

            # add values for plot
            if i in [0, 1, 2, 3, 4]:
                # files_wo_syntax_error: files for which no edited file is syntactically correct
                files_wo_syntax_error = query_bucket_specific_considered_files - set(
                    [
                        x
                        for x in files_subset_with_syntax_error[query][
                            legends[i]
                        ].keys()
                        if len(files_subset_with_syntax_error[query][legends[i]][x])
                        == min(
                            top,
                            get_file_fixes(
                                query,
                                x,
                                query_metadata,
                                args.generated_folder_path,
                            ),
                        )
                    ]
                )

                assert temp_valid == len(files_wo_syntax_error)
                unfixed_ones = 0
                fixed_ones = 0
                unfixed_files = []

                for ff in files_wo_syntax_error:
                    top_effective = get_file_fixes(
                        query, ff, query_metadata, args.generated_folder_path
                    )
                    # files in files_wo_syntax_error may have some syn. incorrect files
                    if (
                        ff in edited_files_with_syntax_error
                        and ff in query_unfixed_files[query][3]
                    ):
                        if len(edited_files_with_syntax_error[ff]) + len(
                            query_unfixed_files[query][3][ff]
                        ) == min(top, top_effective):
                            unfixed_ones += 1
                            unfixed_files.append(ff)
                        else:
                            fixed_ones += 1
                    # files in files_wo_syntax_error with 0 syn. incorrect files, but no fixed files
                    elif ff in query_unfixed_files[query][3]:
                        if len(query_unfixed_files[query][3][ff]) == min(
                            top, top_effective
                        ):
                            unfixed_ones += 1
                            unfixed_files.append(ff)
                        else:
                            fixed_ones += 1
                    else:
                        fixed_ones += 1

                assert temp_valid == unfixed_ones + fixed_ones

                query_stats[query]["types"].append(legends[i])
                query_stats[query]["validity"].append("syn. correct")
                query_stats[query]["fixed"].append("fixed")
                query_stats[query]["values"].append(temp_valid - unfixed_ones)
                aggregate_data = update_aggregate_values(
                    aggregate_data,
                    legends[i],
                    "syn. correct",
                    "fixed",
                    temp_valid - unfixed_ones,
                )

                # update edited_file_lists
                unfixed_files_dict = query_unfixed_files[query][3]
                for ff in unfixed_files:
                    query_valid_unfixed_files.add((ff, i))
                for fg in query_valid_files:
                    if fg[1] == i and fg not in query_valid_unfixed_files:
                        # query_valid_fixed_files.add(fg)

                        # files which are not syntactically incorrect
                        if fg[0] in edited_files_with_syntax_error:
                            _plausible_all_files = set(
                                x
                                for x in [
                                    fg[0].split(".py")[0] + f"_{i}.py"
                                    for i in range(top)
                                ]
                                if x not in edited_files_with_syntax_error[fg[0]]
                            )
                        else:
                            _plausible_all_files = set(
                                [fg[0].split(".py")[0] + f"_{i}.py" for i in range(top)]
                            )

                        gff = f"{args.generated_folder_path}/{folder_name}"

                        _all_files = set()
                        for apf in _plausible_all_files:
                            if os.path.exists(f"{gff}/{apf}"):
                                _all_files.add(apf)

                        # files which are syntactically correct and not marked by codeql, i.e., fixed files
                        if fg[0] in unfixed_files_dict:
                            if _all_files - set(unfixed_files_dict[fg[0]]):
                                query_valid_fixed_files[fg] = _all_files - set(
                                    unfixed_files_dict[fg[0]]
                                )
                        else:
                            query_valid_fixed_files[fg] = _all_files

                query_stats[query]["types"].append(legends[i])
                query_stats[query]["validity"].append("syn. correct")
                query_stats[query]["fixed"].append("not fixed")
                query_stats[query]["values"].append(unfixed_ones)
                aggregate_data = update_aggregate_values(
                    aggregate_data,
                    legends[i],
                    "syn. correct",
                    "not fixed",
                    unfixed_ones,
                )
            else:
                query_stats[query]["types"].append(legends[i])
                query_stats[query]["validity"].append("syn. correct")
                query_stats[query]["fixed"].append(None)
                query_stats[query]["values"].append(temp_valid)
                aggregate_data = update_aggregate_values(
                    aggregate_data, legends[i], "syn. correct", "None", temp_valid
                )

            query_stats[query]["types"].append(legends[i])
            query_stats[query]["validity"].append("syn. incorrect")
            query_stats[query]["fixed"].append(None)
            query_stats[query]["values"].append(temp_not_valid)
            aggregate_data = update_aggregate_values(
                aggregate_data, legends[i], "syn. incorrect", "None", temp_not_valid
            )

        query_stats[query]["types"].append("WO_model_op")
        query_stats[query]["validity"].append("NA")
        query_stats[query]["fixed"].append("NA")
        query_stats[query]["values"].append(len(files_with_no_generated_files[query]))
        aggregate_data = update_aggregate_values(
            aggregate_data,
            "WO_model_op",
            "NA",
            "NA",
            len(files_with_no_generated_files[query]),
        )
        assert valid + not_valid == total

        edited_file_lists[query]["invalid"] = query_invalid_files
        edited_file_lists[query]["valid_fixed"] = query_valid_fixed_files
        edited_file_lists[query]["valid_unfixed"] = query_valid_unfixed_files
        edited_file_lists[query]["total"] = len(
            considered_files[query]
        ) + len(  # sum of invalid, valid_fixed, valid_unfixed
            files_with_no_generated_files[query]
        )

    # add aggregate_data
    query_stats["Aggregate"] = get_aggregated_plot_data(aggregate_data)
    return query_stats, files_subset_with_syntax_error, edited_file_lists


def generate_pickle_file(args):
    Queries = args.Queries

    queries_meta_file = Path(args.queries_meta_file)
    assert os.path.exists(queries_meta_file)
    with open(queries_meta_file, 'r', encoding='utf-8') as f:
        query_metadata = json.load(f)

    Queries = Queries if Queries is not None else query_metadata.keys()
    args.Queries = Queries

    with open(f"{args.files_subset_pickle_name}", "rb") as f:
        files_subset = pickle.load(f)

    query_unfixed_files = get_unfixed_files(query_metadata, files_subset, args)
    query_stats, _, edited_file_lists = get_plot_data(
        query_metadata, files_subset, query_unfixed_files, args
    )

    for query in query_stats.keys():
        value = query_stats[query]["values"]
        valid = query_stats[query]["validity"]
        fixed = query_stats[query]["fixed"]
        split = query_stats[query]["types"]
        df = pd.DataFrame(dict(split=split, valid=valid, fixed=fixed, value=value))
        if query != "Aggregate":
            fixed = sum(
                df[(df["valid"] == "syn. correct") & (df["fixed"] == "fixed")]["value"]
            )
            total = sum(df["value"])
            assert fixed == len(edited_file_lists[query]["valid_fixed"].keys())
            assert total == edited_file_lists[query]["total"]

    if not Path(args.output_path).exists():
        os.makedirs(Path(args.output_path))
    pkl_file_path = Path(args.output_path) / Path(
        args.description + "_filefix_types_Top" + str(args.n) + ".pickle"
    )
    logging.info(f"Saving pickle file at {pkl_file_path}")
    with open(pkl_file_path, "wb") as handle:
        pickle.dump(edited_file_lists, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_path", "-i", type=str, required=True)
    parser.add_argument("--generated_folder_path", "-g", type=str, required=True)
    parser.add_argument("--codeql_db_path", "-d", type=str, required=True)
    parser.add_argument("--codeql_result_path", "-r", type=str, required=True)
    parser.add_argument("--output_path", "-o", type=str, default="../coreoutput")
    parser.add_argument(
        "--files_subset_pickle_name", "-s", type=str, default="files_subset_2000.pkl"
    )
    parser.add_argument("--description", "-inf", type=str, default="demo")
    parser.add_argument("--n", type=int, default=10)
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

    generate_pickle_file(args)

# %%
