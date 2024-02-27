import subprocess
import os
import argparse
import json
import pickle
from tqdm import tqdm
import shutil


def get_diffs(args):

    assert os.path.exists(args.parent_folder)
    assert os.path.exists(args.generations_folder)
    assert os.path.exists(args.analysis_file)
    assert os.path.exists(args.metadata_file)
    if args.save_diffs != "False":
        if os.path.exists(args.save_folder):
            shutil.rmtree(args.save_folder)
            os.makedirs(args.save_folder)


    with open(args.analysis_file, "rb") as f:
        analysis = pickle.load(f)

    with open(args.metadata_file, "r") as f:
        metadata = json.load(f)

    for query in tqdm(analysis.keys()):
        query_folderName = metadata[query]["folder_name"]
        assert os.path.exists(os.path.join(args.parent_folder, query_folderName))
        assert os.path.exists(os.path.join(args.generations_folder, query_folderName))
        if args.save_diffs != "False":
            if os.path.exists(os.path.join(args.save_folder, query_folderName)):
                os.rmdir(os.path.join(args.save_folder, query_folderName))
            os.makedirs(os.path.join(args.save_folder, query_folderName))

        if args.deduplicate:
            total_deduped = 0
            total_valid_files = 0

        for pf in analysis[query]["valid_fixed"]:
            assert os.path.exists(
                os.path.join(args.parent_folder, query_folderName, pf[0])
            )

            if args.save_diffs == "joined":
                diffstr = ""

            if args.deduplicate:
                diffs_index = {}

            for qf in analysis[query]["valid_fixed"][pf]:
                assert os.path.exists(
                    os.path.join(args.generations_folder, query_folderName, qf)
                )

                cmd = "git diff --ignore-space-at-eol --ignore-blank-lines --ignore-cr-at-eol --no-index {} {}".format(
                    os.path.join(
                        args.parent_folder, '"' + query_folderName + '"', pf[0]
                    ),
                    os.path.join(
                        args.generations_folder, '"' + query_folderName + '"', qf
                    ),
                )
                proc = subprocess.Popen(
                    cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                out, err = proc.communicate()
                assert err.decode("utf-8") == ""
                out_decoded = out.decode("utf-8")
                out_split = out_decoded.splitlines()
                diff = out_split[4:]
                # assert diff != []
                diff = "\n".join(diff)
                if args.deduplicate:
                    total_valid_files += 1
                    if diff not in diffs_index:
                        diffs_index[diff] = [qf]
                    else:
                        # print('copy diff {}'.format(qf))
                        total_deduped += 1
                        diffs_index[diff].append(qf)

                else:
                    if args.save_diffs == "joined":
                        diffstr += "Diff between {} and {}\n".format(pf[0], qf)
                        diffstr += diff
                        diffstr += "\n========================================================\n"
                    elif args.save_diffs == "seperate":
                        with open(
                            os.path.join(
                                args.save_folder,
                                query_folderName,
                                qf.split(".")[0] + ".diff",
                            ),
                            "w",
                        ) as f:
                            f.write(diff)
                    else:
                        print("Diff between {} and {}\n".format(pf[0], qf))
                        print(diff)
                        print(
                            "========================================================"
                        )

            if args.deduplicate:

                for diff, qfs in diffs_index.items():
                    if args.save_diffs == "joined":
                        diffstr += (
                            "Diff between {} and ".format(pf[0]) + ", ".join(qfs) + "\n"
                        )
                        diffstr += diff
                        diffstr += "\n========================================================\n"

                    elif args.save_diffs == "seperate":
                        with open(
                            os.path.join(
                                args.save_folder,
                                query_folderName,
                                qfs[0].split(".")[0] + ".diff",
                            ),
                            "w",
                        ) as f:
                            f.write(diff)
                    else:
                        print(
                            "Diff between {} and ".format(pf[0]) + ", ".join(qfs) + "\n"
                        )
                        print(diff)
                        print(
                            "========================================================"
                        )

                if args.save_diffs == "joined" or args.save_diffs == "seperate":
                    with open(
                        os.path.join(
                            args.save_folder,
                            query_folderName,
                            "{}_diffmap.pkl".format(pf[0].split(".")[0]),
                        ),
                        "wb",
                    ) as f:
                        pickle.dump(diffs_index, f)

            if args.save_diffs == "joined":
                with open(
                    os.path.join(
                        args.save_folder,
                        query_folderName,
                        "{}.diff".format(pf[0].split(".")[0]),
                    ),
                    "w",
                ) as f:
                    f.write(diffstr)

        if args.deduplicate:
            print(f"{query} ->  {total_deduped} out of {total_valid_files}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--parent_folder", type=str, required=True)
    parser.add_argument("-g", "--generations_folder", type=str, required=True)
    parser.add_argument("-a", "--analysis_file", type=str, required=True)
    parser.add_argument("-m", "--metadata_file", type=str, default="metadata/python/metadata.json")
    parser.add_argument(
        "--save_diffs", choices=["False", "seperate", "joined"], default="seperate"
    )
    parser.add_argument('-s', "--save_folder", type=str)
    parser.add_argument("--deduplicate", action="store_true", dest="deduplicate")
    args = parser.parse_args()
    get_diffs(args)
