import argparse
import re
import json
from pathlib import Path
import os
import shutil
from tqdm import tqdm

file_extensions = {
    "python": ".py",
    "java": ".java",
}
def get_results(
        ranker_results,
        diffs_folder,
        parent_folder,
        generations_folder,
        Queries,
        metadata_file,
        output_folder,
        output_type,
        output_format,
        output_report_path,
):
    assert(os.path.exists(ranker_results))
    ranker_results = Path(ranker_results)

    if output_format == "both" or output_format == "mfiles":
        assert os.path.exists(generations_folder)
        generations_folder = Path(generations_folder)
    if output_folder == "both" or output_format == "diffs" or output_type == "smallest":
        assert os.path.exists(diffs_folder)
        diffs_folder = Path(diffs_folder)

    with open(metadata_file, "r") as f:
        query_metadata = json.load(f)

    query_list = Queries if Queries is not None else query_metadata.keys()

    score_expression_regex = re.compile("Score: (\d)")
    results_json = {}

    for query_id in tqdm(query_list):
        query_data = query_metadata[query_id]

        ranker_results_query_folder = ranker_results / query_data['folder_name'] 

        parent_diff_results_map = {}

        for file in sorted(os.listdir(ranker_results_query_folder)):
            if file.endswith(".json"):

                parent_file = file.split("_")[0] + ".py"
                diff_file = file.split("_logs.json")[0] + ".diff"

                with open(ranker_results_query_folder / file, 'r', encoding='utf-8') as f:
                    ranker_res = json.load(f)
                    llm_response = ranker_res['record']['Results'][0][1][0]
                    scores = score_expression_regex.findall(llm_response)
                    assert len(scores) == 1
                    if int(scores[0]) > 1:
                        if parent_file not in parent_diff_results_map:
                            parent_diff_results_map[parent_file] = []
                        parent_diff_results_map[parent_file].append(diff_file)
        
        if os.path.exists(os.path.join(output_folder , query_data['folder_name'])):
            shutil.rmtree(os.path.join(output_folder , query_data['folder_name']))
        os.makedirs(os.path.join(output_folder , query_data['folder_name']))

        for parent_file, diff_files in parent_diff_results_map.items():
            sized_diff_files = [(diff, os.path.getsize(diffs_folder / query_data['folder_name'] / diff)) for diff in diff_files]
            sorted_diffs = sorted(sized_diff_files , key=lambda x: x[1])
            if output_type == "smallest":
                sorted_diffs = [sorted_diffs[0]]
            for (diff, diff_size) in sorted_diffs:
                if output_format == "both" or output_format == "diffs":
                    diff_path = Path(diffs_folder) / query_data['folder_name'] / diff
                    assert diff_path.exists()
                    new_diff_path = Path(output_folder) / query_data['folder_name'] / diff
                    shutil.copy(diff_path, new_diff_path)
                if output_format == "both" or output_format == "mfiles":
                    mfiles_path = Path(generations_folder) / query_data['folder_name'] / (diff.split(".diff")[0] + ".py")
                    assert mfiles_path.exists()
                    new_mfiles_path = Path(output_folder) / mfiles_path.parts[-2] / mfiles_path.parts[-1]
                    shutil.copy(mfiles_path, new_mfiles_path)
            parent_diff_results_map[parent_file] = sorted_diffs
        
        results_json[query_id] = parent_diff_results_map
    
    print(results_json)
    with open(output_report_path, 'w', encoding='utf-8') as f:
        json.dump(results_json, f, indent=4)



if __name__ == "__main__":

    #parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--ranker_results", type=str, required=True)
    parser.add_argument("-p", "--parent_folder", type=str)
    parser.add_argument("-d", "--diffs_folder", type=str)
    parser.add_argument("-g", "--generations_folder", type=str)
    parser.add_argument("-t", "--output_type", choices=["smallest", "all"], default="all")
    parser.add_argument("-f", "--output_format", choices=["diffs", "mfiles", "both"], default="both")
    parser.add_argument("--Queries", nargs="+", help="CodeQL Queries to run")
    parser.add_argument("-m", "--metadata_file", type=str, help="Queries Metadata json file", default="metadata/python/metadata.json")
    parser.add_argument("-o", "--output_folder", type=str, required=True)
    parser.add_argument("-j", "--output_report_path", type=str, required=True)
    parser.add_argument("-l", "--language", type=str, default="py")
    args = parser.parse_args()

    get_results(
        ranker_results = args.ranker_results,
        diffs_folder = args.diffs_folder,
        parent_folder = args.parent_folder,
        generations_folder = args.generations_folder,
        Queries = args.Queries,
        metadata_file = args.metadata_file,
        output_folder = args.output_folder,
        output_type=args.output_type,
        output_format=args.output_format,
        output_report_path = args.output_report_path,
    )