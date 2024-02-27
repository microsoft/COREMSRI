import os
import pandas as pd
import json
from pathlib import Path
import sys
import copy
import logging
import pickle
import argparse
from tqdm import tqdm

from dotenv import load_dotenv
load_dotenv()


from localise import Localise
from openai_handler import OpenAIHandler
from prompt_handler import PromptConstructor
from log import Log
from patch_fixer import Patcher
from Utils.file_handling import post_process_adjust_indentation, sanitize_llm_response

from Utils.file_handling import StructuredCodeSplitter
from Utils.file_handling import count_tokens

logging.basicConfig(level=getattr(logging, "INFO"))

file_extensions = {
    "python": ".py",
    "java": ".java",
}


def fix(
    Exp_folder,
    Target_folder,
    result_csv_path,
    files_subset_pickle_name,
    timeout_mf,
    timeout,
    minm_timeout,
    max_attempts,
    localisation_strategy,
    deduplicate,
    localisation_flag,
    model,
    max_tokens,
    temperature,
    stop,
    n,
    system_message,
    Queries,
    dry_run,
    overwrite,
    lang,
    queries_meta_file,
    template_file,
):

    api_key = os.environ["OPENAI_API_KEY"]
    api_base = os.environ["OPENAI_API_BASE"] if "OPENAI_API_BASE" in os.environ else None
    api_type = os.environ['OPENAI_API_TYPE'] if "OPENAI_API_TYPE" in os.environ else 'openai'
    api_version = os.environ["OPENAI_API_VERSION"] if "OPENAI_API_TYPE" in os.environ else None

    # assert input args
    assert len(n) == len(temperature)

    # Extraction module
    all_queries_data = json.load(open(queries_meta_file, "r", encoding="utf-8"))

    Localise_obj = Localise()

    split_type_data = {}

    if Queries is not None:
        query_list = Queries
    else:
        query_list = all_queries_data.keys()

    for query_id in tqdm(query_list):
        query_data = all_queries_data[query_id]

        print(f"----------{query_data['name']}----------")
        print(f"Query ID: {query_id}\nQuery_data: {query_data}")

        query_folderName = query_data["folder_name"]

        query_result_file = pd.read_csv(
            Path(result_csv_path)
            / query_folderName
            / f"results_{query_folderName}.csv",
            names=[
                "Vulnerability",
                "Vulnerability Desc",
                "Warning/Error",
                "ErrorOutput",
                "File",
                "StartLine",
                "StartChar",
                "EndLine",
                "EndChar",
            ],
        )

        query_output_dir = Path(Target_folder) / query_folderName
        query_output_log_dir = query_output_dir / "logs"
        query_output_log_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Logging to {query_output_log_dir}")

        # sort files alphabetically
        file_names = []
        print("=================================================")
        print(os.getcwd())
        print("=================================================")
        for file in os.listdir(Path(Exp_folder) / query_folderName):
            if file.endswith(file_extensions[lang]):
                file_names.append(file)

        # sort file_names alphabetically
        file_names.sort()

        with open(template_file, "r") as f:
            prompt_template_text = f.read()

        prompt_constructor = PromptConstructor(
            template={
                "basic": prompt_template_text,
            },
            values={
                "query_name": query_data["name"],
                "description": query_data["desc"],
                "recommendation": query_data["recm"] if "recm" in query_data else None,
                "language": lang,
                "Ruleset": query_data["ruleset"],
            },
            strategy_variables={
                "localisation_flag": localisation_flag,
                "localisation_strategy": localisation_strategy,
                "deduplicate": deduplicate,
                "localise_obj": Localise_obj,
                "recommendation_flag": query_data["contextual"],
            },
            token_counter=lambda s: count_tokens(s, model_name=model),
            max_length=max_tokens,
            splitter=StructuredCodeSplitter(
                lang=lang,
                config={
                    "chunk_lim": max_tokens,
                },
            ),
        )

        print("len of file_names", len(file_names))
        split_type_buckets = [[], [], [], [], []]
        for file_name in tqdm(file_names):
            print(f"{'='*50}\nProcessing file: {file_name}\n{'='*50}")
            answer_span = copy.deepcopy(
                query_result_file[query_result_file["File"] == f"/{file_name}"][
                    ["StartLine", "StartChar", "EndLine", "EndChar", "ErrorOutput"]
                ]
            )
            answer_span_line_locs = list(
                zip(answer_span["StartLine"], answer_span["EndLine"])
            )
            # rl = namedtuple('rl', ['start_line', 'start_column', 'end_line', 'end_column', 'supporting_fact_locations'])

            line_of_interest_namedtuple_map = {}
            for i, row in answer_span.iterrows():
                line_of_interest_namedtuple_map[row["StartLine"] - 1] = (
                    Localise_obj.get_result_locations(row, "positive")
                )

            file_path = Path(Exp_folder / Path(f"{query_folderName}/{file_name}"))
            file_contents = file_path.read_text(encoding="utf-8")
            file_lines = file_contents.splitlines()

            try:
                all_prompts_T = prompt_constructor.construct(
                    file_contents=file_contents,
                    answer_span_locs=answer_span_line_locs,
                    line_of_interest_namedtuple_map=line_of_interest_namedtuple_map,
                )
            except Exception as e:
                logging.info(f"Prompt construction failed for file: {file_name}")
                logging.info(str(e))
                split_type_buckets[2].append(file_name)
                continue

            # logging prompt data:
            print(f"Number of prompts: {len(all_prompts_T)}")
            for i, prompt in enumerate(all_prompts_T):
                print(f"Prompt {i}: {prompt.metadata['source_example'][1]}")
                print(
                    f"Prompt {i} length: {count_tokens(prompt.value, model_name=model)[0]}"
                )
                print(f"Prompt {i} split type: {prompt.metadata['split_type']}")

            split_types = set([p.metadata["split_type"] for p in all_prompts_T])
            if "whole_file" in split_types:
                split_type_buckets[0].append(file_name)
            elif len(split_types) == 0:
                print(f"Prompts list is empty for file: {file_path}")
                split_type_buckets[2].append(file_name)
            elif len(split_types) != 1:
                split_type_buckets[3].append(file_name)
            elif list(split_types)[0] == "method_block":
                split_type_buckets[1].append(file_name)
            elif list(split_types)[0] == "window":
                split_type_buckets[2].append(file_name)
            elif list(split_types)[0] == "class_block":
                split_type_buckets[4].append(file_name)
            else:
                raise ValueError(f"Un-handleable split type: {split_types}")

            if dry_run is True:
                for prompt in all_prompts_T:
                    print("=" * 50)
                    print(prompt.value)
                continue

            # setting model to change temp and n over iterations
            total_generations = 0
            for n_idx, temp_idx in zip(n, temperature):

                # if (overwrite is False) and (query_output_dir / Path(f'{file_name.split(".")[0]}_{total_generations + n_idx - 1}' + file_extensions[lang])).exists():
                print("total_generations", total_generations, n_idx, temp_idx)
                if (overwrite is False) and os.path.exists(
                    os.path.join(
                        query_output_dir,
                        f'{file_name.split(".")[0]}_{total_generations + n_idx - 1}'
                        + file_extensions[lang],
                    )
                ):
                    log = f"Skipping file: {file_name} -> "
                    for i in range(n_idx):
                        log += f"{total_generations + i}" + file_extensions[lang] + ", "
                    logging.info(log)
                    total_generations += n_idx
                    continue


                model_config = {
                    "api_key": api_key,
                    "api_type": api_type,
                    "api_version": api_version,
                    "api_base": api_base,
                    "model": model,
                    "temperature": temp_idx,
                    "n": n_idx,
                    "retry_timeout": timeout,
                    "max_retry_attempts": max_attempts,
                    "system_message": system_message,
                    'max_tokens': max_tokens,
                }
                print(f"Model config: {model_config}")
                print(f"prompt_size: {count_tokens(all_prompts_T[0].value, model_name=model)[0]}")

                model_handler = OpenAIHandler(config=model_config)

                # Actually sending the prompt to OpenAI API and getting responses
                original_responses = model_handler.get_responses([p.value for p in all_prompts_T])
                post_process_responses = copy.deepcopy(original_responses)

                for i, responses in enumerate(original_responses):
                    indentation_level = all_prompts_T[i].metadata["indentation_level"]
                    for j, response in enumerate(responses):
                        response = sanitize_llm_response(response)
                        if (
                            all_prompts_T[i].metadata["source_example"][4]
                            == "method_block"
                        ):
                            response = (
                                post_process_adjust_indentation(
                                    indentation_level, response
                                )
                            )
                        post_process_responses[i][j] = response

                seq_responses = list(zip(*post_process_responses))  # 3*4

                for i, responses in enumerate(seq_responses):
                    file_name_new = file_name.split(".")[0]
                    output_idx = str(i + total_generations)
                    target_file_name = (
                        f"{file_name_new}_{output_idx}" + file_extensions[lang]
                    )
                    target_file_path = query_output_dir / target_file_name
                    Logger = Log(query_output_log_dir, f"{target_file_name}_logs")
                    record = {
                        "File": file_name,
                        "Query": query_id,
                        "Results": [
                            (p.value, r, p.metadata["source_example"])
                            for p, r in zip(all_prompts_T, responses)
                        ],
                    }
                    Logger.create_logs(record, model_config)

                    patch_fixer = Patcher()
                    lines_updated = copy.deepcopy(file_lines)
                    adjustment = 0

                    for prompt, response in zip(all_prompts_T, responses):
                        example = prompt.metadata["source_example"]
                        final_response_text = [response.text if response.success else "".join(file_lines[example[2][0] + adjustment : example[2][1] + 1 + adjustment])]  # type: ignore
                        lines_updated, adjustment_new = patch_fixer.stitch(
                            final_response_text,
                            lines_updated,
                            list(
                                range(
                                    example[2][0] + adjustment,
                                    example[2][1] + 1 + adjustment,
                                )
                            ),
                        )
                        adjustment += adjustment_new

                    with open(target_file_path, "w", encoding="utf-8") as f:
                        f.write("\n".join(lines_updated))

                total_generations += n_idx

        split_type_data[query_id] = split_type_buckets
    pickle.dump(split_type_data, open(f"{files_subset_pickle_name}", "wb"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--Exp_folder", "-e", type=str, required=True)
    parser.add_argument("--Target_folder", "-t", type=str, required=True)
    parser.add_argument("--result_csv_path", "-r", type=str, required=True)
    parser.add_argument(
        "--Queries",  # name on the CLI - drop the `--` for positional/required parameters
        nargs="+",  # 1 or more values expected => creates a list
        type=str,
    )

    parser.add_argument("--Index_folder", type=str, default="index_data")
    parser.add_argument(
        "--files_subset_pickle_name", "-s", type=str, default="files_subset_2000.pkl"
    )

    parser.add_argument("--encoding", type=str, default="UTF-8")
    parser.add_argument("--timeout", type=int, default=64)
    parser.add_argument("--max_attempts", "-a", type=int, default=5)
    parser.add_argument("--log", type=str, default="WARNING")
    parser.add_argument("--timeout_mf", type=int, default=2)
    parser.add_argument("--include_related_lines", type=bool, default=True)
    parser.add_argument("--max_prompt_size", type=int, default=4000)
    parser.add_argument("--extra_buffer", type=int, default=200)
    parser.add_argument("--localisation_flag", type=bool, default=False)
    parser.add_argument("--basic_strategy", type=str, default="no_localisation")
    parser.add_argument("--localisation_strategy", type=str, default="generic_context")
    parser.add_argument("--deduplicate", type=str, default="partial")
    parser.add_argument("--max_fs_examples", type=int, default=5)
    parser.add_argument("--examples_per_prompt", type=int, default=1)
    parser.add_argument("--example_diff_type", type=str, default="patch")
    parser.add_argument("--index_type", type=str, default="window")
    parser.add_argument("--few_shot_seed", type=int, default=42)
    parser.add_argument("--min_timeout", type=int, default=20)

    parser.add_argument("--model", type=str, default="gpt-3.5-turbo-16k-0613")
    parser.add_argument("--max_tokens", type=int, default=4000)
    parser.add_argument("--temperature", nargs="+", type=float, default=[0, 0.75, 1])
    parser.add_argument("--n", nargs="+", type=int, default=[1, 6, 3])
    parser.add_argument("--stop", type=str, default="```")
    parser.add_argument(
        "--system_message",
        type=str,
        default="""Assistant is an AI chatbot that helps developers perform code quality related tasks. In particular, it can help with modifying input code, as per the suggestions given by the developers. It should not modify the functionality of the input code in any way. It keeps the changes minimal to ensure that the warning or bug is fixed. It does not introduce any performance improvements, refactoring, rewrites of code, other than what is essential for the task.""",
    )

    parser.add_argument(
        "--experiment", type=str, default="basic"
    )  # to replicate paper results hardcode experiment
    parser.add_argument(
        "--basic_template", type=str, default="templateA_lines.j2"
    )  # adding flexibility in use of templates
    parser.add_argument("--template", type=str, default="line-error")
    parser.add_argument("--dry_run", dest="dry_run", action="store_true")
    parser.add_argument("--overwrite", dest="overwrite", action="store_true")
    parser.add_argument("--language", type=str, default="python")
    parser.add_argument(
        "--queries_meta_file", type=str, default="metadata/python/metadata.json"
    )
    parser.add_argument(
        "--template_file",
        type=str,
        default="templates/proposer_template.j2",
    )

    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log.upper()))

    fix(
        Exp_folder=args.Exp_folder,
        Target_folder=args.Target_folder,
        result_csv_path=args.result_csv_path,
        files_subset_pickle_name=args.files_subset_pickle_name,
        timeout_mf=args.timeout_mf,
        timeout=args.timeout,
        minm_timeout=args.min_timeout,
        max_attempts=args.max_attempts,
        localisation_strategy=args.localisation_strategy,
        deduplicate=args.deduplicate,
        localisation_flag=args.localisation_flag,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        stop=args.stop,
        n=args.n,
        system_message=args.system_message,
        Queries=args.Queries,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        lang=args.language,
        queries_meta_file=args.queries_meta_file,
        template_file=args.template_file,
    )
