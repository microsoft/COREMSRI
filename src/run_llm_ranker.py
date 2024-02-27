import os
import logging
import argparse
from jinja2 import Environment
from pathlib import Path
import json
from tqdm import tqdm

from dotenv import load_dotenv

load_dotenv()

from openai_handler import OpenAIHandler
from log import Log
from Utils.file_handling import count_tokens

_env = Environment()


def verify_files(
    diffs_folder,
    queries,
    metadata_file,
    prompt_template_file,
    model,
    system_message,
    max_tokens,
    temperature,
    stop,
    n,
    timeout_mf,
    timeout,
    max_attempts,
    output_log_dir,
    dry_run,
    overwrite,
):

    assert os.path.exists(diffs_folder)

    # Verify the generated file using LLMs
    model_config = {
        "api_key": os.environ["OPENAI_API_KEY"],
        "api_base": os.environ["OPENAI_API_BASE"] if "OPENAI_API_BASE" in os.environ else None,
        "api_type": os.environ['OPENAI_API_TYPE'] if "OPENAI_API_TYPE" in os.environ else 'openai',
        "api_version": os.environ["OPENAI_API_VERSION"] if "OPENAI_API_TYPE" in os.environ else None,
        "system_message": system_message,
        "retry_timeout": timeout,
        "retry_max_attempts": max_attempts,
        "model": model,
        "temperature": temperature,
        "n": n,
    }
    model_handler = OpenAIHandler(config=model_config)

    # load files
    prompt_template = open(prompt_template_file, "r").read()

    with open(metadata_file, "r") as f:
        query_metadata = json.load(f)

    query_list = queries if queries is not None else query_metadata.keys()
    for query_id in tqdm(query_list):
        query_data = query_metadata[query_id]

        query_folderName = query_data["folder_name"]
        assert os.path.exists(os.path.join(args.diffs_folder, query_folderName))

        prompt_diffs = []
        for file in os.listdir(f"{diffs_folder}/{query_folderName}"):
            if file.endswith(".diff"):
                prompt_diffs.append(file)

        os.makedirs(f"{output_log_dir}/{query_folderName}", exist_ok=True)
        prompt_diffs.sort()
        print("Query: {}    prompts: {}".format(query_id, len(prompt_diffs)))
        logging.info("Query: {}    prompts: {}".format(query_id, len(prompt_diffs)))
        for diff_file in tqdm(prompt_diffs):

            query_output_dir = f"{output_log_dir}/{query_folderName}"

            Logger = Log(query_output_dir, diff_file.split(".")[0] + "_logs")

            with open(
                os.path.join(args.diffs_folder, query_folderName, diff_file), "r"
            ) as f:
                diff = f.read()

            prompt_args = {
                "query_name": query_id,
                "Ruleset": query_data["ruleset"],
                "description": query_data["desc"],
                "fixed_code_diff": diff,
                "chain_of_thought": True,
                "evaluation_recommendations": (
                    query_data["recm"] if "recm" in query_data else None
                ),
            }

            # prompt creation
            template = _env.from_string(prompt_template)
            try:
                prompt = template.render(**prompt_args)
            except Exception as e:
                print(f"Error in rendering prompt template: {e}")
                return None

            # logging prompt data

            print(
                f"Diff: {diff_file} Prompt-len: {count_tokens(prompt, model_name=model)[0]}"
            )
            logging.info(
                f"Diff: {diff_file} Prompt-len: {count_tokens(prompt, model_name=model)[0]}"
            )

            assert type(prompt) == str

            # call model
            if dry_run is True:
                continue
            if overwrite is False:
                if os.path.exists(
                    os.path.join(
                        query_output_dir, diff_file.split(".")[0] + "_logs.log"
                    )
                ):
                    print(
                        "Skipping: {} Prompt-len: {}".format(
                            diff_file, count_tokens(prompt, model_name=model)[0]
                        )
                    )
                    continue

            response = model_handler.get_responses([prompt])

            try:
                record = {
                    "File": diff_file,
                    "Query": query_id,
                    "Results": [(prompt, response[0][0], "")],
                }
                Logger.create_logs(record, model_config)
            except Exception as e:
                logging.warning(
                    f"Error while trying to save results (Bad Response from LLM) : {e}"
                )
                print(f"Error while trying to save results (Bad Response from LLM) : {e}")
                continue


    return None


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--diffs_folder", type=str, required=True)
    parser.add_argument("--Queries", nargs="+", help="CodeQL Queries to run")
    parser.add_argument("-m", "--metadata_file", type=str, help="Queries Metadata json file", default="metadata/python/metadata.json")
    parser.add_argument(
        "--prompt_template_file",
        type=str,
        help="Prompt template jinja file",
        default="templates/ranker_template.j2"
    )
    parser.add_argument("--model", type=str, help="Model to use", default="gpt-4-0613")
    parser.add_argument(
        "--system_message",
        type=str,
        help="System message to use",
        default="""Assistant is an AI chatbot that helps developers perform code quality related tasks. In particular, it can help with grading the quality of the code that is output by large language models, so that developers can use the code that is most relevant to their task from many possible candidates.""",
    )
    parser.add_argument(
        "--max_tokens", type=int, help="Max tokens to use", default=1000
    )
    parser.add_argument(
        "--temperature", type=float, help="Temperature to use", default=0.0
    )
    parser.add_argument("--stop", type=str, help="Stop to use", default=None)
    parser.add_argument("--n", type=int, help="N to use", default=1)
    parser.add_argument("--timeout", type=int, help="Timeout to use", default=8)
    parser.add_argument(
        "--max_attempts", type=int, help="Max attempts to use", default=5
    )
    parser.add_argument("--timeout_mf", type=int, help="Timeout to use", default=2)
    parser.add_argument(
        "-o", "--output_log_dir", type=str, help="Output log directory", required=True
    )

    parser.add_argument("--dry_run", dest="dry_run", action="store_true")
    parser.add_argument("--overwrite", dest="overwrite", action="store_true")

    args = parser.parse_args()

    # check environment variables
    if (
        "OPENAI_API_KEY" not in os.environ
    ):
        raise Exception(
            "Environment variables not set. Please set OPENAI_API_KEY"
        )

    verify_files(
        diffs_folder=args.diffs_folder,
        queries=args.Queries,
        metadata_file=args.metadata_file,
        prompt_template_file=args.prompt_template_file,
        model=args.model,
        system_message=args.system_message,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        stop=args.stop,
        n=args.n,
        timeout_mf=args.timeout_mf,
        max_attempts=args.max_attempts,
        timeout=args.timeout,
        output_log_dir=args.output_log_dir,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )
