from jinja2 import Environment
from dataclasses import dataclass, field
from typing import Any, Callable

import re


@dataclass
class Prompt:
    value: str
    metadata: dict[str, Any] = field(default_factory=dict)


def get_empty_prompt(template, **kwargs):
    _env = Environment()
    template = _env.from_string(template)
    empty_prompt = template.render(
        buggy_code="",
        **kwargs,
    )
    return empty_prompt


def simple_combination(issues):
    final_issues = []
    for issue in issues:
        found_subset = False
        for j, final_issue in enumerate(final_issues):
            if final_issue[4] in ["whole_file", "class_block", "method_block"]:
                if set(range(final_issue[2][0], final_issue[2][1] + 1)).issubset(
                    set(range(issue[2][0], issue[2][1] + 1))
                ):
                    final_issues[j] = [
                        issue[0],
                        issue[1] + final_issue[1],
                        issue[2],
                        issue[3] + final_issue[3],
                        issue[4],
                    ]
                    # print(f"found subset: {final_issue[2]} subset of {issue[2]}")
                    found_subset = True
                    break
                elif set(range(issue[2][0], issue[2][1] + 1)).issubset(
                    set(range(final_issue[2][0], final_issue[2][1] + 1))
                ):
                    final_issues[j] = [
                        final_issue[0],
                        final_issue[1] + issue[1],
                        final_issue[2],
                        final_issue[3] + issue[3],
                        final_issue[4],
                    ]
                    # print(f"found subset: {issue[2]} subset of {final_issue[2]}")
                    found_subset = True
                    break
        if not found_subset:
            final_issues.append(issue)
    return final_issues


def find_indentation_and_adjust(final_examples_to_fix):

    # initialize an empty list to store the indentation level
    indentation = []

    for j, example in enumerate(final_examples_to_fix):
        code_block, answer_spans, code_block_indices, line_of_interest_list, flag = (
            example
        )
        if flag == "method_block":
            # split the code_block into lines
            lines = code_block.split("\n")
            # loop through the lines
            for line in lines:
                # if the line starts with def, find the number of spaces before it and append it to the list
                if line.lstrip().startswith("def"):
                    index = line.find("def")
                    indentation.append(line[:index])
                    # break the loop as we only need the first def statement
                    break
            # if the list is not empty, meaning we found a def statement
            if indentation:
                # loop through the lines again
                for i, line in enumerate(lines):
                    # if the line has the same or more indentation as the def statement, remove the indentation from the line
                    if line.startswith(indentation[0]):
                        lines[i] = line[len(indentation[0]) :]
            # join the lines back into a code_block
            new_code_block = "\n".join(lines)
            final_examples_to_fix[j][0] = new_code_block
        else:
            indentation.append("")
        # return the indentation list and the new code_block
    return final_examples_to_fix, indentation


class PromptConstructor:
    def __init__(
        self,
        template: dict,
        values: dict,
        strategy_variables: dict,
        splitter: Callable,
        token_counter: Callable,
        max_length: int = 8000,
        buffer: int = 100,
    ):
        self.template = template
        self.values = values
        self.strategy_variables = strategy_variables
        self.token_counter = token_counter
        self.max_length = max_length
        self.splitter = splitter
        self.buffer = buffer

        self._env = Environment()

    def __get_localization_arguments(self):
        neural_localization = False
        generic_context = False
        if self.strategy_variables["localisation_strategy"] == "neural_localization":
            neural_localization = True
        elif self.strategy_variables["localisation_strategy"] == "generic_context":
            generic_context = True

        return (
            neural_localization,
            generic_context,
            self.strategy_variables["deduplicate"],
        )

    def __sanitize_codeql_message(self, message: str) -> str:
        # sanitize_pattern = r"\|\"\"(.)+:\/\/\/(.)*(:\d+){4}\"\""
        sanitize_pattern = r"\[\[(.)+?\]\]"

        matches = re.finditer(sanitize_pattern, message, re.MULTILINE)
        for _, match in enumerate(matches, start=1):
            # message = message.replace(match.group(), '')
            message = message.replace(match.group().split("|")[1], "")
        message = (
            message.replace("|", " ")
            .replace("\n", " ")
            .replace("[[", "")
            .replace("]]", ",")
            .replace(";", "")
            .replace("  ", " ")
        )

        return message

    def construct(
        self,
        file_contents: str,
        answer_span_locs: list[tuple[int, int]],
        line_of_interest_namedtuple_map: dict,
        **kwargs,
    ) -> list[Prompt]:
        empty_prompt = get_empty_prompt(self.template["basic"], **kwargs)
        empty_prompt_len, _ = self.token_counter(empty_prompt)
        remaining = self.max_length - empty_prompt_len
        if remaining < 0:
            raise ValueError(
                f"Context length exceeded. Max allowed context length is {self.max_length} tokens."
            )

        file_lines = file_contents.splitlines()
        answer_spans_list = [
            (loc, "\n".join(file_lines[loc[0] - 1 : loc[1]]))
            for loc in answer_span_locs
        ]

        examples_to_fix = []

        for i, (span_loc, answer_span) in enumerate(answer_spans_list):
            # Call the python_splitter function here
            # ans[0] is the start location of line of interest
            line_of_interest = span_loc[0] - 1

            buffer = empty_prompt_len + self.buffer
            code_block, adjusted_line_of_interest, code_block_indices, flag = (
                self.splitter(
                    content=file_contents,
                    line_of_interest=line_of_interest,
                    buffer=buffer,
                    token_counter=lambda s: self.token_counter(s)[0],
                )
            )

            examples_to_fix.append(
                [
                    str(code_block),
                    [answer_span],
                    list(code_block_indices),
                    [line_of_interest],
                    flag,
                ]
            )

        # final_examples_to_fix = process_examples_to_fix(examples_to_fix)
        final_examples_to_fix = simple_combination(examples_to_fix)
        final_examples_to_fix, indentation = find_indentation_and_adjust(
            final_examples_to_fix
        )

        template = self._env.from_string(self.template["basic"])
        prompts = []
        for i, example in enumerate(final_examples_to_fix):
            extra_info_per_prompt = {}
            if (
                self.strategy_variables["recommendation_flag"]
                and example[4] != "whole_file"
            ):
                res_list = []
                error_msg_list = []
                for line_of_interest in example[3]:
                    res_list.append(line_of_interest_namedtuple_map[line_of_interest])
                    error_msg_list.append(
                        self.__sanitize_codeql_message(
                            line_of_interest_namedtuple_map[line_of_interest].message
                        )
                    )

                neural_localization, generic_context, deduplicate = (
                    self.__get_localization_arguments()
                )
                ctxt_blocks_for_result_location = self.strategy_variables[
                    "localise_obj"
                ].localize_helper(
                    query_name=self.values["query_name"],
                    program_content=file_contents,
                    result_locations=res_list,
                    neural_localization=neural_localization,
                    generic_context=generic_context,
                    deduplicate=deduplicate,
                )
                context_blocks = ""

                for ctxt_as_pair in ctxt_blocks_for_result_location:
                    buggy_lines = set(
                        [
                            i
                            for rl in ctxt_as_pair[1]
                            for i in range(rl.start_line, rl.end_line + 1)
                        ]
                    )
                    for ctxt_block in ctxt_as_pair[0]:
                        block_lines = set(
                            [
                                i
                                for i in range(
                                    ctxt_block.start_line, ctxt_block.end_line + 1
                                )
                            ]
                            if not ctxt_block.other_lines
                            else ctxt_block.other_lines
                        )
                        if not block_lines.intersection(buggy_lines):
                            newline_removed_content = "\n".join(
                                line for line in ctxt_block.content.split("\n") if line
                            )
                            context_blocks += newline_removed_content
                            context_blocks += "\n\n"

                extra_info_per_prompt["error_message"] = error_msg_list
                extra_info_per_prompt["relevant_code"] = context_blocks

                prompt_str = template.render(
                    buggy_code=example[0],
                    answer_spans=example[1],
                    **self.values,
                    **extra_info_per_prompt,
                    **kwargs,
                )

            else:
                extra_info_per_prompt = {}
                error_msg_list = []

                for line_of_interest in example[3]:
                    error_msg_list.append(
                        self.__sanitize_codeql_message(
                            line_of_interest_namedtuple_map[line_of_interest].message
                        )
                    )

                extra_info_per_prompt["error_message"] = error_msg_list

                # to avoid falling back FALLBACK schema in no_localization line-error
                # if self.strategy_variables["no_localisation_strategy"] == "line-error":
                #     template = self._env.from_string(self.template["more_info"])

                prompt_str = template.render(
                    buggy_code=example[0],
                    answer_spans=example[1],
                    **self.values,
                    **extra_info_per_prompt,
                    **kwargs,
                )

            prompts.append(
                Prompt(
                    value=prompt_str,
                    metadata={
                        "source_example": example,
                        "indentation_level": indentation[i],
                        "split_type": flag,
                    },
                )
            )
        return prompts
