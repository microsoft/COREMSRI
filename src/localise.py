import pandas as pd
import math
import re
from collections import namedtuple
from basecontexts import (
    ContextRetrievalError,
    BaseContexts,
    CLASS_FUNCTION,
    CLASS_OTHER,
    STUB,
)
from Utils.file_handling import python_parser

Span = namedtuple("Span", "type start_line start_column end_line end_column")
result_location = namedtuple(
    "result_location",
    [
        "start_line",
        "start_column",
        "end_line",
        "end_column",
        "supporting_fact_locations",
        "message",
    ],
)
sf_location = namedtuple(
    "sf_location", ["start_line", "start_column", "end_line", "end_column"]
)


class Localise:
    def __init__(self):

        self.__columns__ = [
            "CodeQL Vulnerability",
            "Vulnerability Desc",
            "Warning/Error",
            "CodeQL Output",
            "File",
            "StartLine",
            "StartChar",
            "EndLine",
            "EndChar",
        ]
        # ["Name", "Description", "Severity", "Message",
        #             "Path", "Start_line", "Start_column",
        #             "End_line", "End_column"]

    def partial_overlap(self, set1, set2):
        for x in set1:
            for y in set2:
                if x == y:
                    return True
        return False

    def get_combined_results(self, res_list, result_location):
        for x in res_list:
            if x == result_location:
                return res_list

        res_list.append(result_location)
        return res_list

    def are_same_block(self, context_block, supporting_fact_block):
        return (
            context_block.start_line == supporting_fact_block.start_line
            and context_block.end_line == supporting_fact_block.end_line
            and context_block.other_lines == supporting_fact_block.other_lines
        )

    def deduplicate_partial_overlap(
        self, result_location, blocks_and_metadata_list, ctxt_blocks_for_result_location
    ):
        indices_having_overlap = []
        for j, x in enumerate(ctxt_blocks_for_result_location):
            if self.partial_overlap(x[0], blocks_and_metadata_list):
                indices_having_overlap.append(j)

        if indices_having_overlap:
            updated_ctxt_blocks_for_result_location = []
            combined_blocks = []
            combined_result_locations = []
            for j, x in enumerate(ctxt_blocks_for_result_location):
                if j not in indices_having_overlap:
                    updated_ctxt_blocks_for_result_location.append(x)
                else:
                    for block in x[0]:
                        combined_blocks.append(block)
                    for rs in x[1]:
                        combined_result_locations = self.get_combined_results(
                            combined_result_locations, rs
                        )
            # add current block and result
            for block in blocks_and_metadata_list:
                combined_blocks.append(block)
            combined_result_locations.append(result_location)
            # dedup
            deduplicated_combined_blocks = []
            for i in range(len(combined_blocks)):
                duplicate_present = False
                for j in range(len(combined_blocks)):
                    if self.are_same_block(combined_blocks[i], combined_blocks[j]) and (
                        combined_blocks[i].relevant != combined_blocks[j].relevant
                    ):
                        duplicate_present = True
                        combined_blocks[i].relevant = True
                        combined_blocks[j].relevant = True
                    elif combined_blocks[i] == combined_blocks[j]:
                        duplicate_present = True

                    if duplicate_present:
                        break
                if combined_blocks[i] not in deduplicated_combined_blocks:
                    deduplicated_combined_blocks.append(combined_blocks[i])
            # sort
            deduplicated_combined_blocks = sorted(
                deduplicated_combined_blocks, key=lambda x: (x.start_line, x.end_line)
            )

            updated_ctxt_blocks_for_result_location.append(
                (deduplicated_combined_blocks, combined_result_locations)
            )
        else:
            updated_ctxt_blocks_for_result_location = ctxt_blocks_for_result_location
            updated_ctxt_blocks_for_result_location.append(
                (blocks_and_metadata_list, [result_location])
            )

        return updated_ctxt_blocks_for_result_location

    def deduplicate_complete_overlap(
        self, result_location, blocks_and_metadata_list, ctxt_blocks_for_result_location
    ):
        updated_ctxt_blocks_for_result_location = []
        already_exist = False
        for x in ctxt_blocks_for_result_location:
            if x[0] == blocks_and_metadata_list:
                temp_result_set = self.get_combined_results(x[1], result_location)
                updated_ctxt_blocks_for_result_location.append((x[0], temp_result_set))
                already_exist = True
            else:
                updated_ctxt_blocks_for_result_location.append(x)
        if not already_exist:
            updated_ctxt_blocks_for_result_location.append(
                (blocks_and_metadata_list, [result_location])
            )

        return updated_ctxt_blocks_for_result_location

    def supporting_fact_overlaps(self, supporting_fact, result_span):
        """
        Whether supporting_fact overlaps result_span.
        Args:
            supporting_fact: supoortiung fact span
            result_span: result_span span
        Returns:
            True/False
        """
        if (
            supporting_fact.end_line >= result_span.start_line
            and supporting_fact.end_line <= result_span.end_line
        ) or (
            supporting_fact.start_line >= result_span.start_line
            and supporting_fact.start_line <= result_span.end_line
        ):
            if (
                supporting_fact.end_line == result_span.start_line
                and supporting_fact.end_column < result_span.start_column
            ):
                return False
            elif (
                supporting_fact.start_line == result_span.end_line
                and supporting_fact.start_column > result_span.end_column
            ):
                return False
            else:
                return True

        return False

    def get_result_locations(self, result_row, positive_or_negative_examples):
        """
        This function returns result location along with
        set of non-overlapping supporting facts.
        Args:
            result_row: specific row of CodeQL result csv.
        Returns:
            A set of non-overlapping supporting fact and spans.
        """
        supporting_facts = set()
        nonoverlapping_supporting_facts = []
        # Sometimes the results don't mention proper start/end line/
        # column. The following check helps avoid errors arising out of
        # this issue.
        if math.isnan(result_row.StartLine) is True:
            return
        start_line = int(result_row.StartLine)
        if math.isnan(result_row.EndLine) is True:
            return
        end_line = int(result_row.EndLine)
        if math.isnan(result_row.StartChar) is True:
            return
        start_column = int(result_row.StartChar)
        if math.isnan(result_row.EndChar) is True:
            return
        end_column = int(result_row.EndChar)

        # add the result span
        result_span = Span("RESULT", start_line, start_column, end_line, end_column)

        # get supporting fact spans
        if positive_or_negative_examples == "positive":
            matches = re.findall(
                r"relative:\/\/\/[a-zA-Z0-9_.]*:(\d+):(\d+):(\d+):(\d+)",
                str(result_row.ErrorOutput),
            )
        else:
            matches = []

        for match in matches:
            if len(match) != 4:
                continue
            elif (int(match[2]) - int(match[0]) == 0) and (
                int(match[3]) - int(match[1]) == 0
            ):
                # to ignore built-in spans like 0:0:0:0
                continue
            try:
                supporting_fact = Span(
                    "SUPPORTING_FACT",
                    int(match[0]),
                    int(match[1]),
                    int(match[2]),
                    int(match[3]),
                )
                supporting_facts.add(supporting_fact)
            except ValueError:
                # if regex capture something similar to span
                # but not a span
                continue
        # check overlap with result span
        for sf in supporting_facts:
            if not (
                self.supporting_fact_overlaps(sf, result_span)
                or self.supporting_fact_overlaps(result_span, sf)
            ):
                nonoverlapping_supporting_facts.append(sf)

        # print(start_line, start_column, end_line, end_column, nonoverlapping_supporting_facts)
        res_result_location = result_location(
            start_line,
            start_column,
            end_line,
            end_column,
            nonoverlapping_supporting_facts,
            str(result_row.ErrorOutput),
        )

        return res_result_location

    def get_context_with_codeql_message(self, program_content, result_location):
        def are_same_block(context_block, supporting_fact_block):
            return (
                context_block.start_line == supporting_fact_block.start_line
                and context_block.end_line == supporting_fact_block.end_line
                and context_block.other_lines == supporting_fact_block.other_lines
            )

        # main body
        context_object = BaseContexts(python_parser)

        blocks_and_metadata_list = [
            context_object.get_local_block(
                program_content,
                result_location.start_line - 1,
                result_location.end_line - 1,
            )
        ]
        for sf in result_location.supporting_fact_locations:
            blocks_and_metadata_list.append(
                context_object.get_local_block(
                    program_content, sf.start_line - 1, sf.end_line - 1
                )
            )

        all_blocks = context_object.get_all_blocks(program_content)
        all_blocks = sorted(all_blocks, key=lambda x: (x.start_line, x.end_line))
        for block in all_blocks:
            # to ensure deduplication
            block.relevant = True

        initial_blocks_and_metadata_list = blocks_and_metadata_list.copy()
        for block in initial_blocks_and_metadata_list:
            # to ensure deduplication
            block.relevant = True
            if block.block_type == CLASS_FUNCTION:
                local_class = context_object.get_local_class(
                    program_content, block.start_line, block.end_line
                )
                for block_i in all_blocks:
                    if (
                        block_i not in blocks_and_metadata_list
                        and block_i.start_line >= local_class.start_line
                        and block_i.end_line <= local_class.end_line
                        and block_i.block_type in [CLASS_FUNCTION, CLASS_OTHER]
                    ):
                        if block_i.block_type == CLASS_OTHER:
                            blocks_and_metadata_list.append(block_i)
                        elif (
                            block_i.block_type == CLASS_FUNCTION
                            and block_i.metadata.split(".")[-1] == "__init__"
                        ):
                            blocks_and_metadata_list.append(block_i)
            elif block.block_type == CLASS_OTHER:
                local_class = context_object.get_local_class(
                    program_content, block.start_line, block.end_line
                )
                for block_i in all_blocks:
                    if (
                        block_i not in blocks_and_metadata_list
                        and block_i.start_line >= local_class.start_line
                        and block_i.end_line <= local_class.end_line
                        and block_i.block_type == CLASS_FUNCTION
                        and block_i.metadata.split(".")[-1] == "__init__"
                    ):
                        blocks_and_metadata_list.append(block_i)
        blocks_and_metadata_list = sorted(
            blocks_and_metadata_list, key=lambda x: (x.start_line, x.end_line)
        )

        # deduplicate wrt to relevant
        deduplicated_blocks_and_metadata_list = []
        for i in range(len(blocks_and_metadata_list)):
            duplicate_present = False
            for j in range(len(blocks_and_metadata_list)):
                if are_same_block(
                    blocks_and_metadata_list[i], blocks_and_metadata_list[j]
                ) and (
                    blocks_and_metadata_list[i].relevant
                    != blocks_and_metadata_list[j].relevant
                ):
                    duplicate_present = True
                    blocks_and_metadata_list[i].relevant = True
                    blocks_and_metadata_list[j].relevant = True
                elif blocks_and_metadata_list[i] == blocks_and_metadata_list[j]:
                    duplicate_present = True

                if duplicate_present:
                    break
            if blocks_and_metadata_list[i] not in deduplicated_blocks_and_metadata_list:
                deduplicated_blocks_and_metadata_list.append(
                    blocks_and_metadata_list[i]
                )

        return deduplicated_blocks_and_metadata_list

    def localize_helper(
        self,
        query_name,
        program_content,
        tree_sitter_parser=python_parser,
        source_code_file_path=None,
        message=None,
        result_locations=None,
        aux_result_path=None,
        neural_localization=False,
        generic_context=False,
        deduplicate="partial",
    ):
        """
        This is helper function to get query specific context/Block given
        program_content, start_line, end_line and name of the query.
        Args:
            name : name of the query
            program_content: Program in string format
            parser : tree_sitter_parser
            file_path : file of program_content (filepath in CodeQueires dataset)
            message : CodeQL message
            result_locations: list of named_tuple with fileds (start_line, end_line, start_column, end_column, supporting_fact_locations)
                        supporting_fact_locations can be a list of named_tuples with fileds (start_line, end_line, start_column, end_column)
            aux_result_path: auxiliary query results path
            neural_localization: boolean flag that specifies whether to use neural approach for localization. Defaults to False.
            generic_context: boolean flag that determines whether to use generic context localization process. Defaults to False.
                            neural_localization should be set to False.
            deduplicate: `deduplicate` is a parameter that controls how the localization results are handled in case of multiple result
                        locations are given for a (query, file) pair. Possible values are `complete` amd `partial`. Defaults to `partial`.
        Returns:
            A list consisting context Blocks with relevance label
        """
        if not neural_localization and not generic_context:
            if aux_result_path:
                aux_result_df = pd.read_csv(aux_result_path, names=self.__columns__)
            else:
                aux_result_df = None
        elif neural_localization:
            all_blocks = self.get_all_blocks(program_content)

        try:
            ctxt_blocks_for_result_location = []
            for _, result_location in enumerate(result_locations):
                # if neural_localization:
                #     blocks_and_metadata_list = get_predicted_relevance_blocks(all_blocks, query_name)

                #     # TODO: verify whether we want this.
                #     ctxt_blocks_for_result_location.append((blocks_and_metadata_list, []))
                # else:
                if generic_context:
                    blocks_and_metadata_list = self.get_context_with_codeql_message(
                        program_content, result_location
                    )
                # else:
                #     blocks_and_metadata_list = get_all_context(query_name, program_content,
                #                                                 tree_sitter_parser,
                #                                                 source_code_file_path,
                #                                                 message, result_location,
                #                                                 aux_result_df)
                # remove STUB blocks
                blocks_and_metadata_list = [
                    block
                    for block in blocks_and_metadata_list
                    if block.block_type != STUB
                ]
                # deduplicate blocks
                if deduplicate == "complete":
                    ctxt_blocks_for_result_location = self.deduplicate_complete_overlap(
                        result_location,
                        blocks_and_metadata_list,
                        ctxt_blocks_for_result_location,
                    )
                elif deduplicate == "partial":
                    ctxt_blocks_for_result_location = self.deduplicate_partial_overlap(
                        result_location,
                        blocks_and_metadata_list,
                        ctxt_blocks_for_result_location,
                    )

            return ctxt_blocks_for_result_location
        except ContextRetrievalError as error:
            error_att = error.args[0]
            print("msg: ", error_att["message"], "\n", "type: ", error_att["type"])

    def get_all_blocks(self, program_content):
        """
        This is helper function to get all code blocks.
        Args:
            program_content: Program in string format
        Returns:
            A list consisting all code Blocks.
        """
        context_object = BaseContexts(python_parser)
        blocks_and_metadata_list = context_object.get_all_blocks(program_content)
        blocks_and_metadata_list = sorted(
            blocks_and_metadata_list, key=lambda x: (x.start_line, x.end_line)
        )

        return blocks_and_metadata_list
