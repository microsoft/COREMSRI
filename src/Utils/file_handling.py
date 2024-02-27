import os
import tempfile
import subprocess
from pathlib import Path
import copy
import re

from tree_sitter import Language, Parser

import tiktoken
from jinja2 import Environment
from collections import namedtuple
import logging


__FILE_PATH = Path(__file__).resolve()
__FILE_DIR = __FILE_PATH.parent

Language.build_library(
    str(__FILE_DIR / "build/my-languages.so"),
    [
        str(__FILE_DIR / "tree-sitter-python"),
        str(__FILE_DIR / "tree-sitter-java"),
    ],
)

python_parser = Parser()
python_parser.set_language(
    Language(str(__FILE_DIR / "build/my-languages.so"), "python")
)

java_parser = Parser()
java_parser.set_language(Language(str(__FILE_DIR / "build/my-languages.so"), "java"))


TOKENIZER_MODEL_ALIAS_MAP = {
    "gpt-4": "gpt-3.5-turbo",
    "gpt-35-turbo": "gpt-3.5-turbo",
}


def count_tokens(input_str: str, model_name: str) -> tuple[int, tiktoken.Encoding]:
    if model_name in TOKENIZER_MODEL_ALIAS_MAP:
        model_name = TOKENIZER_MODEL_ALIAS_MAP[model_name]
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(input_str))
    return num_tokens, encoding


class SimpleSplitter:
    def __init__(self, token_counter):
        self.token_counter = token_counter

    def __call__(self, content: str, chunk_lim: int):
        return split_by_line(content, self.token_counter, chunk_lim)


def split_python_file(content, line_of_interest, token_counter, chunk_lim, buffer):
    """
    It takes in a file content, and a line number of interest, and returns a code block, the line
    number of interest in the code block, and the line numbers of the code block in the original
    file
    Note: Total prompt length should be max_tokens//2 - buffer

    TODO: facilitate selection of different code blocks
    Current strategy: if file can be fit with model input limit, then use whole file as relevant block
                        else surrounding function if inside function
                                                else line_of_interest with a window on both sides
    Args:
        content: The content of the file that you want to parse.
        line_of_interest: The line number of the error in the original file (Note: line_of_interest is 0-indexed)
        buffer: max_tokens//2 - buffer//2 is the window size on both sides of the line of interest

    Returns:
        The code block, the adjusted line of interest, code block indices in the original file and the flag
    """
    num_tokens = token_counter(content)
    flag = "whole_file"
    if num_tokens <= (chunk_lim - buffer):
        return (
            content,
            line_of_interest,
            (0, len(content.split("\n")) - 1),
            flag,
        )

    # to handle large-sized files
    max_tokens_available = chunk_lim - buffer
    tree = python_parser.parse(bytes(content, "utf-8"))
    root_node = tree.root_node
    erroneous_code = root_node.type == "ERROR"

    all_method_nodes = []
    all_class_nodes = []

    def get_class_blocks(root_node, depth=0):
        if root_node is None:
            return

        if root_node.type in ["class_definition"]:
            all_class_nodes.append(root_node)
        else:
            for node in root_node.children:
                get_class_blocks(node, depth + 1)

    def get_method_blocks(root_node, depth=0):
        if root_node is None:
            return

        if root_node.type in ["function_definition"]:
            all_method_nodes.append(root_node)
        else:
            for node in root_node.children:
                get_method_blocks(node, depth + 1)

    # get all class nodes
    get_class_blocks(root_node)
    required_class_node = None
    for class_node in all_class_nodes:
        if (
            line_of_interest >= class_node.start_point[0]
            and line_of_interest <= class_node.end_point[0]
        ):
            required_class_node = class_node
            break

    file_lines = content.split("\n")
    if required_class_node is not None:
        code_block_indices = (
            required_class_node.start_point[0],
            required_class_node.end_point[0],
        )
        adjusted_line_of_interest = (
            line_of_interest - required_class_node.start_point[0]
        )
        code_block = ("\n").join(
            [
                line
                for i, line in enumerate(file_lines)
                if (i >= code_block_indices[0] and i <= code_block_indices[1])
            ]
        )
        num_block_tokens = token_counter(code_block)
        if num_block_tokens < max_tokens_available:
            flag = "class_block"
            return code_block, adjusted_line_of_interest, code_block_indices, flag

    # get all method nodes
    get_method_blocks(root_node)
    required_method_node = None
    for method_node in all_method_nodes:
        if (
            line_of_interest >= method_node.start_point[0]
            and line_of_interest <= method_node.end_point[0]
        ):
            required_method_node = method_node
            break

    file_lines = content.split("\n")
    if required_method_node is not None:
        code_block_indices = (
            required_method_node.start_point[0],
            required_method_node.end_point[0],
        )
        adjusted_line_of_interest = (
            line_of_interest - required_method_node.start_point[0]
        )
        code_block = ("\n").join(
            [
                line
                for i, line in enumerate(file_lines)
                if (i >= code_block_indices[0] and i <= code_block_indices[1])
            ]
        )
        num_block_tokens = token_counter(code_block)
        if num_block_tokens < max_tokens_available:
            flag = "method_block"
            return code_block, adjusted_line_of_interest, code_block_indices, flag

    # implicitly handles `erroneous_code==True` case and where method block is too large
    oneside_window_length = (
        max_tokens_available // 2 - token_counter(file_lines[line_of_interest]) // 2
    )
    code_block_index_top = line_of_interest
    code_block_index_bottom = line_of_interest

    # keep adding until chunk full
    # top window
    oneside_window_length_top = copy.deepcopy(oneside_window_length)
    while True:
        index = max(code_block_index_top - 1, -1)
        if index == -1:
            break
        line_token_count = token_counter(file_lines[index])
        if line_token_count >= oneside_window_length_top:
            break
        code_block_index_top = index
        oneside_window_length_top -= line_token_count

    # bottom window
    oneside_window_length_bottom = copy.deepcopy(oneside_window_length)
    while True:
        index = min(code_block_index_bottom + 1, len(file_lines))
        if index == len(file_lines):
            break
        line_token_count = token_counter(file_lines[index])
        if line_token_count >= oneside_window_length_bottom:
            break
        code_block_index_bottom = index
        oneside_window_length_bottom -= line_token_count

    adjusted_line_of_interest = max(line_of_interest - code_block_index_top, 0)
    code_block_indices = (code_block_index_top, code_block_index_bottom)
    code_block = ("\n").join(
        [
            line
            for i, line in enumerate(file_lines)
            if (i >= code_block_indices[0] and i <= code_block_indices[1])
        ]
    )
    flag = "window"
    return code_block, adjusted_line_of_interest, code_block_indices, flag


def split_java_file(content, line_of_interest, token_counter, chunk_lim, buffer):
    """
    It takes in a file content, and a line number of interest, and returns a code block, the line
    number of interest in the code block, and the line numbers of the code block in the original
    file
    Note: Total prompt length should be max_tokens//2 - buffer

    TODO: facilitate selection of different code blocks
    Current strategy: if file can be fit with model input limit, then use whole file as relevant block
                        else surrounding function if inside function
                                                else line_of_interest with a window on both sides
    Args:
        content: The content of the file that you want to parse.
        line_of_interest: The line number of the error in the original file (Note: line_of_interest is 0-indexed)
        buffer: max_tokens//2 - buffer//2 is the window size on both sides of the line of interest

    Returns:
        The code block, the adjusted line of interest, code block indices in the original file and the flag
    """
    num_tokens = token_counter(content)
    flag = "whole_file"
    if num_tokens <= (chunk_lim - buffer):
        return (
            content,
            line_of_interest,
            (0, len(content.split("\n")) - 1),
            flag,
        )

    # to handle large-sized files
    max_tokens_available = chunk_lim - buffer
    tree = java_parser.parse(bytes(content, "utf-8"))
    root_node = tree.root_node
    erroneous_code = root_node.type == "ERROR"

    all_method_nodes = []
    all_class_nodes = []

    def get_class_blocks(root_node, depth=0):
        if root_node is None:
            return

        if root_node.type in ["class_definition"]:
            all_class_nodes.append(root_node)
        else:
            for node in root_node.children:
                get_class_blocks(node, depth + 1)

    def get_method_blocks(root_node, depth=0):
        if root_node is None:
            return

        if root_node.type in ["function_definition"]:
            all_method_nodes.append(root_node)
        else:
            for node in root_node.children:
                get_method_blocks(node, depth + 1)

    # get all class nodes
    get_class_blocks(root_node)
    required_class_node = None
    for class_node in all_class_nodes:
        if (
            line_of_interest >= class_node.start_point[0]
            and line_of_interest <= class_node.end_point[0]
        ):
            required_class_node = class_node
            break

    file_lines = content.split("\n")
    if required_class_node is not None:
        code_block_indices = (
            required_class_node.start_point[0],
            required_class_node.end_point[0],
        )
        adjusted_line_of_interest = (
            line_of_interest - required_class_node.start_point[0]
        )
        code_block = ("\n").join(
            [
                line
                for i, line in enumerate(file_lines)
                if (i >= code_block_indices[0] and i <= code_block_indices[1])
            ]
        )
        num_block_tokens = token_counter(code_block)
        if num_block_tokens < max_tokens_available:
            flag = "class_block"
            return code_block, adjusted_line_of_interest, code_block_indices, flag

    # get all method nodes
    get_method_blocks(root_node)
    required_method_node = None
    for method_node in all_method_nodes:
        if (
            line_of_interest >= method_node.start_point[0]
            and line_of_interest <= method_node.end_point[0]
        ):
            required_method_node = method_node
            break

    file_lines = content.split("\n")
    if required_method_node is not None:
        code_block_indices = (
            required_method_node.start_point[0],
            required_method_node.end_point[0],
        )
        adjusted_line_of_interest = (
            line_of_interest - required_method_node.start_point[0]
        )
        code_block = ("\n").join(
            [
                line
                for i, line in enumerate(file_lines)
                if (i >= code_block_indices[0] and i <= code_block_indices[1])
            ]
        )
        num_block_tokens = token_counter(code_block)
        if num_block_tokens < max_tokens_available:
            flag = "method_block"
            return code_block, adjusted_line_of_interest, code_block_indices, flag

    # implicitly handles `erroneous_code==True` case and where method block is too large
    oneside_window_length = (
        max_tokens_available // 2 - token_counter(file_lines[line_of_interest]) // 2
    )
    code_block_index_top = line_of_interest
    code_block_index_bottom = line_of_interest

    # keep adding until chunk full
    # top window
    oneside_window_length_top = copy.deepcopy(oneside_window_length)
    while True:
        index = max(code_block_index_top - 1, -1)
        if index == -1:
            break
        line_token_count = token_counter(file_lines[index])
        if line_token_count >= oneside_window_length_top:
            break
        code_block_index_top = index
        oneside_window_length_top -= line_token_count

    # bottom window
    oneside_window_length_bottom = copy.deepcopy(oneside_window_length)
    while True:
        index = min(code_block_index_bottom + 1, len(file_lines))
        if index == len(file_lines):
            break
        line_token_count = token_counter(file_lines[index])
        if line_token_count >= oneside_window_length_bottom:
            break
        code_block_index_bottom = index
        oneside_window_length_bottom -= line_token_count

    adjusted_line_of_interest = max(line_of_interest - code_block_index_top, 0)
    code_block_indices = (code_block_index_top, code_block_index_bottom)
    code_block = ("\n").join(
        [
            line
            for i, line in enumerate(file_lines)
            if (i >= code_block_indices[0] and i <= code_block_indices[1])
        ]
    )
    flag = "window"
    return code_block, adjusted_line_of_interest, code_block_indices, flag


LANG_SPLITTER_MAP = {
    "python": split_python_file,
    "java": split_java_file,
}


class StructuredCodeSplitter:
    def __init__(self, lang: str, config: dict):
        self.lang = lang
        self.split_file_func = LANG_SPLITTER_MAP[lang]
        self.config = config
        # file_content, line_of_interest, encoding, python_parser, max_tokens, buffer -> for python

    def __call__(self, content, **kwargs):
        kwargs |= self.config
        return self.split_file_func(content, **kwargs)


def split_by_line(chunk: str, token_counter, chunk_lim) -> list[str]:
    lines = chunk.splitlines()
    chunks = []
    i = 0
    acc = ""
    _token_count = token_counter(acc)

    while True:
        curr_line = lines[i]

        if acc == "":
            _t_acc = curr_line
            _token_count = token_counter(curr_line)
        else:
            _t_acc = acc + "\n" + curr_line
            _token_count += token_counter("\n" + curr_line)

        if not _token_count <= chunk_lim:
            chunks.append(acc)
            acc = curr_line
            _token_count = token_counter(curr_line)
        else:
            acc = _t_acc

        i += 1

        if i >= len(lines):
            chunks.append(acc)
            break

    return chunks


def get_git_diff(file_content, edited_file_content):
    """
    It takes two strings, writes them to temporary files, and then runs `git diff` on those files.

    Args:
        file_content: The original file content
        edited_file_content: The content of the file after it has been edited.

    Returns:
        The diff between the two files.
    """
    tf1 = tempfile.NamedTemporaryFile(delete=False)
    tf1.write(file_content.encode("utf-8"))
    tf1.seek(0)

    tf2 = tempfile.NamedTemporaryFile(delete=False)
    tf2.write(edited_file_content.encode("utf-8"))
    tf2.seek(0)

    pp = subprocess.Popen(
        [
            "git",
            "diff",
            "--histogram",
            "--unified=0",
            "--no-index",
            tf1.name,
            tf2.name,
        ],
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    import pdb

    pdb.set_trace()
    git_diff = ("\n").join(str(line) for line in pp.stdout)

    # https://docs.python.org/2/library/tempfile.html#tempfile.mktemp
    tf1.close()
    tf2.close()
    os.unlink(tf1.name)
    os.unlink(tf2.name)

    return git_diff


GIT_ATTRIBUTES_FOR_LANG_SPECIFIC_DIFF = """*.cs\tdiff=csharp
*.py\tdiff=python"""


def get_file_diff(file_content, edited_file_content, function_context: bool = False):
    temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    attributes_file_path = Path(temp_dir.name) / "git/attributes"
    attributes_file_path.parent.mkdir(parents=True)
    attributes_file_path.write_text(GIT_ATTRIBUTES_FOR_LANG_SPECIFIC_DIFF)
    os.environ["XDG_CONFIG_HOME"] = str(temp_dir.name)

    diff_command = f"git --no-pager diff "

    if function_context:
        diff_command += "-U0 --function-context "

    with (
        tempfile.NamedTemporaryFile(mode="w+", delete=False, encoding="utf-8") as tf1,
        tempfile.NamedTemporaryFile(mode="w+", delete=False, encoding="utf-8") as tf2,
    ):
        tf1.write(file_content)
        tf1.seek(0)
        tf2.write(edited_file_content)
        tf2.seek(0)
        diff_command += f"--no-index --histogram --no-prefix {tf1.name} {tf2.name}"
        logging.info(diff_command)
        result = subprocess.run(
            diff_command,
            shell=True,
            text=True,
            capture_output=True,
            encoding="utf-8",
        )
        output = result.stdout

    temp_dir.cleanup()
    os.environ.pop("XDG_CONFIG_HOME", None)

    return output


def extract_block(file_content, line_of_interest, encoding, parser, max_tokens):
    """
    It takes in a file content, and a line number of interest, and returns a code block, the line
    number of interest in the code block, and the line numbers of the code block in the original
    file

    TODO: facilitate selection of different code blocks
    Current strategy: if file can be fit with model input limit, then use whole file as relevant block
                        else surrounding function if inside function
                                                else line_of_interest with a window on both sides
    Args:
        file_content: The content of the file that you want to parse.
        line_of_interest: The line number of the error in the original file.
        strategy: Strategy to select relevant code block.

    Returns:
        The code block, the adjusted line of interest, and the edited block.
    """
    num_tokens = len(encoding.encode(file_content))
    if num_tokens < max_tokens:
        return (
            file_content,
            line_of_interest,
            (0, len(file_content.split("\n")) - 1),
        )

    # to handle large-sized files
    tree = parser.parse(bytes(file_content, "utf8"))
    root_node = tree.root_node
    erroneous_code = root_node.type == "ERROR"

    all_method_nodes = []

    def get_method_blocks(root_node, depth=0):
        if root_node is None:
            return

        if root_node.type in ["method_declaration", "constructor_declaration"]:
            all_method_nodes.append(root_node)
        else:
            for node in root_node.children:
                get_method_blocks(node, depth + 1)

    # get all method nodes
    get_method_blocks(root_node)
    required_method_node = None
    for method_node in all_method_nodes:
        if (
            line_of_interest >= method_node.start_point[0]
            and line_of_interest <= method_node.end_point[0]
        ):
            required_method_node = method_node
            break

    file_lines = file_content.split("\n")
    if required_method_node is not None:
        code_block_indices = (
            required_method_node.start_point[0],
            required_method_node.end_point[0],
        )
        adjusted_line_of_interest = (
            line_of_interest - required_method_node.start_point[0]
        )
        code_block = ("\n").join(
            [
                line
                for i, line in enumerate(file_lines)
                if (i >= code_block_indices[0] and i <= code_block_indices[1])
            ]
        )
        num_block_tokens = len(encoding.encode(code_block))
        if num_block_tokens < max_tokens:
            return code_block, adjusted_line_of_interest, code_block_indices

    # implicitly handles `erroneous_code==True` case and where method block is too large
    oneside_window_length = 10
    code_block_indices = (
        max(line_of_interest - oneside_window_length, 0),
        min(line_of_interest + oneside_window_length, len(file_lines)),
    )
    adjusted_line_of_interest = max(line_of_interest - oneside_window_length, 0)
    code_block = ("\n").join(
        [
            line
            for i, line in enumerate(file_lines)
            if (i >= code_block_indices[0] and i <= code_block_indices[1])
        ]
    )

    return code_block, adjusted_line_of_interest, code_block_indices


def sanitize_llm_response(response):
    OpenAIResponse = namedtuple("OpenAIResponse", "text finish_reason success")

    response_str: str = response.text
    if response_str.startswith('```python'):
        response_str = response_str.removeprefix('```python\n')
    if response_str.endswith('\n'):
        response_str = response_str.removesuffix('\n')
    if response_str.endswith('```'):
        response_str = response_str.removesuffix('```')

    return OpenAIResponse(response_str, response.finish_reason, response.success)
    

def post_process_adjust_indentation(indentation_level, response):
    OpenAIResponse = namedtuple("OpenAIResponse", "text finish_reason success")

    response_str = response.text

    if response_str == None:
        return response
    # split the prompt_str into lines
    lines = response_str.split("\n")
    # loop through the lines
    for i, line in enumerate(lines):
        # Add indentation_level to the lin
        lines[i] = indentation_level + line
    # join the lines back into a code_block
    new_code_block = "\n".join(lines)
    # return the new code_block
    new_response = OpenAIResponse(
        new_code_block, response.finish_reason, response.success
    )
    return new_response


def replace_in_str(input_str: str, replacements: dict[str, str]) -> str:
    for key, value in replacements.items():
        input_str = input_str.replace(key, value)
    return input_str


def sanitize_names(name: str, replacement: str = "_") -> str:
    return re.sub("[^0-9a-zA-Z]+", replacement, name)


class SpanWindowExtractor:
    def __init__(self, window_size: int = 5):
        self.window_size = window_size

    def __call__(
        self,
        content: str,
        spans: list[tuple[int, int]],
    ) -> list[str]:
        """
        Extracts the window around the spans in the content.
        """
        lines = content.split("\n")
        windows = []
        for span in spans:
            start_line = max(0, span[0] - self.window_size)
            end_line = min(len(lines), span[1] + self.window_size)
            windows.append("\n".join(lines[start_line:end_line]))
        return windows
