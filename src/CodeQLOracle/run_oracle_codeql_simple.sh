#!/bin/bash
declare query=$1
declare file_name=$2
declare exp_folder=$3
declare codeql_db_path=$4
declare codeql_result_path=$5
declare codeql_pack_path=$6

declare new_dir="$exp_folder/$file_name"
declare db_path="$codeql_db_path/$file_name"
declare result_path="$codeql_result_path/$file_name"
declare outputFileName="$result_path/results_$file_name.csv"

codeql database create --language=python --source-root $new_dir $db_path/python-database
codeql database analyze --rerun $db_path/python-database $codeql_pack_path/$query --format=csv --output=$outputFileName --threads=0

echo "--------------------------------------------------"