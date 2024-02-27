#!/bin/bash
mkdir codeql-home

wget https://github.com/github/codeql-cli-binaries/releases/download/v2.5.0/codeql.zip -P codeql-home/
unzip codeql-home/codeql.zip -d codeql-home/

git clone https://github.com/github/codeql.git codeql-home/codeql-repo
cd codeql-home/codeql-repo
git checkout 20416ae0342c66aa05bc099af8e5a020b018a978

echo 'export PATH="$(pwd)/codeql-home/codeql:$PATH"' >> ~/.bashrc 
source ~/.bashrc

codeql resolve languages
codeql resolve qlpacks