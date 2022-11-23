#!/bin/sh
set -e
URL="https://data.mendeley.com/public-files/datasets/rscbjbr9sj/files/f12eaf6d-6023-432f-acc9-80c9d7393433/file_downloaded"

mkdir -p data
wget "${URL}" --output-document data/ChestXRay2017.zip
unzip data/ChestXRay2017.zip -d data
