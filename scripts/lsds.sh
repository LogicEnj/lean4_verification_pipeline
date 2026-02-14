PATH_TO_DATASETS=$1
cd $PATH_TO_DATASETS
find . -name "*.json" -type f -exec sh -c 'echo -n "$1: "; wc -l < "$1"' sh {} \;
cd -
