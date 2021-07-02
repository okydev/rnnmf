#!/bin/bash

source ./env/bin/activate
export PYTHONUNBUFFERED=1
# Preprocess
# python ./run.py \
# -d ./test/ml-10m/0.2/ \
# -a ./test/ml-10m/ \
# -c True \
# -r ./data/movielens/ml-10m_ratings.dat \
# -i ./data/movielens/Plot.idmap \
# -m 1
# -t 0.2

# Run main process
# python ./run.py \
# -d ../data/preprocessed/movielens_10m/cf/0.2_1/ \
# -a ../data/preprocessed/movielens_10m/ \
# -o ./test/movielens_10m/result/1_100_200 \
# -e 200 \
# -p ../data/preprocessed/glove/glove.6B.200d.txt \
# -u 10 \
# -v 100 \
# -g True
strt=$1
n=$2
rating=$3
plot=$4
dest_dir=$5
for i in $( eval echo {$strt..$n} ); do
    test_part=$(awk "BEGIN {print $i/10}")
    train_part=$(awk "BEGIN {print 1-$test_part}")
    python run.py \
    -d ./$dest_dir/$(awk "BEGIN {print $train_part*100}")-$(awk "BEGIN {print ${test_part}*100}")/data/ \
    -a ./$dest_dir/$(awk "BEGIN {print $train_part*100}")-$(awk "BEGIN {print ${test_part}*100}")/data/ \
    -c True \
    -r $rating \
    -i $plot \
    -m 1 \
    -t $test_part

    python run.py \
    -d ./$dest_dir/$(awk "BEGIN {print $train_part*100}")-$(awk "BEGIN {print ${test_part}*100}")/data/ \
    -a ./$dest_dir/$(awk "BEGIN {print $train_part*100}")-$(awk "BEGIN {print ${test_part}*100}")/data/ \
    -o ./$dest_dir/$(awk "BEGIN {print $train_part*100}")-$(awk "BEGIN {print ${test_part}*100}")/result/ \
    -e 200 \
    -u 10 \
    -v 100 \
    -g True
done
