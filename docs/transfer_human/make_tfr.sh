#! /bin/bash

conda activate baskerville

# files
data_path='/home/yuanh/analysis/Borzoi_transfer/tutorial/data'
OUT=${data_path}/tfr
HG38=${data_path}/hg38
CONTIGDATA=${data_path}/trainsplit
FASTA_HUMAN=$HG38/hg38.ml.fa
UMAP_HUMAN=$HG38/umap_k36_t10_l32.bed
BLACK_HUMAN=$HG38/blacklist_hg38_all.bed

# params
LENGTH=524288
CROP=163840
WIDTH=32
FOLDS=8
DOPTS="-c $CROP -d 2 -f $FOLDS -l $LENGTH -p 32 -r 256 --umap_clip 0.5 -w $WIDTH"

# copy sequence contigs, mappability and train/val/test split.
mkdir $OUT
cp ${CONTIGDATA}/* $OUT

# by default, hound_data is run on slurm system, specify --local to run on local machine.
hound_data.py --restart $DOPTS --local -b $BLACK_HUMAN -o $OUT $FASTA_HUMAN -u $OUT/umap_human.bed targets.txt
