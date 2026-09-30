#! /bin/bash

conda activate baskerville

# files
data_path='/home/yuanh/analysis/Borzoi_transfer/tutorial/mouse/data'
MM10=${data_path}/mm10
FASTA_MOUSE=$MM10/mm10.ml.fa
UMAP_MOUSE=$MM10/umap_k36_t10_l32.bed
BLACK_MOUSE=$MM10/blacklist_mm10_all.bed
OUT=${data_path}/tfr
CONTIGDATA=${data_path}/trainsplit

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
hound_data.py --restart $DOPTS --local -b $BLACK_MOUSE -o $OUT $FASTA_MOUSE -u $OUT/umap_mouse.bed targets_tutorial.txt
#hound_data.py --restart $DOPTS -b $BLACK_MOUSE -o $OUT $FASTA_MOUSE -u $OUT/umap_mouse.bed targets_tutorial.txt
