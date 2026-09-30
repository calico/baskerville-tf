#!/usr/bin/env python
# Copyright 2023 Calico LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================
import argparse
import json
import os

import h5py
import numpy as np
import pandas as pd
import pyBigWig
import tensorflow as tf
from tqdm import tqdm

from baskerville import bed
from baskerville import dataset
from baskerville import dna
from baskerville import seqnn


"""
hound_predbed.py

Predict sequences from a BED file.
"""


def main():
    parser = argparse.ArgumentParser(description="Predict sequences from a BED file.")
    parser.add_argument(
        "-b",
        "--bigwig_indexes",
        default=None,
        help="Comma-separated list of target indexes to write BigWigs",
    )
    parser.add_argument(
        "-e",
        "--embed_layer",
        default=None,
        type=int,
        help="Embed sequences using the specified layer index.",
    )
    parser.add_argument(
        "-f",
        "--genome_fasta",
        default=None,
        help="Genome FASTA for sequences [Default: %(default)s]",
    )
    parser.add_argument(
        "-g",
        "--genome_file",
        default=None,
        help="Chromosome length information [Default: %(default)s]",
    )
    parser.add_argument(
        "--head",
        default=0,
        type=int,
        help="Model head to evaluate [Default: %(default)s]",
    )
    parser.add_argument(
        "-l",
        "--site_length",
        default=None,
        type=int,
        help="Prediction site length. [Default: params.seq_length]",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        default="pred_out",
        help="Output directory [Default: %(default)s]",
    )
    parser.add_argument(
        "-p",
        "--processes",
        default=None,
        type=int,
        help="Number of processes, passed by multi script",
    )
    parser.add_argument(
        "--rc",
        default=False,
        action="store_true",
        help="Average the fwd and rc predictions [Default: %(default)s]",
    )
    parser.add_argument(
        "-s",
        "--sum",
        default=False,
        action="store_true",
        help="Sum site predictions [Default: %(default)s]",
    )
    parser.add_argument(
        "--shifts",
        default="0",
        help="Ensemble prediction shifts [Default: %(default)s]",
    )
    parser.add_argument(
        "-t",
        "--targets_file",
        default=None,
        help="File specifying target indexes and labels in table format",
    )
    parser.add_argument(
        "-u",
        "--untransform",
        default=False,
        action="store_true",
        help="Untransform predictions [Default: %default]",
    )
    parser.add_argument(
        "--untransform_old",
        default=False,
        action="store_true",
        help="Untransform predictions, using the original recipe [Default: %default]",
    )
    parser.add_argument("params_file", help="Parameters file")
    parser.add_argument("model_file", help="Model file")
    parser.add_argument("bed_file", help="BED file")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    args.shifts = [int(shift) for shift in args.shifts.split(",")]

    if args.bigwig_indexes is not None:
        args.bigwig_indexes = [int(bi) for bi in args.bigwig_indexes.split(",")]
    else:
        args.bigwig_indexes = []

    if len(args.bigwig_indexes) > 0:
        bigwig_dir = "%s/bigwig" % args.out_dir
        if not os.path.isdir(bigwig_dir):
            os.mkdir(bigwig_dir)

    #################################################################
    # read parameters and collet target information

    with open(args.params_file) as params_open:
        params = json.load(params_open)
    params_model = params["model"]

    if args.targets_file is None:
        target_slice = None
    else:
        targets_df = pd.read_table(args.targets_file, index_col=0)
        target_slice = targets_df.index

        # update strand pairs for new indexing
        orig_new_index = dict(zip(targets_df.index, np.arange(targets_df.shape[0])))
        targets_strand_pair = [orig_new_index[ti] for ti in targets_df.strand_pair]
        params_model["strand_pair"] = [np.array(targets_strand_pair)]

    #################################################################
    # setup model

    # initialize model
    seqnn_model = seqnn.SeqNN(params_model)
    seqnn_model.restore(args.model_file, args.head)
    seqnn_model.build_slice(target_slice)
    seqnn_model.build_ensemble(args.rc, args.shifts)

    if args.embed_layer is not None:
        seqnn_model.build_embed(args.embed_layer)
    _, preds_length, preds_depth = seqnn_model.model.output.shape

    if isinstance(preds_length, tf.compat.v1.Dimension):
        preds_length = preds_length.value
        preds_depth = preds_depth.value

    preds_window = seqnn_model.model_strides[0]
    seq_crop = seqnn_model.target_crops[0] * preds_window

    #################################################################
    # sequence dataset

    if args.site_length is None:
        args.site_length = preds_window * preds_length
        print("site_length: %d" % args.site_length)

    # construct model sequences
    model_seqs_dna, model_seqs_coords = bed.make_bed_seqs(
        args.bed_file, args.genome_fasta, params_model["seq_length"], stranded=False
    )
    num_seqs = len(model_seqs_dna)

    #################################################################
    # setup output

    if preds_length % 2 != 0:
        print("WARNING: preds_lengh is odd and therefore asymmetric.")
    preds_mid = preds_length // 2

    assert args.site_length % preds_window == 0
    site_preds_length = args.site_length // preds_window

    if site_preds_length % 2 != 0:
        print("WARNING: site_preds_length is odd and therefore asymmetric.")
    site_preds_start = preds_mid - site_preds_length // 2
    site_preds_end = site_preds_start + site_preds_length

    # initialize HDF5
    out_h5_file = "%s/predict.h5" % args.out_dir
    if os.path.isfile(out_h5_file):
        os.remove(out_h5_file)
    out_h5 = h5py.File(out_h5_file, "w")

    # create predictions
    if args.sum:
        out_h5.create_dataset("preds", dtype="float16", shape=(num_seqs, preds_depth))
    else:
        out_h5.create_dataset(
            "preds", dtype="float16", shape=(num_seqs, site_preds_length, preds_depth)
        )

    # store site coordinates
    site_seqs_coords = bed.read_bed_coords(args.bed_file, args.site_length)
    site_seqs_chr, site_seqs_start, site_seqs_end = zip(*site_seqs_coords)
    site_seqs_chr = np.array(site_seqs_chr, dtype="S")
    site_seqs_start = np.array(site_seqs_start)
    site_seqs_end = np.array(site_seqs_end)
    out_h5.create_dataset("chrom", data=site_seqs_chr)
    out_h5.create_dataset("start", data=site_seqs_start)
    out_h5.create_dataset("end", data=site_seqs_end)

    #################################################################
    # predict scores, write output

    for si, seq_dna in tqdm(enumerate(model_seqs_dna)):
        seq_1hot = np.expand_dims(dna.dna_1hot(seq_dna), axis=0)
        preds_seq = seqnn_model.predict(seq_1hot)[0]

        if args.untransform_old:
            preds_seq = dataset.untransform_preds1(preds_seq, targets_df)
        elif args.untransform:
            preds_seq = dataset.untransform_preds(preds_seq, targets_df)

        # slice site
        preds_site = preds_seq[site_preds_start:site_preds_end, :]

        # optionally, sum
        if args.sum:
            preds_site = np.sum(preds_site, axis=0)

        # clip to float16 max
        preds_write = np.clip(preds_site, 0, np.finfo(np.float16).max)

        # write
        out_h5["preds"][si] = preds_write

        # write bigwig
        for ti in args.bigwig_indexes:
            bw_file = "%s/s%d_t%d.bw" % (bigwig_dir, si, ti)
            bigwig_write(
                preds_seq[:, ti],
                model_seqs_coords[si],
                bw_file,
                args.genome_file,
                seq_crop,
            )

    # close output HDF5
    out_h5.close()


def bigwig_open(bw_file, genome_file):
    """Open the bigwig file for writing and write the header."""

    bw_out = pyBigWig.open(bw_file, "w")

    chrom_sizes = []
    for line in open(genome_file):
        a = line.split()
        chrom_sizes.append((a[0], int(a[1])))

    bw_out.addHeader(chrom_sizes)

    return bw_out


def bigwig_write(signal, seq_coords, bw_file, genome_file, seq_crop=0):
    """Write a signal track to a BigWig file over the region
         specified by seqs_coords.

    Args
     signal:      Sequences x Length signal array
     seq_coords:  (chr,start,end)
     bw_file:     BigWig filename
     genome_file: Chromosome lengths file
     seq_crop:    Sequence length cropped from each side of the sequence.
    """
    target_length = len(signal)

    # open bigwig
    bw_out = bigwig_open(bw_file, genome_file)

    # initialize entry arrays
    entry_starts = []
    entry_ends = []

    # set entries
    chrm, start, end = seq_coords
    preds_pool = (end - start - 2 * seq_crop) // target_length

    bw_start = start + seq_crop
    for li in range(target_length):
        bw_end = bw_start + preds_pool
        entry_starts.append(bw_start)
        entry_ends.append(bw_end)
        bw_start = bw_end

    # add
    bw_out.addEntries(
        [chrm] * target_length,
        entry_starts,
        ends=entry_ends,
        values=[float(s) for s in signal],
    )

    bw_out.close()


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
