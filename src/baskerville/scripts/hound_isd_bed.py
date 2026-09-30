#!/usr/bin/env python
# Copyright 2024 Calico LLC
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
from optparse import OptionParser

import json
import os

import h5py
import numpy as np
import pandas as pd
import pybedtools

from baskerville import bed
from baskerville import dataset
from baskerville import dna
from baskerville import seqnn
from baskerville import snps
from baskerville.gene import Transcriptome


from collections import OrderedDict


def main():
    """
    hound_isd_sed.py

    Perform an in silico deletion mutagenesis of sequences in a BED file,
    where predictions are centered on the variant and SED/logSED scores can be calculated.
    Outputs a separate .h5 file for each .bed entry.

    Usage:
        hound_isd_sed.py [options] <params_file> <model_file> <bed_file>
    Options:
        -f <genome_fasta>    Genome FASTA for sequences
        -g <genes_gtf>       GTF for gene definition
        -o <out_dir>        Output directory
        -p <processes>      Number of processes, passed by multi script
        --rc                Ensemble forward and reverse complement predictions
        --shifts            Ensemble prediction shifts
        --span              Aggregate entire gene span
        --stats             Comma-separated list of stats to save.
        -s <del_size>       Deletion size for ISD
        --target_genes      List of target genes in .tsv format, length must match input bed entries
        -t <targets_file>   File specifying target indexes and labels in table format
        --untransform_old   Untransform old models
    """

    usage = "usage: %prog [options] <params_file> <model_file> <bed_file>"
    parser = OptionParser(usage)
    parser.add_option(
        "-f",
        dest="genome_fasta",
        default=None,
        help="Genome FASTA for sequences [Default: %default]",
    )
    parser.add_option(
        "-g",
        dest="genes_gtf",
        default="/group/fdna/public/genomes/hg38/genes/gencode41/gencode41_basic_nort.gtf",
        help="GTF for gene definition [Default %default]",
    )
    parser.add_option(
        "-o",
        dest="out_dir",
        default="sat_del",
        help="Output directory [Default: %default]",
    )
    parser.add_option(
        "-p",
        dest="processes",
        default=None,
        type="int",
        help="Number of processes, passed by multi script",
    )
    parser.add_option(
        "--rc",
        dest="rc",
        default=False,
        action="store_true",
        help="Ensemble forward and reverse complement predictions [Default: %default]",
    )
    parser.add_option(
        "--shifts",
        dest="shifts",
        default="0",
        help="Ensemble prediction shifts [Default: %default]",
    )
    parser.add_option(
        "--span",
        dest="span",
        default=False,
        action="store_true",
        help="Aggregate entire gene span [Default: %default]",
    )
    parser.add_option(
        "--stats",
        dest="snp_stats",
        default="logSED",
        help="Comma-separated list of stats to save. [Default: %default]",
    )
    parser.add_option(
        "-s",
        dest="del_size",
        default=1,
        type="int",
        help="Deletion size for ISD [Default: %default]",
    )
    parser.add_option(
        "--target_genes",
        dest="target_genes",
        default=None,
        help="List of target genes in .tsv format, length must match input bed entries [Default %default]",
    )
    parser.add_option(
        "-t",
        dest="targets_file",
        default=None,
        type="str",
        help="File specifying target indexes and labels in table format",
    )
    parser.add_option(
        "--untransform_old",
        dest="untransform_old",
        default=False,
        action="store_true",
        help="Untransform old models [Default: %default]",
    )
    (options, args) = parser.parse_args()

    if len(args) == 3:
        # single worker
        params_file = args[0]
        model_file = args[1]
        bed_file = args[2]
    else:
        parser.error("Must provide parameter and model files and BED file")

    options.shifts = [int(shift) for shift in options.shifts.split(",")]
    options.snp_stats = [snp_stat for snp_stat in options.snp_stats.split(",")]

    #################################################################
    # read parameters and targets

    # read model parameters
    with open(params_file) as params_open:
        params = json.load(params_open)
    params_model = params["model"]

    # create output directory if it does not exist
    if not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)

    # read targets
    if options.targets_file is None:
        parser.error("Must provide targets file to clarify stranded datasets")
    targets_df = pd.read_csv(options.targets_file, sep="\t", index_col=0)

    if options.target_genes is not None:
        target_genes = pd.read_csv(
            options.target_genes, sep="\t", index_col=None, header=None
        )
        target_genes.columns = ["gene_id"]

    # handle strand pairs
    if "strand_pair" in targets_df.columns:
        # prep strand
        targets_strand_df = dataset.targets_prep_strand(targets_df)

        # set strand pairs (using new indexing)
        orig_new_index = dict(zip(targets_df.index, np.arange(targets_df.shape[0])))
        targets_strand_pair = np.array(
            [orig_new_index[ti] for ti in targets_df.strand_pair]
        )
        params_model["strand_pair"] = [targets_strand_pair]

        # construct strand sum transform
        strand_transform = dataset.make_strand_transform(targets_df, targets_strand_df)
    else:
        targets_strand_df = targets_df
        strand_transform = None
    num_targets = targets_strand_df.shape[0]

    #################################################################
    # setup model

    seqnn_model = seqnn.SeqNN(params_model)
    seqnn_model.restore(model_file)
    seqnn_model.build_slice(targets_df.index)
    seqnn_model.build_ensemble(options.rc)

    #################################################################
    # load transcriptome

    transcriptome = Transcriptome(options.genes_gtf)
    gene_strand = {}
    for gene_id, gene in transcriptome.genes.items():
        gene_strand[gene_id] = gene.strand

    #################################################################
    # sequence dataset

    # read sequences from BED
    seqs_dna, seqs_coords, ism_lengths = bed.make_ntwise_bed_seqs(
        bed_file, options.genome_fasta, params_model["seq_length"], stranded=True
    )

    # helper variables
    seq_mid = params_model["seq_length"] // 2
    model_stride = seqnn_model.model_strides[0]

    #################################################################
    # setup output: each seq is a separate file because of different lengths
    # in each file, num_seqs = number of nt-wise deletions in bed range

    for isq in seqs_dna.keys():
        scores_h5_file = f"{options.out_dir}/scores_{isq}.h5"
        if os.path.isfile(scores_h5_file):
            os.remove(scores_h5_file)
        scores_h5 = h5py.File(scores_h5_file, "w")
        scores_h5.create_dataset(
            "seqs", dtype="bool", shape=(len(seqs_dna[isq]), ism_lengths[isq], 4)
        )
        print("Seqs shape:", (len(seqs_dna[isq]), ism_lengths[isq], 4))
        for snp_stat in options.snp_stats:
            scores_h5.create_dataset(
                snp_stat, dtype="float16", shape=(len(seqs_dna[isq]), num_targets)
            )

        # centered on deletion but mutation range length is variable
        mut_start = seq_mid - ism_lengths[isq] // 2
        mut_end = seq_mid + ism_lengths[isq] // 2

        # store mutagenesis sequence coordinates
        scores_chr = []
        scores_start = []
        scores_end = []
        scores_strand = []
        del_loci = []
        for seq_chr, seq_start, seq_end, seq_strand in seqs_coords[isq]:
            scores_chr.append(seq_chr)
            scores_strand.append(seq_strand)
            score_start = seq_start
            score_end = seq_end
            del_loci.append(seq_start + seq_mid)
            scores_start.append(score_start)
            scores_end.append(score_end)

        scores_h5.create_dataset("chr", data=np.array(scores_chr, dtype="S"))
        scores_h5.create_dataset("start", data=np.array(scores_start))
        scores_h5.create_dataset("end", data=np.array(scores_end))
        scores_h5.create_dataset("del_loci", data=np.array(del_loci))
        scores_h5.create_dataset("strand", data=np.array(scores_strand, dtype="S"))

        #################################################################
        # predict scores, write output

        for si, seq_dna in enumerate(seqs_dna[isq]):
            print(f"Predicting {si}", flush=True)

            # make list of shifts for reference stitching
            ref_shifts = []
            for shift in options.shifts:
                ref_shifts.append(shift)
                ref_shifts.append(shift - options.del_size)

            # 1 hot code DNA
            ref_1hot = dna.dna_1hot(seq_dna)
            ref_1hot = np.expand_dims(ref_1hot, axis=0)

            # save sequence: always centered on the current snp
            print("current seqs shape:", np.shape(ref_1hot[0, mut_start : mut_end + 1]))
            scores_h5["seqs"][si] = ref_1hot[0, mut_start : mut_end + 1].astype("bool")

            # predict reference
            ref_preds = []
            for shift in ref_shifts:
                # shift sequence and predict
                ref_1hot_shift = dna.hot1_augment(ref_1hot, shift=shift)
                ref_preds_shift = seqnn_model.predict_transform(
                    ref_1hot_shift,
                    targets_df,
                    strand_transform,
                    options.untransform_old,
                )
                ref_preds.append(ref_preds_shift)

            # copy and modify
            alt_1hot = np.copy(ref_1hot)

            # left-matched shift: delete 1 nucleotide at position mi
            dna.hot1_delete(alt_1hot[0], seq_mid, options.del_size)

            # predict alternate
            alt_preds = []
            for shift in options.shifts:
                # shift sequence and predict
                alt_1hot_shift = dna.hot1_augment(alt_1hot, shift=shift)
                alt_preds_shift = seqnn_model.predict_transform(
                    alt_1hot_shift,
                    targets_df,
                    strand_transform,
                    options.untransform_old,
                )
                alt_preds.append(alt_preds_shift)
            alt_preds = np.array(alt_preds)

            out_seq_len = alt_preds.shape[1]
            out_seq_center = out_seq_len // 2
            out_seq_crop_len = out_seq_len * model_stride

            # stitch reference predictions at the mutated nucleotide
            snp_seq_bin = out_seq_center

            ref_preds_stitch = snps.stitch_preds(ref_preds, options.shifts, snp_seq_bin)
            ref_preds_stitch = np.array(ref_preds_stitch)

            #################################################################
            # map SNP sequences to gene positions
            delseq_gene_slice = map_delseq_genes(
                seqs_coords[isq][si],
                out_seq_crop_len,
                options.del_size,
                transcriptome,
                model_stride,
                options.span,
            )

            # slicing all genes in the window
            gene_slice_all = []
            # for each overlapping gene
            if options.target_genes is not None:
                for gene_id, gene_slice in delseq_gene_slice.items():
                    if str(target_genes.iloc[isq]["gene_id"]) in gene_id:
                        print(
                            f"Found gene {str(target_genes.iloc[isq]['gene_id'])} in {gene_id}!"
                        )
                        gene_slice_all.extend(gene_slice)
            for gene_id, gene_slice in delseq_gene_slice.items():
                gene_slice_all.extend(gene_slice)

            gene_slice_all = list(set(gene_slice_all))

            # slice gene positions
            ref_preds_gene = np.array(ref_preds_stitch)[:, gene_slice_all, :]
            alt_preds_gene = alt_preds[:, gene_slice_all, :]

            # ref/alt_preds is B x L x T
            num_shifts, _, _ = ref_preds_stitch.shape

            # sum across length
            ref_preds_gene_sum = ref_preds_gene.sum(axis=(0, 1)) / num_shifts
            alt_preds_gene_sum = alt_preds_gene.sum(axis=(0, 1)) / num_shifts

            ref_preds_sum = ref_preds_stitch.sum(axis=(0, 1)) / num_shifts
            alt_preds_sum = alt_preds.sum(axis=(0, 1)) / num_shifts

            # SED/logSED are handled outside of snps.compute_scores
            if "SED" in options.snp_stats:
                sed = alt_preds_gene_sum - ref_preds_gene_sum
                scores_h5["SED"][si, :] = clip_float(sed).astype("float16")
            if "logSED" in options.snp_stats:
                log_sed = np.log2(alt_preds_gene_sum + 1) - np.log2(
                    ref_preds_gene_sum + 1
                )
                print("logSED shape:", log_sed.shape)
                scores_h5["logSED"][si, :] = log_sed.astype("float16")
            if "SUMlog" in options.snp_stats:
                sum_log = np.log2(alt_preds_sum + 1) - np.log2(ref_preds_sum + 1)
                scores_h5["SUMlog"][si, :] = sum_log.astype("float16")

            # compute sed here
            ism_scores = snps.compute_scores(
                ref_preds_stitch, alt_preds, options.snp_stats, None
            )
            for snp_stat in options.snp_stats:
                if snp_stat == "SED" or snp_stat == "logSED" or snp_stat == "SUMlog":
                    continue
                scores_h5[snp_stat][si, :] = ism_scores[snp_stat]

        # close output HDF5
        scores_h5.close()


def clip_float(x, dtype=np.float16):
    return np.clip(x, np.finfo(dtype).min, np.finfo(dtype).max)


def make_del_bedt(coords, seq_len: int, del_size: int):
    """Make a BedTool object for all SNP sequences, where seq_len considers cropping."""
    left_len = seq_len // 2
    right_len = seq_len // 2

    snpseq_bed_lines = []
    seq_mid = (coords[1] + coords[2]) // 2

    # bound sequence start at 0 (true sequence will be N padded)
    snpseq_start = max(0, seq_mid - left_len)
    snpseq_end = seq_mid + right_len

    # correct end for alternative indels
    snpseq_end += del_size
    snpseq_bed_lines.append("%s %d %d %d" % (coords[0], snpseq_start, snpseq_end, 0))

    snpseq_bedt = pybedtools.BedTool("\n".join(snpseq_bed_lines), from_string=True)
    return snpseq_bedt


def map_delseq_genes(
    coords,
    seq_len: int,
    del_size: int,
    transcriptome,
    model_stride: int,
    span: bool,
    majority_overlap: bool = True,
    intron1: bool = False,
):
    """Intersect SNP sequences with gene exons, constructing a list
    mapping sequence indexes to dictionaries of gene_ids to their
    exon-overlapping positions in the sequence.

    Args:
       snps ([bvcf.SNP]): SNP list.
       seq_len (int): Sequence length, after model cropping.
       transcriptome (Transcriptome): Transcriptome.
       model_stride (int): Model stride.
       span (bool): If True, use gene span instead of exons.
       majority_overlap (bool): If True, only consider bins for which
         the majority of the space overlaps an exon.
       intron1 (bool): If True, include intron bins adjacent to junctions.
    """

    # make gene BEDtool
    if span:
        genes_bedt = transcriptome.bedtool_span()
    else:
        genes_bedt = transcriptome.bedtool_exon()

    # make SNP sequence BEDtool
    snpseq_bedt = make_del_bedt(coords, seq_len, del_size)

    # map SNPs to genes
    snpseq_gene_slice = OrderedDict()

    for overlap in genes_bedt.intersect(snpseq_bedt, wo=True):
        # print("Overlap:", overlap)
        gene_id = overlap[3]
        gene_start = int(overlap[1])
        gene_end = int(overlap[2])
        seq_start = int(overlap[7])
        seq_end = int(overlap[8])

        # adjust for left overhang padded
        seq_len_chop = seq_end - seq_start
        seq_start -= seq_len - seq_len_chop

        # clip left boundaries
        gene_seq_start = max(0, gene_start - seq_start)
        gene_seq_end = max(0, gene_end - seq_start)

        if majority_overlap:
            # requires >50% overlap
            bin_start = int(np.round(gene_seq_start / model_stride))
            bin_end = int(np.round(gene_seq_end / model_stride))
        else:
            # any overlap
            bin_start = int(np.floor(gene_seq_start / model_stride))
            bin_end = int(np.ceil(gene_seq_end / model_stride))

        if intron1:
            bin_start -= 1
            bin_end += 1

        # clip boundaries
        bin_max = int(seq_len / model_stride)
        bin_start = min(bin_start, bin_max)
        bin_end = min(bin_end, bin_max)
        bin_start = max(0, bin_start)
        bin_end = max(0, bin_end)

        if bin_end - bin_start > 0:
            # save gene bin positions
            snpseq_gene_slice.setdefault(gene_id, []).extend(range(bin_start, bin_end))

    # handle possible overlaps
    for gene_id, gene_slice in snpseq_gene_slice.items():
        snpseq_gene_slice[gene_id] = np.unique(gene_slice)

    return snpseq_gene_slice


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
