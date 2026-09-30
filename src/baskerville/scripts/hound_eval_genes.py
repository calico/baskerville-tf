#!/usr/bin/env python
# Copyright 2021 Calico LLC
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
import gc
import json
import os
import time

from intervaltree import IntervalTree
import numpy as np
import pandas as pd
import pybedtools
import pyranges as pr
from qnorm import quantile_normalize
from scipy.stats import pearsonr
from sklearn.metrics import explained_variance_score

import pygene
from baskerville import dataset
from baskerville import seqnn

"""
hound_eval_genes.py

Measure accuracy at gene-level.
"""


################################################################################
# main
################################################################################
def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model on genes.")
    parser.add_argument(
        "--head",
        dest="head_i",
        default=0,
        type=int,
        help="Parameters head to evaluate [Default: %(default)s]",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        default="evalg_out",
        help="Output directory for evaluation statistics [Default: %(default)s]",
    )
    parser.add_argument(
        "--pseudo_qtl",
        default=None,
        type=float,
        help="Quantile of coverage to add as pseudo counts to genes [Default: %(default)s]",
    )
    parser.add_argument(
        "--rc",
        default=False,
        action="store_true",
        help="Average the fwd and rc predictions [Default: %(default)s]",
    )
    parser.add_argument(
        "--save_span",
        default=False,
        action="store_true",
        help="Store predicted/measured gene span coverage profiles [Default: %(default)s]",
    )
    parser.add_argument(
        "--shifts",
        default="0",
        help="Ensemble prediction shifts [Default: %(default)s]",
    )
    parser.add_argument(
        "--span",
        default=False,
        action="store_true",
        help="Aggregate entire gene span [Default: %(default)s]",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split label for eg TFR pattern [Default: %(default)s]",
    )
    parser.add_argument(
        "-t",
        "--targets_file",
        default=None,
        help="File specifying target indexes and labels in table format",
    )
    parser.add_argument("params_file", help="JSON file with model parameters")
    parser.add_argument("model_file", help="Trained model file.")
    parser.add_argument("data_dir", help="Train/valid/test data directory")
    parser.add_argument("genes_gtf_file", help="GTF file with gene annotations")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # parse shifts to integers
    args.shifts = [int(shift) for shift in args.shifts.split(",")]

    #######################################################
    # inputs

    # read targets
    if args.targets_file is None:
        args.targets_file = f"{args.data_dir}/targets.txt"
    targets_df = pd.read_csv(args.targets_file, index_col=0, sep="\t")

    # set target groups
    if "group" not in targets_df.columns:
        targets_group = []
        for ti in range(targets_df.shape[0]):
            description = targets_df.iloc[ti].description
            tg = description.split(":")[0]
            targets_group.append(tg)
        targets_df["group"] = targets_group

    # read model parameters
    with open(args.params_file) as params_open:
        params = json.load(params_open)
    params_model = params["model"]

    # prep strand
    targets_strand_df = dataset.targets_prep_strand(targets_df)
    num_targets_strand = targets_strand_df.shape[0]

    # set strand pairs (using new indexing)
    orig_new_index = dict(zip(targets_df.index, np.arange(targets_df.shape[0])))
    targets_strand_pair = [orig_new_index[ti] for ti in targets_df.strand_pair]
    params_model["strand_pair"] = [np.array(targets_strand_pair)]

    # construct eval data
    eval_data = dataset.SeqDataset(
        args.data_dir, split_label=args.split, batch_size=1, mode="eval"
    )

    # initialize model
    seqnn_model = seqnn.SeqNN(params_model)
    seqnn_model.restore(args.model_file, args.head_i)
    seqnn_model.build_slice(targets_df.index)
    seqnn_model.build_ensemble(args.rc, args.shifts)

    #######################################################
    # sequence intervals

    # read data parameters
    with open(f"{args.data_dir}/statistics.json") as data_open:
        data_stats = json.load(data_open)
        pool_width = data_stats["pool_width"]

    # read sequence positions
    seqs_df = pd.read_csv(
        f"{args.data_dir}/sequences.bed",
        sep="\t",
        names=["Chromosome", "Start", "End", "Name"],
    )
    seqs_df = seqs_df[seqs_df.Name == args.split]
    seqs_pr = pr.PyRanges(seqs_df)

    # TEMP?
    # predict every other sorted sequence
    seqs_sort_df = seqs_df.sort_values(["Chromosome", "Start"]).copy()
    seqs_sort_df.reset_index(inplace=True, drop=True)
    sort_predict = np.zeros(seqs_df.shape[0], dtype=bool)
    sort_predict[::2] = True
    indexes_predict = seqs_sort_df.index[sort_predict]
    seqs_predict = np.zeros(len(seqs_df), dtype=bool)
    seqs_predict[indexes_predict] = True

    #######################################################
    # make gene BED

    t0 = time.time()
    print("Making gene BED...", end="")
    genes_bed_file = f"{args.out_dir}/genes.bed"
    if args.span:
        make_genes_span(genes_bed_file, args.genes_gtf_file, args.out_dir)
    else:
        make_genes_exon(genes_bed_file, args.genes_gtf_file, args.out_dir)

    genes_pr = pr.read_bed(genes_bed_file)
    print("DONE in %ds" % (time.time() - t0))

    # count gene normalization lengths
    gene_lengths = {}
    gene_strand = {}
    for line in open(genes_bed_file):
        a = line.rstrip().split("\t")
        gene_id = a[3]
        gene_seg_len = int(a[2]) - int(a[1])
        gene_lengths[gene_id] = gene_lengths.get(gene_id, 0) + gene_seg_len
        gene_strand[gene_id] = a[5]

    #######################################################
    # intersect genes w/ preds, targets

    # intersect seqs, genes
    t0 = time.time()
    print("Intersecting sequences w/ genes...", end="")
    seqs_genes_pr = seqs_pr.join(genes_pr)
    print("DONE in %ds" % (time.time() - t0), flush=True)

    # hash preds/targets by gene_id
    gene_preds_dict = {}
    gene_targets_dict = {}

    si = 0
    for x, y in eval_data.dataset:
        if seqs_predict[si]:
            yh = None
            y = y.numpy()[..., targets_df.index]

            # predict only if gene overlaps
            seq = seqs_df.iloc[si]
            cseqs_genes_df = seqs_genes_pr[seq.Chromosome].df
            if cseqs_genes_df.shape[0] == 0:
                # empty. no genes on this chromosome
                seq_genes_df = cseqs_genes_df
            else:
                seq_genes_df = cseqs_genes_df[cseqs_genes_df.Start == seq.Start]

            for _, seq_gene in seq_genes_df.iterrows():
                gene_id = seq_gene.Name_b
                gene_start = seq_gene.Start_b
                gene_end = seq_gene.End_b
                seq_start = seq_gene.Start

                # clip boundaries
                gene_seq_start = max(0, gene_start - seq_start)
                gene_seq_end = max(0, gene_end - seq_start)

                # round down and up, respectively
                bin_start = int(np.floor(gene_seq_start / pool_width))
                bin_end = int(np.ceil(gene_seq_end / pool_width))

                # predict
                if yh is None:
                    yh = seqnn_model(x)

                # slice gene region
                yhb = yh[0, bin_start:bin_end].astype("float16")
                yb = y[0, bin_start:bin_end].astype("float16")

                if len(yb) > 0:
                    gene_preds_dict.setdefault(gene_id, []).append(yhb)
                    gene_targets_dict.setdefault(gene_id, []).append(yb)

        # advance sequence table index
        si += 1
        if si % 16 == 0:
            gc.collect()

    #######################################################
    # # aggregate gene bin values into arrays
    gene_targets = []
    gene_preds = []
    gene_ids = sorted(gene_targets_dict.keys())
    gene_within = []
    gene_wvar = []
    num_genes = len(gene_ids)

    for gene_id in gene_ids:
        gene_preds_gi = np.concatenate(gene_preds_dict[gene_id], axis=0).astype(
            "float32"
        )
        gene_targets_gi = np.concatenate(gene_targets_dict[gene_id], axis=0).astype(
            "float32"
        )

        # slice strand
        if gene_strand[gene_id] == "+":
            gene_strand_mask = (targets_df.strand != "-").to_numpy()
        else:
            gene_strand_mask = (targets_df.strand != "+").to_numpy()
        gene_preds_gi = gene_preds_gi[:, gene_strand_mask]
        gene_targets_gi = gene_targets_gi[:, gene_strand_mask]

        if gene_targets_gi.shape[0] == 0:
            print(f"Empty gene: {gene_id}")
            print(gene_targets_gi.shape, gene_preds_gi.shape)

        # untransform
        gene_preds_gi = dataset.untransform_preds(gene_preds_gi, targets_strand_df)
        gene_targets_gi = dataset.untransform_preds(gene_targets_gi, targets_strand_df)

        # compute within gene correlation before dropping length axis
        gene_corr_gi = np.zeros(num_targets_strand)
        for ti in range(num_targets_strand):
            if (
                gene_preds_gi[:, ti].var() > 1e-6
                and gene_targets_gi[:, ti].var() > 1e-6
            ):
                preds_log = np.log2(gene_preds_gi[:, ti] + 1)
                targets_log = np.log2(gene_targets_gi[:, ti] + 1)
                gene_corr_gi[ti] = pearsonr(preds_log, targets_log)[0]
            else:
                gene_corr_gi[ti] = np.nan
        gene_within.append(gene_corr_gi)
        gene_wvar.append(gene_targets_gi.var(axis=0))

        # optionally store raw coverage profiles for gene span
        if args.save_span:
            hash_code = str(gene_id.split(".")[0][-1])  # last digit of gene id

            os.makedirs(f"{args.out_dir}/gene_within", exist_ok=True)
            os.makedirs(f"{args.out_dir}/gene_within/{hash_code}", exist_ok=True)
            os.makedirs(f"{args.out_dir}/gene_within/{hash_code}/preds", exist_ok=True)
            os.makedirs(
                f"{args.out_dir}/gene_within/{hash_code}/targets", exist_ok=True
            )
            np.save(
                f"{args.out_dir}/gene_within/{hash_code}/preds/{gene_id}_preds.npy",
                gene_preds_gi.astype("float16"),
            )
            np.save(
                f"{args.out_dir}/gene_within/{hash_code}/targets/{gene_id}_targets.npy",
                gene_targets_gi.astype("float16"),
            )

        # mean across nucleotides
        gene_preds_gi = gene_preds_gi.mean(axis=0) / float(pool_width)
        gene_targets_gi = gene_targets_gi.mean(axis=0) / float(pool_width)

        # scale by gene length
        gene_preds_gi *= gene_lengths[gene_id]
        gene_targets_gi *= gene_lengths[gene_id]

        gene_preds.append(gene_preds_gi)
        gene_targets.append(gene_targets_gi)

    gene_targets = np.array(gene_targets)
    gene_preds = np.array(gene_preds)
    gene_within = np.array(gene_within)
    gene_wvar = np.array(gene_wvar)

    # add pseudo coverage
    if args.pseudo_qtl is not None:
        for ti in range(num_targets_strand):
            nonzero_index = np.nonzero(gene_targets[:, ti] != 0.0)[0]

            pseudo_t = np.quantile(
                gene_targets[:, ti][nonzero_index], q=args.pseudo_qtl
            )
            pseudo_p = np.quantile(gene_preds[:, ti][nonzero_index], q=args.pseudo_qtl)

            gene_targets[:, ti] += pseudo_t
            gene_preds[:, ti] += pseudo_p

    # log2 transform
    gene_targets = np.log2(gene_targets + 1)
    gene_preds = np.log2(gene_preds + 1)

    #######################################################
    # quantile and mean normalize

    gene_targets_norm = np.zeros_like(gene_targets)
    gene_targets_norm0 = np.zeros_like(gene_targets)
    gene_preds_norm = np.zeros_like(gene_preds)
    gene_preds_norm0 = np.zeros_like(gene_preds)

    target_groups = sorted(targets_strand_df.group.unique())
    for group in target_groups:
        group_mask = targets_strand_df.group == group

        gene_targets_norm[:, group_mask] = quantile_normalize(
            gene_targets[:, group_mask], ncpus=2
        )
        gene_targets_norm0[:, group_mask] = gene_targets_norm[:, group_mask]
        gene_targets_norm0[:, group_mask] -= gene_targets_norm[:, group_mask].mean(
            axis=-1, keepdims=True
        )

        gene_preds_norm[:, group_mask] = quantile_normalize(
            gene_preds[:, group_mask], ncpus=2
        )
        gene_preds_norm0[:, group_mask] = gene_preds_norm[:, group_mask]
        gene_preds_norm0[:, group_mask] -= gene_preds_norm[:, group_mask].mean(
            axis=-1, keepdims=True
        )

    #######################################################
    # save values

    # targets
    genes_targets_df = pd.DataFrame(
        gene_targets, index=gene_ids, columns=targets_strand_df.identifier
    )
    genes_targets_df.to_csv(
        f"{args.out_dir}/gene_targets.tsv.gz",
        sep="\t",
        compression="gzip",
        float_format="%.3f",
    )
    genes_targets_norm_df = pd.DataFrame(
        gene_targets_norm, index=gene_ids, columns=targets_strand_df.identifier
    )
    genes_targets_norm_df.to_csv(
        f"{args.out_dir}/gene_targets_norm.tsv.gz",
        sep="\t",
        compression="gzip",
        float_format="%.3f",
    )

    # predictions
    genes_preds_df = pd.DataFrame(
        gene_preds, index=gene_ids, columns=targets_strand_df.identifier
    )
    genes_preds_df.to_csv(
        f"{args.out_dir}/gene_preds.tsv.gz",
        sep="\t",
        compression="gzip",
        float_format="%.3f",
    )
    genes_preds_norm_df = pd.DataFrame(
        gene_preds_norm, index=gene_ids, columns=targets_strand_df.identifier
    )
    genes_preds_norm_df.to_csv(
        f"{args.out_dir}/gene_preds_norm.tsv.gz",
        sep="\t",
        compression="gzip",
        float_format="%.3f",
    )

    # within
    genes_within_df = pd.DataFrame(
        gene_within, index=gene_ids, columns=targets_strand_df.identifier
    )
    genes_within_df.to_csv(
        f"{args.out_dir}/gene_within.tsv.gz",
        sep="\t",
        compression="gzip",
        float_format="%.3f",
    )
    genes_var_df = pd.DataFrame(
        gene_wvar, index=gene_ids, columns=targets_strand_df.identifier
    )
    genes_var_df.to_csv(
        f"{args.out_dir}/gene_var.tsv.gz",
        sep="\t",
        compression="gzip",
        float_format="%.3f",
    )

    #######################################################
    # track metrics

    wvar_t = np.percentile(gene_wvar, 80, axis=0)

    acc_pearsonr = []
    acc_r2 = []
    acc_npearsonr = []
    acc_nr2 = []
    acc_wpearsonr = []
    for ti in range(num_targets_strand):
        r_ti = pearsonr(gene_targets[:, ti], gene_preds[:, ti])[0]
        acc_pearsonr.append(r_ti)
        r2_ti = explained_variance_score(gene_targets[:, ti], gene_preds[:, ti])
        acc_r2.append(r2_ti)
        nr_ti = pearsonr(gene_targets_norm0[:, ti], gene_preds_norm0[:, ti])[0]
        acc_npearsonr.append(nr_ti)
        nr2_ti = explained_variance_score(
            gene_targets_norm0[:, ti], gene_preds_norm0[:, ti]
        )
        acc_nr2.append(nr2_ti)
        var_mask = gene_wvar[:, ti] > wvar_t[ti]
        wr_ti = gene_within[:, ti][var_mask].mean()
        acc_wpearsonr.append(wr_ti)

    task_metrics = pd.DataFrame(
        {
            "identifier": targets_strand_df.identifier,
            "group": targets_strand_df.group,
            "pearsonr": acc_pearsonr,
            "r2": acc_r2,
            "pearsonr_norm": acc_npearsonr,
            "r2_norm": acc_nr2,
            "pearsonr_gene": acc_wpearsonr,
            "description": targets_strand_df.description,
        }
    )
    task_metrics.to_csv(f"{args.out_dir}/task_metrics.tsv", sep="\t")

    print(f"{num_genes} genes")
    print("Overall PearsonR:     %.4f" % np.mean(task_metrics.pearsonr))
    print("Overall R2:           %.4f" % np.mean(task_metrics.r2))
    print("Normalized PearsonR:  %.4f" % np.mean(task_metrics.pearsonr_norm))
    print("Normalized R2:        %.4f" % np.mean(task_metrics.r2_norm))
    print("Within-gene PearsonR: %.4f" % np.mean(task_metrics.pearsonr_gene))

    #######################################################
    # gene metrics

    gene_metrics = pd.DataFrame({"gene_id": gene_ids})

    # for each group
    for group in target_groups:
        group_mask = targets_strand_df.group == group
        gene_targets_group = gene_targets_norm[:, group_mask]
        gene_preds_group = gene_preds_norm[:, group_mask]

        # compute gene metrics
        gene_group_r = np.zeros(num_genes)
        gene_group_r2 = np.zeros(num_genes)
        gene_group_sd = np.zeros(num_genes)
        for gi, gene_id in enumerate(gene_ids):
            gene_group_sd[gi] = gene_targets_group[gi].std()
            if gene_preds_group[gi].std() < 1e-6:
                gene_group_r[gi] = 0
                gene_group_r2[gi] = 0
            else:
                gene_group_r[gi] = pearsonr(
                    gene_targets_group[gi], gene_preds_group[gi]
                )[0]
                gene_group_r2[gi] = explained_variance_score(
                    gene_targets_group[gi], gene_preds_group[gi]
                )

        # save
        gene_metrics[f"{group}_r"] = gene_group_r
        gene_metrics[f"{group}_r2"] = gene_group_r2
        gene_metrics[f"{group}_sd"] = gene_group_sd

        # summarize
        print(f"{group} R:  {np.mean(gene_group_r):.4f}")
        print(f"{group} R2: {np.mean(gene_group_r2):.4f}")

        # summarize highly variable
        sd50_mask = gene_group_sd > np.median(gene_group_sd)
        group_r_sd50 = gene_group_r[sd50_mask].mean()
        group_r2_sd50 = gene_group_r2[sd50_mask].mean()
        print(f"{group} R (sd50):  {group_r_sd50:.4f}")
        print(f"{group} R2 (sd50): {group_r2_sd50:.4f}")
        print("")

    # write
    gene_metrics.to_csv(
        f"{args.out_dir}/gene_metrics.tsv", sep="\t", float_format="%.4f"
    )


def genes_aggregate(genes_bed_file, values_bedgraph):
    """Aggregate values across genes.

    Args:
      genes_bed_file (str): BED file of genes.
      values_bedgraph (str): BedGraph file of values.

    Returns:
      gene_values (dict): Dictionary of gene values.
    """
    values_bt = pybedtools.BedTool(values_bedgraph)
    genes_bt = pybedtools.BedTool(genes_bed_file)

    gene_values = {}

    for overlap in genes_bt.intersect(values_bt, wo=True):
        gene_id = overlap[3]
        value = overlap[7]
        gene_values[gene_id] = gene_values.get(gene_id, 0) + value

    return gene_values


def make_genes_exon(genes_bed_file: str, genes_gtf_file: str, out_dir: str):
    """Make a BED file with each genes' exons, excluding exons overlapping
      across genes.

    Args:
      genes_bed_file (str): Output BED file of genes.
      genes_gtf_file (str): Input GTF file of genes.
      out_dir (str): Output directory for temporary files.
    """
    # read genes
    genes_gtf = pygene.GTF(genes_gtf_file)

    # write gene exons
    agenes_bed_file = f"{out_dir}/genes_all.bed"
    agenes_bed_out = open(agenes_bed_file, "w")
    for gene_id, gene in genes_gtf.genes.items():
        # collect exons
        gene_intervals = IntervalTree()
        for tx_id, tx in gene.transcripts.items():
            for exon in tx.exons:
                gene_intervals[exon.start - 1 : exon.end] = True

        # union
        gene_intervals.merge_overlaps()

        # write
        for interval in sorted(gene_intervals):
            cols = [
                gene.chrom,
                str(interval.begin),
                str(interval.end),
                gene_id,
                ".",
                gene.strand,
            ]
            print("\t".join(cols), file=agenes_bed_out)
    agenes_bed_out.close()

    # find overlapping exons
    genes1_bt = pybedtools.BedTool(agenes_bed_file)
    genes2_bt = pybedtools.BedTool(agenes_bed_file)
    overlapping_exons = set()
    for overlap in genes1_bt.intersect(genes2_bt, s=True, wo=True):
        gene1_id = overlap[3]
        gene1_start = int(overlap[1])
        gene1_end = int(overlap[2])
        overlapping_exons.add((gene1_id, gene1_start, gene1_end))

        gene2_id = overlap[9]
        gene2_start = int(overlap[7])
        gene2_end = int(overlap[8])
        overlapping_exons.add((gene2_id, gene2_start, gene2_end))

    # filter for nonoverlapping exons
    genes_bed_out = open(genes_bed_file, "w")
    for line in open(agenes_bed_file):
        a = line.split()
        start = int(a[1])
        end = int(a[2])
        gene_id = a[-1]
        if (gene_id, start, end) not in overlapping_exons:
            print(line, end="", file=genes_bed_out)
    genes_bed_out.close()


def make_genes_span(
    genes_bed_file: str, genes_gtf_file: str, out_dir: str, stranded: bool = True
):
    """Make a BED file with the span of each gene.

    Args:
      genes_bed_file (str): Output BED file of genes.
      genes_gtf_file (str): Input GTF file of genes.
      out_dir (str): Output directory for temporary files.
      stranded (bool): Perform stranded intersection.
    """
    # read genes
    genes_gtf = pygene.GTF(genes_gtf_file)

    # write all gene spans
    agenes_bed_file = f"{out_dir}/genes_all.bed"
    agenes_bed_out = open(agenes_bed_file, "w")
    for gene_id, gene in genes_gtf.genes.items():
        start, end = gene.span()
        cols = [gene.chrom, str(start - 1), str(end), gene_id, ".", gene.strand]
        print("\t".join(cols), file=agenes_bed_out)
    agenes_bed_out.close()

    # find overlapping genes
    genes1_bt = pybedtools.BedTool(agenes_bed_file)
    genes2_bt = pybedtools.BedTool(agenes_bed_file)
    overlapping_genes = set()
    for overlap in genes1_bt.intersect(genes2_bt, s=stranded, wo=True):
        gene1_id = overlap[3]
        gene2_id = overlap[7]
        if gene1_id != gene2_id:
            overlapping_genes.add(gene1_id)
            overlapping_genes.add(gene2_id)

    # filter for nonoverlapping genes
    genes_bed_out = open(genes_bed_file, "w")
    for line in open(agenes_bed_file):
        gene_id = line.split()[-1]
        if gene_id not in overlapping_genes:
            print(line, end="", file=genes_bed_out)
    genes_bed_out.close()


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
