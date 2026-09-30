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
import tensorflow as tf
from tqdm import tqdm

from baskerville import dataset
from baskerville import seqnn
from baskerville import vcf


"""
hound_predvcf.py

Predict full genomic windows for variants from a VCF file, 
generating predictions at 32bp-bin for both reference and alternate alleles.

Note: Currently supports SNPs only. INDELs will use reference sequence for alternate allele.
"""


def main():
    parser = argparse.ArgumentParser(description="Predict full genomic windows for VCF variants.")

    parser.add_argument(
        "-f",
        "--genome_fasta",
        default=None,
        help="Genome FASTA for sequences [Default: %(default)s]",
    )
    parser.add_argument(
        "--head",
        default=0,
        type=int,
        help="Model head to evaluate [Default: %(default)s]",
    )

    parser.add_argument(
        "-o",
        "--out_dir",
        default="predvcf_out",
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
        "--untransform_old",
        default=False,
        action="store_true",
        help="Untransform predictions, using the original recipe [Default: %default]",
    )
    parser.add_argument("params_file", help="Parameters file")
    parser.add_argument("model_file", help="Model file")
    parser.add_argument("vcf_file", help="VCF file")
    args = parser.parse_args()

    # validate inputs
    if not os.path.exists(args.params_file):
        raise FileNotFoundError(f"Parameters file not found: {args.params_file}")
    if not os.path.exists(args.model_file):
        raise FileNotFoundError(f"Model file not found: {args.model_file}")
    if not os.path.exists(args.vcf_file):
        raise FileNotFoundError(f"VCF file not found: {args.vcf_file}")
    if args.genome_fasta and not os.path.exists(args.genome_fasta):
        raise FileNotFoundError(f"Genome FASTA file not found: {args.genome_fasta}")
    if args.targets_file and not os.path.exists(args.targets_file):
        raise FileNotFoundError(f"Targets file not found: {args.targets_file}")

    os.makedirs(args.out_dir, exist_ok=True)

    args.shifts = [int(shift) for shift in args.shifts.split(",")]

    #################################################################
    # read parameters and collect target information

    with open(args.params_file) as params_open:
        params = json.load(params_open)
    params_model = params["model"]

    # read targets
    if args.targets_file is None:
        raise ValueError("Must provide targets file to clarify stranded datasets")
    targets_df = pd.read_table(args.targets_file, index_col=0)
    target_slice = targets_df.index

    # handle strand pairs (following hound_predbed.py pattern)
    if "strand_pair" in targets_df.columns:
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

    _, preds_length, num_targets = seqnn_model.model.output.shape

    if isinstance(preds_length, tf.compat.v1.Dimension):
        preds_length = preds_length.value
        num_targets = num_targets.value

    preds_window = seqnn_model.model_strides[0]
    seq_crop = seqnn_model.target_crops[0] * preds_window

    #################################################################
    # load VCF variants

    # load variants from VCF
    print(f"Loading variants from {args.vcf_file}")
    variants = vcf.vcf_snps(args.vcf_file)
    
    if len(variants) == 0:
        raise ValueError("No variants found in VCF file")
    
    print(f"Loaded {len(variants)} variants from VCF")

    # get one hot coded input sequences for both reference and alternate alleles
    print("Extracting sequences and one-hot encoding...")
    seqs_1hot, seq_headers, variants_processed, seqs_dna = vcf.snps_seq1(
        variants, params_model["seq_length"], args.genome_fasta, return_seqs=True
    )
    num_seqs = seqs_1hot.shape[0]
    num_variants = len(variants_processed)
    
    if num_seqs == 0:
        raise ValueError("No valid sequences could be extracted from variants")
    
    # vcf.snps_seq1 returns sequences for both ref and alt alleles
    # So we expect 2 sequences per variant (ref + alt)
    if num_seqs != 2 * num_variants:
        raise ValueError(f"Expected {2 * num_variants} sequences (2 per variant), got {num_seqs}")
    
    # use the processed variants (some may have been filtered out)
    variants = variants_processed
    
    print(f"Successfully processed {num_variants} variants ({num_seqs} sequences)")

    #################################################################
    # setup output

    # always use full prediction length
    site_preds_length = preds_length
    site_length = preds_window * preds_length
    
    print(f"Model input sequence length: {params_model['seq_length']} bp")
    print(f"Model prediction length: {site_length} bp ({site_preds_length} bins × {preds_window} bp)")
    print(f"Model cropping: {seq_crop} bp from each side")

    # initialize HDF5
    out_h5_file = "%s/predict.h5" % args.out_dir
    if os.path.isfile(out_h5_file):
        os.remove(out_h5_file)
    out_h5 = h5py.File(out_h5_file, "w")

    # create predictions datasets for both ref and alt (per-position)
    out_h5.create_dataset(
        "preds_ref", dtype="float16", shape=(num_variants, site_preds_length, num_targets)
    )
    out_h5.create_dataset(
        "preds_alt", dtype="float16", shape=(num_variants, site_preds_length, num_targets)
    )

    # store variant information
    var_ids = np.array([str(v) for v in variants], dtype="S")
    var_chroms = np.array([v.chr for v in variants], dtype="S")
    var_positions = np.array([v.pos for v in variants])
    var_refs = np.array([v.ref_allele for v in variants], dtype="S")
    var_alts = np.array([v.alt_alleles[0] for v in variants], dtype="S")
    
    # compute prediction window coordinates
    pred_starts = np.array([v.pos - params_model["seq_length"] // 2 + seq_crop for v in variants])
    pred_ends = np.array([v.pos - params_model["seq_length"] // 2 + seq_crop + site_length for v in variants])
    
    out_h5.create_dataset("variant_id", data=var_ids)
    out_h5.create_dataset("chrom", data=var_chroms)
    out_h5.create_dataset("pos", data=var_positions)
    out_h5.create_dataset("ref", data=var_refs)
    out_h5.create_dataset("alt", data=var_alts)
    out_h5.create_dataset("start", data=pred_starts)
    out_h5.create_dataset("end", data=pred_ends)

    #################################################################
    # predict scores and write output

    for vi in tqdm(range(num_variants), desc="Predicting variants"):
        # sequences are in pairs: ref at 2*vi, alt at 2*vi+1
        ref_seq_idx = 2 * vi
        alt_seq_idx = 2 * vi + 1
        
        # get pre-generated sequences
        ref_1hot = np.expand_dims(seqs_1hot[ref_seq_idx], axis=0)
        alt_1hot = np.expand_dims(seqs_1hot[alt_seq_idx], axis=0)
        
        # predict reference allele
        ref_preds = seqnn_model.predict(ref_1hot)[0]
        
        # apply untransform if requested
        if args.untransform_old:
            ref_preds = dataset.untransform_preds1(ref_preds, targets_df)
        else:
            ref_preds = dataset.untransform_preds(ref_preds, targets_df)

        # predict alternate allele
        alt_preds = seqnn_model.predict(alt_1hot)[0]
        
        # apply untransform if requested
        if args.untransform_old:
            alt_preds = dataset.untransform_preds1(alt_preds, targets_df)
        else:
            alt_preds = dataset.untransform_preds(alt_preds, targets_df)

        # clip to float16 range and write predictions
        preds_ref_write = np.clip(ref_preds, 0, np.finfo(np.float16).max)
        preds_alt_write = np.clip(alt_preds, 0, np.finfo(np.float16).max)
        
        out_h5["preds_ref"][vi] = preds_ref_write
        out_h5["preds_alt"][vi] = preds_alt_write

    # close output HDF5
    out_h5.close()
    
    print(f"\nCompleted predictions for {num_variants} variants")
    print(f"Results saved to: {out_h5_file}")
    print(f"Prediction window size: {site_length} bp ({site_preds_length} bins × {preds_window} bp)")
    print(f"Number of targets: {num_targets}")

################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main() 