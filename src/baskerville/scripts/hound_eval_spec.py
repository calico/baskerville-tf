#!/usr/bin/env python
# Copyright 2020 Calico LLC
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
from tqdm import tqdm

import numpy as np
import pandas as pd
from qnorm import quantile_normalize
from scipy.stats import pearsonr
from tensorflow.keras import mixed_precision

from baskerville import dataset
from baskerville import seqnn

"""
hound_eval_spec.py

Test the accuracy of a trained model on targets/predictions normalized across targets.
"""


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument(
        "--f16",
        default=False,
        action="store_true",
        help="use mixed precision for inference [Default: %(default)s]",
    )
    parser.add_argument(
        "--head",
        dest="head_i",
        default=0,
        type=int,
        help="Parameters head to evaluate [Default: %(default)s]",
    )
    parser.add_argument(
        "-m",
        "--group_min",
        default=20,
        type=int,
        help="Minimum target group size to consider [Default: %(default)s]",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        default="spec_out",
        help="Output directory for evaluation statistics [Default: %(default)s]",
    )
    parser.add_argument(
        "--rank",
        default=False,
        action="store_true",
        help="Compute Spearman rank correlation [Default: %(default)s]",
    )
    parser.add_argument(
        "--rc",
        default=False,
        action="store_true",
        help="Average the fwd and rc predictions [Default: %(default)s]",
    )
    parser.add_argument(
        "--save",
        default=False,
        action="store_true",
        help="Save targets and predictions numpy arrays [Default: %(default)s]",
    )
    parser.add_argument(
        "--shifts",
        default="0",
        help="Ensemble prediction shifts [Default: %(default)s]",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split label for eg TFR pattern [Default: %(default)s]",
    )
    parser.add_argument(
        "--step",
        default=1,
        type=int,
        help="Step across positions [Default: %(default)s]",
    )
    parser.add_argument(
        "-t",
        "--targets_file",
        default=None,
        help="File specifying target indexes and labels in table format",
    )
    parser.add_argument(
        "--target_groups",
        default=None,
        type=str,
        help="Comma separated string of target groups",
    )
    parser.add_argument(
        "--tfr",
        default=None,
        help="Subsetting TFR pattern appended to data_dir/tfrecords [Default: %(default)s]",
    )
    parser.add_argument(
        "--var_pct",
        default=1.0,
        type=float,
        help="Highly variable site proportion to take [Default: %default]",
    )
    parser.add_argument("params_file", help="JSON file with model parameters")
    parser.add_argument("model_file", help="Trained model file.")
    parser.add_argument("data_dir", help="Train/valid/test data directory")
    args = parser.parse_args()

    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)

    # parse shifts to integers
    args.shifts = [int(shift) for shift in args.shifts.split(",")]

    #######################################################
    # targets

    # read table
    if args.targets_file is None:
        args.targets_file = f"{args.data_dir}/targets.txt"
    targets_df = pd.read_csv(args.targets_file, index_col=0, sep="\t")
    num_targets = targets_df.shape[0]

    # set target groups
    if "group" not in targets_df.columns:
        targets_group = []
        for ti in range(num_targets):
            description = targets_df.iloc[ti].description
            if description.find(":") == -1:
                tg = "*"
            else:
                desc_split = description.split(":")
                if desc_split[0] == "CHIP":
                    tg = "/".join(desc_split[:2])
                else:
                    tg = desc_split[0]
            targets_group.append(tg)
        targets_df["group"] = targets_group

    if args.target_groups is None:
        args.target_groups = sorted(set(targets_df.group))

    #######################################################
    # setup

    # read parameters
    with open(args.params_file) as params_open:
        params = json.load(params_open)
    params_model = params["model"]
    params_train = params["train"]

    # update strand pairs for new indexing
    orig_new_index = dict(zip(targets_df.index, np.arange(targets_df.shape[0])))
    targets_strand_pair = [orig_new_index[ti] for ti in targets_df.strand_pair]
    params_model["strand_pair"] = [np.array(targets_strand_pair)]

    # construct eval data
    eval_data = dataset.SeqDataset(
        args.data_dir,
        split_label=args.split,
        batch_size=params_train["batch_size"],
        mode="eval",
        tfr_pattern=args.tfr,
    )

    # initialize model
    if args.f16:
        mixed_precision.set_global_policy("mixed_float16")  # set global policy
        seqnn_model = seqnn.SeqNN(params_model)  # create model
        seqnn_model.restore(args.model_file, args.head_i)
        seqnn_model.append_activation()  # add additional activation to cast float16 output to float32
    else:
        # initialize model
        seqnn_model = seqnn.SeqNN(params_model)
        seqnn_model.restore(args.model_file, args.head_i)
    seqnn_model.build_slice(targets_df.index)
    seqnn_model.build_ensemble(args.rc, args.shifts)
    if args.step > 1:
        seqnn_model.step(args.step)
        step_i = np.arange(0, eval_data.target_length, args.step)

    #######################################################
    # targets/predictions

    # predict
    t0 = time.time()
    print("Model predictions...", flush=True, end="")
    eval_preds = []
    eval_targets = []

    for x, y in tqdm(eval_data.dataset):
        # predict
        yh = seqnn_model(x)
        eval_preds.append(yh)

        y = y.numpy().astype("float16")
        y = y[:, :, np.array(targets_df.index)]
        if args.step > 1:
            y = y[:, step_i, :]
        eval_targets.append(y)

        gc.collect()

    # flatten
    eval_preds = np.concatenate(eval_preds, axis=0)
    eval_targets = np.concatenate(eval_targets, axis=0)
    print("DONE in %ds" % (time.time() - t0))
    print("targets", eval_targets.shape)

    #######################################################
    # process groups

    targets_spec = np.zeros(num_targets)

    for tg in args.target_groups:
        group_mask = np.array(targets_df.group == tg)
        group_df = targets_df[group_mask]
        num_targets_group = group_mask.sum()
        print("%-15s  %4d" % (tg, num_targets_group), flush=True)

        if num_targets_group < args.group_min:
            targets_spec[group_mask] = np.nan

        else:
            # slice group
            eval_preds_group = eval_preds[:, :, group_mask]
            eval_preds_group = eval_preds_group.reshape((-1, num_targets_group))
            eval_targets_group = eval_targets[:, :, group_mask]
            eval_targets_group = eval_targets_group.reshape((-1, num_targets_group))

            # fix stranded
            stranded = False
            if "strand_pair" in group_df.columns:
                stranded = (group_df.strand_pair != group_df.index).all()
            if stranded:
                # reshape to concat +/-, assuming they're adjacent
                num_targets_group //= 2
                eval_preds_group = np.reshape(eval_preds_group, (-1, num_targets_group))
                eval_targets_group = np.reshape(
                    eval_targets_group, (-1, num_targets_group)
                )

            # quantile normalize
            t0 = time.time()
            print(" Quantile normalize...", flush=True, end="")
            eval_preds_norm = quantile_normalize(eval_preds_group, ncpus=2)
            del eval_preds_group
            eval_targets_norm = quantile_normalize(eval_targets_group, ncpus=2)
            del eval_targets_group
            print("DONE in %ds" % (time.time() - t0))

            # upcast
            eval_preds_norm = eval_preds_norm.astype(np.float32)
            eval_targets_norm = eval_targets_norm.astype(np.float32)

            # highly variable filter
            if args.var_pct < 1:
                t0 = time.time()
                print(" Highly variable position filter...", flush=True, end="")
                eval_targets_var = eval_targets_group.var(axis=1)
                high_var_t = np.percentile(eval_targets_var, 100 * (1 - args.var_pct))
                high_var_mask = eval_targets_var >= high_var_t
                eval_preds_norm = eval_preds_norm[high_var_mask]
                eval_targets_norm = eval_targets_norm[high_var_mask]
                print("DONE in %ds" % (time.time() - t0))

            # mean normalize
            eval_preds_norm -= eval_preds_norm.mean(axis=-1, keepdims=True)
            eval_targets_norm -= eval_targets_norm.mean(axis=-1, keepdims=True)

            # compute correlations
            t0 = time.time()
            print(" Compute correlations...", flush=True, end="")
            pearsonr_group = np.zeros(num_targets_group)
            for ti in range(num_targets_group):
                eval_preds_norm_ti = eval_preds_norm[:, ti]
                eval_targets_norm_ti = eval_targets_norm[:, ti]
                pearsonr_group[ti] = pearsonr(eval_preds_norm_ti, eval_targets_norm_ti)[
                    0
                ]
            print("DONE in %ds" % (time.time() - t0))

            if stranded:
                pearsonr_group = np.repeat(pearsonr_group, 2)

            # save
            targets_spec[group_mask] = pearsonr_group

            # print
            print(" PearsonR %.4f" % pearsonr_group[ti], flush=True)

            # clean
            del eval_preds_norm
            del eval_targets_norm
            gc.collect()

    # write target-level statistics
    targets_acc_df = pd.DataFrame(
        {
            "index": targets_df.index,
            "pearsonr": targets_spec,
            "identifier": targets_df.identifier,
            "description": targets_df.description,
        }
    )
    acc_file = f"{args.out_dir}/acc.txt"
    targets_acc_df.to_csv(acc_file, sep="\t", index=False, float_format="%.5f")


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
