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
import shutil

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import mixed_precision

from baskerville import dataset
from baskerville import seqnn
from baskerville import trainer
from baskerville import transfer

"""
hound_transfer.py

Modified from hound_train.py.
Additional argument to allow for transfer learning from existing Hound model.
"""


def main():
    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument(
        "-k",
        "--keras_fit",
        action="store_true",
        default=False,
        help="Train with Keras fit method [Default: %(default)s]",
    )
    parser.add_argument(
        "-m",
        "--mixed_precision",
        action="store_true",
        default=False,
        help="Train with mixed precision [Default: %(default)s]",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        default="train_out",
        help="Output directory [Default: %(default)s]",
    )
    parser.add_argument(
        "-l",
        "--log_dir",
        default="log_out",
        help="Tensorboard log directory [Default: %(default)s]",
    )
    parser.add_argument(
        "--restore",
        default=None,
        help="model trunk h5 file [Default: %(default)s]",
    )
    parser.add_argument(
        "--trunk",
        action="store_true",
        default=False,
        help="Restore only model trunk [Default: %(default)s]",
    )
    parser.add_argument(
        "--tfr_train",
        default=None,
        help="Training TFR pattern string appended to data_dir/tfrecords [Default: %(default)s]",
    )
    parser.add_argument(
        "--tfr_eval",
        default=None,
        help="Evaluation TFR pattern string appended to data_dir/tfrecords [Default: %(default)s]",
    )

    parser.add_argument("params_file", help="JSON file with model parameters")

    parser.add_argument(
        "data_dirs", nargs="+", help="Train/valid/test data directorie(s)"
    )
    parser.add_argument(
        "--skip_train",
        action="store_true",
        default=False,
        help="report trainable params and skip training [Default: %(default)s]",
    )
    args = parser.parse_args()

    if args.keras_fit and len(args.data_dirs) > 1:
        print("Cannot use keras fit method with multi-genome training.")
        exit()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.params_file != "%s/params.json" % args.out_dir:
        shutil.copy(args.params_file, "%s/params.json" % args.out_dir)

    # model parameters
    with open(args.params_file) as params_open:
        params = json.load(params_open)

    params_model = params["model"]

    # train parameters
    params_train = params["train"]

    # transfer parameters
    params_transfer = params["transfer"]
    transfer_mode = params_transfer.get("mode", "full")
    transfer_adapter = params_transfer.get("adapter", None)
    transfer_latent = params_transfer.get("adapter_latent", 8)
    transfer_conv_select = params_transfer.get("conv_select", 4)
    transfer_conv_rank = params_transfer.get("conv_latent", 4)
    transfer_lora_alpha = params_transfer.get("lora_alpha", 16)
    transfer_locon_alpha = params_transfer.get("locon_alpha", 1)

    if transfer_mode not in ["full", "linear", "adapter"]:
        raise ValueError("transfer mode must be one of full, linear, adapter")

    # read datasets
    train_data = []
    eval_data = []
    strand_pairs = []

    for data_dir in args.data_dirs:
        # set strand pairs
        targets_df = pd.read_csv("%s/targets.txt" % data_dir, sep="\t", index_col=0)
        if "strand_pair" in targets_df.columns:
            strand_pairs.append(np.array(targets_df.strand_pair))

        # load train data
        train_data.append(
            dataset.SeqDataset(
                data_dir,
                split_label="train",
                batch_size=params_train["batch_size"],
                shuffle_buffer=params_train.get("shuffle_buffer", 128),
                mode="train",
                tfr_pattern=args.tfr_train,
            )
        )

        # load eval data
        eval_data.append(
            dataset.SeqDataset(
                data_dir,
                split_label="valid",
                batch_size=params_train["batch_size"],
                mode="eval",
                tfr_pattern=args.tfr_eval,
            )
        )

    params_model["strand_pair"] = strand_pairs

    if args.mixed_precision:
        mixed_precision.set_global_policy("mixed_float16")

    if params_train.get("num_gpu", 1) == 1:
        ########################################
        # one GPU

        # initialize model
        params_model["verbose"] = False
        seqnn_model = seqnn.SeqNN(params_model)

        # restore
        if args.restore:
            seqnn_model.restore(args.restore, trunk=args.trunk)

        # head params
        print(
            "params in new head: %d"
            % transfer.param_count(seqnn_model.model.layers[-2])
        )

        ####################
        # transfer options #
        ####################
        if transfer_mode == "full":
            seqnn_model.model.trainable = True

        elif transfer_mode == "linear":
            seqnn_model.model_trunk.trainable = False

        ############
        # adapters #
        ############
        elif transfer_mode == "adapter":
            # attention adapter
            if transfer_adapter is not None:
                if transfer_adapter == "houlsby":
                    seqnn_model.model = transfer.add_houlsby(
                        seqnn_model.model, strand_pairs[0], latent_size=transfer_latent
                    )
                elif transfer_adapter == "lora":
                    transfer.add_lora(
                        seqnn_model.model,
                        rank=transfer_latent,
                        alpha=transfer_lora_alpha,
                        mode="default",
                    )

                elif transfer_adapter == "lora_full":
                    transfer.add_lora(
                        seqnn_model.model,
                        rank=transfer_latent,
                        alpha=transfer_lora_alpha,
                        mode="full",
                    )

                elif transfer_adapter == "ia3":
                    seqnn_model.model = transfer.add_ia3(
                        seqnn_model.model, strand_pairs[0]
                    )

                elif transfer_adapter == "locon":  # lora on conv+att
                    seqnn_model.model = transfer.add_locon(
                        seqnn_model.model,
                        strand_pairs[0],
                        conv_select=transfer_conv_select,
                        rank=transfer_conv_rank,
                        alpha=transfer_locon_alpha,
                    )

                elif transfer_adapter == "lora_conv":  # lora on att, unfreeze_conv
                    transfer.add_lora_conv(
                        seqnn_model.model, conv_select=transfer_conv_select
                    )

                elif transfer_adapter == "houlsby_se":  # adapter on conv+att
                    seqnn_model.model = transfer.add_houlsby_se(
                        seqnn_model.model,
                        strand_pair=strand_pairs[0],
                        conv_select=transfer_conv_select,
                        se_rank=transfer_conv_rank,
                    )

        #################
        # final summary #
        #################
        seqnn_model.model.summary()

        if args.mixed_precision:
            # add additional activation to cast float16 output to float32
            seqnn_model.append_activation()
            # run with loss scaling
            seqnn_trainer = trainer.Trainer(
                params_train,
                train_data,
                eval_data,
                args.out_dir,
                args.log_dir,
                loss_scale=True,
            )
        else:
            seqnn_trainer = trainer.Trainer(
                params_train, train_data, eval_data, args.out_dir, args.log_dir
            )

        # compile model
        seqnn_trainer.compile(seqnn_model)

        if args.skip_train:
            exit(0)

        # train model
        if args.keras_fit:
            seqnn_trainer.fit_keras(seqnn_model)
        else:
            if len(args.data_dirs) == 1:
                seqnn_trainer.fit_tape(seqnn_model)
            else:
                seqnn_trainer.fit2(seqnn_model)

        #############################
        # post-training adjustments #
        #############################
        if transfer_mode == "adapter":
            # for: houlsby and houlsby_se, overwrite json file
            if transfer_adapter == "houlsby":
                transfer.modify_json(
                    input_json=args.params_file,
                    output_json="%s/params.json" % args.out_dir,
                    adapter=transfer_adapter,
                    latent=transfer_latent,
                )

            if transfer_adapter == "houlsby_se":
                transfer.modify_json(
                    input_json=args.params_file,
                    output_json="%s/params.json" % args.out_dir,
                    adapter=transfer_adapter,
                    conv_select=transfer_conv_select,
                    se_rank=transfer_conv_rank,
                )

            # for lora, ia3, locon, save weight to: model_best.mergeW.h5
            if transfer_adapter in ["lora", "lora_full", "lora_conv"]:
                seqnn_model.model.load_weights("%s/model_best.h5" % args.out_dir)
                transfer.merge_lora(seqnn_model.model)
                seqnn_model.save("%s/model_best.mergeW.h5" % args.out_dir)
                transfer.var_reorder("%s/model_best.mergeW.h5" % args.out_dir)

            if transfer_adapter == "ia3":
                # ia3 model
                ia3_model = seqnn_model.model
                ia3_model.load_weights("%s/model_best.h5" % args.out_dir)
                # original model
                seqnn_model2 = seqnn.SeqNN(params_model)
                seqnn_model2.restore(args.restore, trunk=args.trunk)
                original_model = seqnn_model2.model
                # merge weights into original model
                transfer.merge_ia3(original_model, ia3_model)
                original_model.save("%s/model_best.mergeW.h5" % args.out_dir)

            if transfer_adapter == "locon":
                # locon model
                locon_model = seqnn_model.model
                locon_model.load_weights("%s/model_best.h5" % args.out_dir)
                # original model
                seqnn_model2 = seqnn.SeqNN(params_model)
                seqnn_model2.restore(args.restore, trunk=args.trunk)
                original_model = seqnn_model2.model
                # merge weights into original model
                transfer.merge_locon(original_model, locon_model)
                original_model.save("%s/model_best.mergeW.h5" % args.out_dir)

    else:
        ########################################
        # multi GPU

        strategy = tf.distribute.MirroredStrategy()

        with strategy.scope():
            if not args.keras_fit:
                # distribute data
                for di in range(len(args.data_dirs)):
                    train_data[di].distribute(strategy)
                    eval_data[di].distribute(strategy)

            # initialize model
            seqnn_model = seqnn.SeqNN(params_model)

            # restore
            if args.restore:
                seqnn_model.restore(args.restore, trunk=args.trunk)

            # initialize trainer
            seqnn_trainer = trainer.Trainer(
                params_train,
                train_data,
                eval_data,
                args.out_dir,
                strategy,
                params_train["num_gpu"],
                args.keras_fit,
            )

            # compile model
            seqnn_trainer.compile(seqnn_model)

        # train model
        if args.keras_fit:
            seqnn_trainer.fit_keras(seqnn_model)
        else:
            if len(args.data_dirs) == 1:
                seqnn_trainer.fit_tape(seqnn_model)
            else:
                seqnn_trainer.fit2(seqnn_model)


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
