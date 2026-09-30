# Copyright 2023 Calico LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================
import glob
import json
import sys

from natsort import natsorted
import numpy as np
import pandas as pd
from scipy.sparse import dok_matrix
import tensorflow as tf

gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

# TFRecord constants
TFR_INPUT = "sequence"
TFR_OUTPUT = "target"


def file_to_records(filename: str):
    """Read TFRecord file into tf.data.Dataset."""
    return tf.data.TFRecordDataset(filename, compression_type="ZLIB")


class SeqDataset:
    """Labeled sequence dataset for Tensorflow.

    Args:
      data_dir (str): Dataset directory.
      split_label (str): Dataset split, e.g. train, valid, test.
      batch_size (int): Batch size.
      shuffle_buffer (int): Shuffle buffer size. Defaults to 128.
      seq_length_crop (int): Sequence length to crop from sides. Defaults to 0.
      mode (str): Dataset mode, e.g. train/eval. Defaults to 'eval'.
      tfr_pattern (str): TFRecord pattern to glob. Defaults to split_label.
      targets_slice_file (str): Targets table from which to slice a target subset.
    """

    def __init__(
        self,
        data_dir: str,
        split_label: str,
        batch_size: int,
        shuffle_buffer: int = 128,
        seq_length_crop: int = 0,
        mode: str = "eval",
        tfr_pattern: str = None,
        targets_slice_file: str = None,
    ):
        self.data_dir = data_dir
        self.split_label = split_label
        self.batch_size = batch_size
        self.shuffle_buffer = shuffle_buffer
        self.seq_length_crop = seq_length_crop
        self.mode = mode
        self.tfr_pattern = tfr_pattern

        # read data parameters
        data_stats_file = "%s/statistics.json" % self.data_dir
        with open(data_stats_file) as data_stats_open:
            data_stats = json.load(data_stats_open)
        self.seq_length = data_stats["seq_length"]

        # set defaults
        self.seq_depth = data_stats.get("seq_depth", 4)
        self.seq_1hot = data_stats.get("seq_1hot", False)
        self.target_length = data_stats["target_length"]
        self.num_targets = data_stats["num_targets"]
        self.pool_width = data_stats["pool_width"]

        # slice targets
        if targets_slice_file is None:
            self.targets_slice = None
        else:
            targets_df = pd.read_csv(targets_slice_file, index_col=0, sep="\t")
            self.targets_slice = np.array(targets_df.index)

        # extract or compute sequence statistics
        if self.tfr_pattern is None:
            self.tfr_path = "%s/tfrecords/%s-*.tfr" % (self.data_dir, self.split_label)
            self.num_seqs = data_stats["%s_seqs" % self.split_label]
        else:
            self.tfr_path = "%s/tfrecords/%s" % (self.data_dir, self.tfr_pattern)
            self.compute_stats()

        # make tf.data.Dataset object
        self.make_dataset()

    def batches_per_epoch(self):
        """Compute number of batches per epoch."""
        return self.num_seqs // self.batch_size

    def distribute(self, strategy):
        """Wrap Dataset to distribute across devices."""
        self.dataset = strategy.experimental_distribute_dataset(self.dataset)

    def generate_parser(self, raw: bool = False):
        """Generate parser function for TFRecordDataset."""

        def parse_proto(example_protos):
            """Parse TFRecord protobuf."""

            # define features
            features = {
                TFR_INPUT: tf.io.FixedLenFeature([], tf.string),
                TFR_OUTPUT: tf.io.FixedLenFeature([], tf.string),
            }

            # parse example into features
            parsed_features = tf.io.parse_single_example(
                example_protos, features=features
            )

            # decode sequence
            sequence = tf.io.decode_raw(parsed_features[TFR_INPUT], tf.uint8)
            if not raw:
                if self.seq_1hot:
                    sequence = tf.reshape(sequence, [self.seq_length])
                    sequence = tf.one_hot(sequence, 1 + self.seq_depth, dtype=tf.uint8)
                    sequence = sequence[:, :-1]  # drop N
                else:
                    sequence = tf.reshape(sequence, [self.seq_length, self.seq_depth])
                if self.seq_length_crop > 0:
                    crop_len = (self.seq_length - self.seq_length_crop) // 2
                    sequence = sequence[crop_len:-crop_len, :]
                sequence = tf.cast(sequence, tf.float32)

            # decode targets
            targets = tf.io.decode_raw(parsed_features[TFR_OUTPUT], tf.float16)
            if not raw:
                targets = tf.reshape(targets, [self.target_length, self.num_targets])
                targets = tf.cast(targets, tf.float32)
                if self.targets_slice is not None:
                    targets = targets[:, self.targets_slice]

            return sequence, targets

        return parse_proto

    def make_dataset(self, cycle_length=4):
        """Make tf.data.Dataset w/ transformations."""

        # initialize dataset from TFRecords glob
        tfr_files = natsorted(glob.glob(self.tfr_path))
        if tfr_files:
            dataset = tf.data.Dataset.from_tensor_slices(tfr_files)
        else:
            print("Cannot order TFRecords %s" % self.tfr_path, file=sys.stderr)
            dataset = tf.data.Dataset.list_files(self.tfr_path)

        # train
        if self.mode == "train":
            # repeat
            dataset = dataset.repeat()

            # interleave files
            dataset = dataset.interleave(
                map_func=file_to_records,
                cycle_length=cycle_length,
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )

            # shuffle
            dataset = dataset.shuffle(
                buffer_size=self.shuffle_buffer, reshuffle_each_iteration=True
            )

        # valid/test
        else:
            # flat mix files
            dataset = dataset.flat_map(file_to_records)

        # map parser across files
        dataset = dataset.map(self.generate_parser())

        # batch
        dataset = dataset.batch(self.batch_size)

        # prefetch
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        # hold on
        self.dataset = dataset

    def compute_stats(self):
        """Iterate over the TFRecords to count sequences, and infer
        seq_depth and num_targets."""
        with tf.name_scope("stats"):
            # read TF Records
            dataset = tf.data.Dataset.list_files(self.tfr_path)
            dataset = dataset.flat_map(file_to_records)
            dataset = dataset.map(self.generate_parser(raw=True))
            dataset = dataset.batch(1)

        self.num_seqs = 0
        if self.num_targets is not None:
            targets_nonzero = np.zeros(self.num_targets, dtype="bool")

        for seq_raw, targets_raw in dataset:
            # infer num_targets
            targets1 = targets_raw.numpy().reshape(self.target_length, -1)
            if self.num_targets is None:
                self.num_targets = targets1.shape[-1]
                targets_nonzero = (targets1 != 0).sum(axis=0) > 0
            else:
                assert self.num_targets == targets1.shape[-1]
                targets_nonzero = np.logical_or(
                    targets_nonzero, (targets1 != 0).sum(axis=0) > 0
                )

            # count sequences
            self.num_seqs += 1

        # warn user about nonzero targets
        if self.num_seqs > 0:
            self.num_targets_nonzero = (targets_nonzero > 0).sum()
            print(
                "%s has %d sequences with %d/%d targets"
                % (
                    self.tfr_path,
                    self.num_seqs,
                    self.num_targets_nonzero,
                    self.num_targets,
                ),
                flush=True,
            )
        else:
            self.num_targets_nonzero = None
            print(
                "%s has %d sequences with 0 targets" % (self.tfr_path, self.num_seqs),
                flush=True,
            )

    def numpy(
        self,
        return_inputs=True,
        return_outputs=True,
        step=1,
        target_slice=None,
        dtype="float16",
    ):
        """Convert TFR inputs and/or outputs to numpy arrays."""
        with tf.name_scope("numpy"):
            # initialize dataset from TFRecords glob
            tfr_files = natsorted(glob.glob(self.tfr_path))
            if tfr_files:
                # dataset = tf.data.Dataset.list_files(tf.constant(tfr_files), shuffle=False)
                dataset = tf.data.Dataset.from_tensor_slices(tfr_files)
            else:
                print("Cannot order TFRecords %s" % self.tfr_path, file=sys.stderr)
                dataset = tf.data.Dataset.list_files(self.tfr_path)

            # read TF Records
            dataset = dataset.flat_map(file_to_records)
            dataset = dataset.map(self.generate_parser(raw=True))
            dataset = dataset.batch(1)

        # initialize inputs and outputs
        seqs_1hot = []
        targets = []

        # collect inputs and outputs
        for seq_raw, targets_raw in dataset:
            # sequence
            if return_inputs:
                seq_1hot = seq_raw.numpy().reshape((self.seq_length, -1))
                if self.seq_length_crop > 0:
                    crop_len = (self.seq_length - self.seq_length_crop) // 2
                    seq_1hot = seq_1hot[crop_len:-crop_len, :]
                seqs_1hot.append(seq_1hot)

            # targets
            if return_outputs:
                targets1 = targets_raw.numpy().astype(dtype)
                targets1 = np.reshape(targets1, (self.target_length, -1))
                if target_slice is not None:
                    targets1 = targets1[:, target_slice]
                if step > 1:
                    step_i = np.arange(0, self.target_length, step)
                    targets1 = targets1[step_i, :]
                targets.append(targets1)

        # make arrays
        seqs_1hot = np.array(seqs_1hot)
        targets = np.array(targets, dtype=dtype)

        # return
        if return_inputs and return_outputs:
            return seqs_1hot, targets
        elif return_inputs:
            return seqs_1hot
        else:
            return targets


def make_strand_transform(targets_df, targets_strand_df):
    """Make a sparse matrix to sum strand pairs.

    Args:
        targets_df (pd.DataFrame): Targets DataFrame.
        targets_strand_df (pd.DataFrame): Targets DataFrame, with strand pairs collapsed.

    Returns:
        scipy.sparse.dok_matrix: Sparse matrix to sum strand pairs.
    """

    # initialize sparse matrix
    strand_transform = dok_matrix((targets_df.shape[0], targets_strand_df.shape[0]))

    # track which strand pairs we've seen
    seen_pairs = set()

    # fill in matrix
    ti = 0
    sti = 0
    for _, target in targets_df.iterrows():
        strand_transform[ti, sti] = True
        if target.strand_pair == target.name:
            # Unstranded target
            sti += 1
        else:
            # Stranded target - check if we've seen its pair
            if target.strand_pair in seen_pairs:
                # This is the second member of the pair, increment sti
                sti += 1
            else:
                # This is the first member of the pair, mark it as seen
                seen_pairs.add(target.name)
        ti += 1

    return strand_transform


def targets_prep_strand(targets_df):
    """Adjust targets table for merged stranded datasets.

    Args:
        targets_df: pandas DataFrame of targets

    Returns:
        targets_df: pandas DataFrame of targets, with stranded
            targets collapsed into a single row
    """
    # attach strand
    targets_strand = []
    for _, target in targets_df.iterrows():
        if target.strand_pair == target.name:
            targets_strand.append(".")
        else:
            targets_strand.append(target.identifier[-1])
    targets_df["strand"] = targets_strand

    # collapse stranded
    strand_mask = targets_df.strand != "-"
    targets_strand_df = targets_df[strand_mask]

    return targets_strand_df


def untransform_preds(preds, targets_df, unscale=False, unclip=True):
    """Undo the squashing transformations performed for the tasks.

    Args:
      preds (np.array): Predictions LxT.
      targets_df (pd.DataFrame): Targets information table.

    Returns:
      preds (np.array): Untransformed predictions LxT.
    """
    # clip soft
    if unclip:
        cs = np.expand_dims(np.array(targets_df.clip_soft), axis=0)
        preds_unclip = cs - 1 + (preds - cs + 1) ** 2
        preds = np.where(preds > cs, preds_unclip, preds)

    # sqrt
    sqrt_mask = np.array([ss.find("_sqrt") != -1 for ss in targets_df.sum_stat])
    preds[:, sqrt_mask] = -1 + (preds[:, sqrt_mask] + 1) ** 2  # (4 / 3)

    # scale
    if unscale:
        scale = np.expand_dims(np.array(targets_df.scale), axis=0)
        preds = preds / scale

    return preds


def untransform_preds1(preds, targets_df, unscale=False, unclip=True):
    """Undo the squashing transformations performed for the tasks.

    Args:
      preds (np.array): Predictions LxT.
      targets_df (pd.DataFrame): Targets information table.

    Returns:
      preds (np.array): Untransformed predictions LxT.
    """
    # scale
    scale = np.expand_dims(np.array(targets_df.scale), axis=0)
    preds = preds / scale

    # clip soft
    if unclip:
        cs = np.expand_dims(np.array(targets_df.clip_soft), axis=0)
        preds_unclip = cs + (preds - cs) ** 2
        preds = np.where(preds > cs, preds_unclip, preds)

    # ** 0.75
    sqrt_mask = np.array([ss.find("_sqrt") != -1 for ss in targets_df.sum_stat])
    preds[:, sqrt_mask] = (preds[:, sqrt_mask]) ** (4 / 3)

    # unscale
    if not unscale:
        preds = preds * scale

    return preds
