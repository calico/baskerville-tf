import argparse
import json
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as tf_trt

from baskerville import seqnn


precision_dict = {
    "FP32": tf_trt.TrtPrecisionMode.FP32,
    "FP16": tf_trt.TrtPrecisionMode.FP16,
    "INT8": tf_trt.TrtPrecisionMode.INT8,
}


class ModelOptimizer:
    """
    Class of converter for tensorrt
    Args:
      input_saved_model_dir: Folder with saved model of the input model
    """

    def __init__(self, input_saved_model_dir, calibration_data=None):
        self.input_saved_model_dir = input_saved_model_dir
        self.calibration_data = None
        if calibration_data is not None:
            self.set_calibration_data(calibration_data)

    def set_calibration_data(self, calibration_data):
        def calibration_input_fn():
            yield (tf.constant(calibration_data.astype("float32")),)

        self.calibration_data = calibration_input_fn

    def convert(self, precision="FP32"):
        t0 = time.time()
        print("Converting the model.")

        if precision == "INT8" and self.calibration_data is None:
            raise (Exception("No calibration data set!"))

        trt_precision = precision_dict[precision]
        conversion_params = tf_trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
            precision_mode=trt_precision,
            use_calibration=precision == "INT8",
            max_workspace_size_bytes=8000000000,
        )
        self.converter = tf_trt.TrtGraphConverterV2(
            input_saved_model_dir=self.input_saved_model_dir,
            conversion_params=conversion_params,
        )

        if precision == "INT8":
            self.func = self.converter.convert(
                calibration_input_fn=self.calibration_data
            )
        else:
            self.func = self.converter.convert()
        print("Done in %ds" % (time.time() - t0))

    def build(self, seq_length):
        input_shape = (1, seq_length, 4)
        t0 = time.time()
        print("Building TRT engines for shape:", input_shape)

        def input_fn():
            x = np.random.random(input_shape).astype(np.float32)
            x = tf.cast(x, tf.float32)
            yield x

        self.converter.build(input_fn)
        print("Done in %ds" % (time.time() - t0))

    def build_func(self, seq_length):
        input_shape = (1, seq_length, 4)
        t0 = time.time()
        print("Building TRT engines for shape:", input_shape)
        x = np.random.random(input_shape)
        x = tf.cast(x, tf.float32)
        self.func(x)
        print("Done in %ds" % (time.time() - t0))

    def save(self, output_dir):
        self.converter.save(output_saved_model_dir=output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Convert a seqnn model to TensorRT model."
    )
    parser.add_argument(
        "-t", "--targets_file", default=None, help="Path to the target variants file"
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        default="trt_out",
        help="Output directory for storing saved models (original & converted)",
    )
    parser.add_argument(
        "params_file", type=str, help="Path to the JSON parameters file"
    )
    parser.add_argument("model_file", help="Trained model HDF5.")
    args = parser.parse_args()

    # Load parameters
    with open(args.params_file) as params_open:
        params = json.load(params_open)

    # Load keras model into seqnn class
    seqnn_model = seqnn.SeqNN(params["model"])
    seqnn_model.restore(args.model_file)

    # Load target variants
    if args.targets_file is not None:
        targets_df = pd.read_csv(args.targets_file, sep="\t", index_col=0)
        seqnn_model.build_slice(np.array(targets_df.index))

    # ensemble rc
    seqnn_model.build_ensemble(True)

    # save this model to a directory
    seqnn_model.ensemble.save(f"{args.out_dir}/original")

    # Convert the model
    opt_model = ModelOptimizer(f"{args.out_dir}/original")
    opt_model.convert(precision="FP32")
    # opt_model.build(seqnn_model.seq_length)
    opt_model.save(f"{args.out_dir}/convert")


if __name__ == "__main__":
    main()
