import tensorflow as tf
from tensorflow.python.saved_model import tag_constants


class OptimizedModel:
    """
    Class of model optimized with tensorrt
    Args:
      saved_model_dir: Folder with saved model
    """

    def __init__(self, saved_model_dir=None, strand_pair=[]):
        self.loaded_model_fn = None
        self.strand_pair = strand_pair
        if saved_model_dir is not None:
            self.load_model(saved_model_dir)

    def predict(self, input_data):
        if self.loaded_model_fn is None:
            raise (Exception("Haven't loaded a model"))
        x = tf.cast(input_data, tf.float32)
        labeling = self.loaded_model_fn(x)
        try:
            preds = labeling["predictions"].numpy()
        except:
            try:
                preds = labeling["probs"].numpy()
            except:
                try:
                    preds = labeling[next(iter(labeling.keys()))]
                except:
                    raise (
                        Exception("Failed to get predictions from saved model object")
                    )
        return preds

    def load_model(self, saved_model_dir):
        saved_model_loaded = tf.saved_model.load(
            saved_model_dir, tags=[tag_constants.SERVING]
        )
        wrapper_fp32 = saved_model_loaded.signatures["serving_default"]
        self.loaded_model_fn = wrapper_fp32

    def __call__(self, x):
        return self.predict(x)
