# baskerville-tf

> **Superseded** by [baskerville](https://github.com/calico/baskerville) (PyTorch). This TensorFlow version is maintained for [borzoi](https://github.com/calico/borzoi) reproducibility.

#### Sequential regulatory activity predictions with deep convolutional neural networks.

baskerville-tf provides researchers with tools to:

1. Train deep convolutional neural networks to predict regulatory activity along very long chromosome-scale DNA sequences
2. Score variants according to their predicted influence on regulatory activity across the sequence and/or for specific genes.
3. Annotate the specific nucleotides that drive regulatory element function.

---

### Documentation

Documentation page: https://calico.github.io/baskerville-tf/index.html

- [Document page for transfer learning to hg38 tracks](docs/transfer_human/transfer.md)
- [Document page for transfer learning to mm10 tracks](docs/transfer_mouse/transfer_mouse.md)

---

### Installation

```sh
git clone git@github.com:calico/baskerville-tf.git
cd baskerville-tf
pip install .
```

The package installs as `baskerville-tf` and imports as `baskerville`. The command-line scripts live in `src/baskerville/scripts`; put them on your path:
```sh
export BASKERVILLE_DIR=/home/<user_path>/baskerville-tf
export PATH=$BASKERVILLE_DIR/src/baskerville/scripts:$PATH
export PYTHONPATH=$BASKERVILLE_DIR/src/baskerville/scripts:$PYTHONPATH
```

---

#### Contacts

Dave Kelley (codeowner)
