## Transfer Learning Tutorial (Mouse mm10 tracks)

### Required Software
- baskerville-tf

### Prepare Files

To prepare the tracks to w5 file, see the [Document page for transfer learning to hg38 tracks](../transfer_human/transfer.md).

- setup_folds.py from [transfer_human directory](../transfer_human/) - used to set up folder structure for training.

Set data_path to your preferred directory:

```bash
baskerville_path='/home/<user_path>/baskerville-tf'
data_path='/home/yuanh/analysis/Borzoi_transfer/tutorial/mouse/data'

mkdir -p ${data_path}
```

Download four replicate Borzoi pre-trained model trunks:

```bash
gsutil cp -r gs://scbasset_tutorial_data/baskerville_transfer_mouse/pretrain_trunks/ ${data_path}
```
Note:
- Four replicate models have identical train, validation and test splits (test on fold3, validation on fold4, trained on rest). More details in the Borzoi manuscript.
- Fold splits can be found in trainsplit/sequences.bed.
- Model trunk refers to the model weights without the final dense layer (head).


Download mm10 reference information, and train-validation-test-split information:

```bash
gsutil cp -r gs://scbasset_tutorial_data/baskerville_transfer_mouse/mm10/ ${data_path}
gsutil cp -r gs://scbasset_tutorial_data/baskerville_transfer_mouse/trainsplit/ ${data_path}
gunzip ${data_path}/mm10/mm10.ml.fa.gz
```

### Make Target File

Prepare track.w5 files the same way we described in the transfer-to-human-track tutorial.
For parameters of clip, clip_soft, scale, sum_stat etc for diferent modalities, see original Borzoi targets_mouse.txt file as an example:
```bash
gsutil cp -r gs://scbasset_tutorial_data/baskerville_transfer_mouse/borzoi_targets_mouse.txt ${data_path}
```

### Create TFRecords

```bash
cp ${baskerville_path}/docs/transfer_mouse/make_tfr_mouse.sh ./
# change data_path in make_tfr_mouse.sh
./make_tfr_mouse.sh
```

### Parameter Json File

Copy over the json file for transfer (Locon4). Since the transferred model will be single headed, there is no distinction of human versus mouse head. We will keep the head name as head_human in the json file. 

```bash
cp ${baskerville_path}/docs/transfer_human/params.json ./
```

In the json file, change the final head output units to the number of tracks in targets file.

### Transfer learning

The transfer script will be the same as transfer-to-human-track tutorial.

```bash
../transfer_human/setup_folds.py \
  -o train -f 4 \
  params.json \
  ${data_path}/tfr
```

Run hound_transfer.py on training data in fold3 folder (identical to pre-train split) for four replicate models:

```bash
hound_transfer.py -o train_rep0 --trunk --restore ${data_path}/pretrain_trunks/trunk_r0.h5 params.json train/f3c0/data0
hound_transfer.py -o train_rep1 --trunk --restore ${data_path}/pretrain_trunks/trunk_r1.h5 params.json train/f3c0/data0
hound_transfer.py -o train_rep2 --trunk --restore ${data_path}/pretrain_trunks/trunk_r2.h5 params.json train/f3c0/data0
hound_transfer.py -o train_rep3 --trunk --restore ${data_path}/pretrain_trunks/trunk_r3.h5 params.json train/f3c0/data0
```

### Load models

We apply weight merging for lora, ia3, and locon weights, and so there is no architecture changes once the model is trained. You can use the same params.json file, and load the train_rep0/model_best.mergeW.h5 weight file.

For houlsby and houlsby_se, model architectures change due to the insertion of adapter modules. New architecture json file is auto-generated in train_rep0/params.json;






