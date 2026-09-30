## Transfer Learning Tutorial (Human hg38 tracks)

### Required Software
- baskerville-tf
- bamCoverage from [deepTools](https://github.com/deeptools/deepTools/tree/master) is required to make BigWig files.

### Download Tutorial Data


Set data_path to your preferred directory:

```bash
baskerville_path='/path/to/your/baskerville-tf'  # Update this to your baskerville-tf clone path
data_path='/path/to/your/data'  # Update this to your preferred data directory
bam_folder=${data_path}/bam
bw_folder=${data_path}/bw
w5_folder=${data_path}/w5

mkdir -p ${data_path}
```

Download four replicate Borzoi pre-trained model trunks:

```bash
gsutil cp -r gs://scbasset_tutorial_data/baskerville_transfer/pretrain_trunks/ ${data_path}
```
Note:
- Four replicate models have identical train, validation and test splits (test on fold3, validation on fold4, trained on rest). More details in the Borzoi manuscript.
- Fold splits can be found in trainsplit/sequences.bed.
- Model trunk refers to the model weights without the final dense layer (head).


Download hg38 reference information, and train-validation-test-split information:

```bash
gsutil cp -r gs://scbasset_tutorial_data/baskerville_transfer/hg38/ ${data_path}
gsutil cp -r gs://scbasset_tutorial_data/baskerville_transfer/trainsplit/ ${data_path}
gunzip ${data_path}/hg38/hg38.ml.fa.gz
```

Follow `Step 1` to generate BigWig from BAM files. Or, for the purpose of this tutorial, download CPM normalized stranded BigWig files for wild-type (PDL20_TP1_A) and senescent (PDL50_TP7_C) WI38 cell RNA-seq directly, and skip `Step 1`.

```bash
gsutil cp -r gs://scbasset_tutorial_data/baskerville_transfer/bw/ ${data_path}
```

### Step 1 (Optional): Convert BAM to BigWig Files

When you start from bam files, you can first create stranded/unstranded BigWig files depending on the RNA-seq protocol is stranded or not:

```bash
for file in ${bam_folder}/*.bam
do
  bam=`basename ${file}`
  bamCoverage --filterRNAstrand forward --binSize 1 --normalizeUsing CPM --skipNAs -p 16 -b ${bam_folder}/${bam} -o ${bw_folder}/${bam/.bam/}+.bw
  bamCoverage --filterRNAstrand reverse --binSize 1 --normalizeUsing CPM --skipNAs -p 16 -b ${bam_folder}/${bam} -o  ${bw_folder}/${bam/.bam/}-.bw
  echo ${bam}
done
```
`Note`: When working with 10x single-cell RNA-seq data, the strands are flipped. In this case, `--filterRNAstrand forward` refers to the reverse strand, and `--filterRNAstrand reverse` refers to the forward strand.

Or created unstranded BigWig files:
```bash
for file in ${bam_folder}/*.bam
do
  bam=`basename ${file}`
  bamCoverage --binSize 1 --normalizeUsing CPM --skipNAs -p 16 -b ${bam_folder}/${bam} -o ${bw_folder}/${bam/.bam/}+.bw
  echo ${bam}
done
```

### Step 2. Convert BigWig Files to Compressed hdf5 Format (w5) Files

Convert BigWig files to compressed hdf5 format (.w5).

```bash
mkdir ${w5_folder}
for file in ${bw_folder}/*.bw
do
  bw=$(basename "${file}")
  ${baskerville_path}/src/baskerville/scripts/utils/bw_w5.py ${bw_folder}/${bw} ${w5_folder}/${bw/.bw/.w5}
  echo ${bw}
done
```

`Note:` if your BAM/BigWig file chromosomes names are 1, 2, 3, etc (instead of chr1, chr2, chr3, etc), make sure to run the bw_w5.py script with the --chr_prepend option. This will prepend 'chr' to the chromosome names before converting the files to .w5 format.

### Step 3. Make Target File

We have provided the target file for this tutorial example.

Create *targets.txt*:
- (unnamed) => integer index of each track (must start from 0 when training a new model).
- 'identifier' => unique identifier of each experiment (and strand).
- 'file' => local file path to .w5 file.
- 'clip' => hard clipping threshold to be applied to each bin, after soft-clipping (default: 768).
- 'clip_soft' => soft clipping (squashing) threshold (default: 384).
- 'scale' => scale value applied to each bp-level position before clipping (see more detaile below).
- 'sum_stat' => type of bin-level pooling operation (default: 'sum_sqrt', sum and square-root).
- 'strand_pair' => integer index of the other stranded track of an experiment (same index as current row if unstranded).
- 'description' => text description of experiment.

**Note on 'scale':** A scaling factor is applied when creating the TFRecord data. Borzoi models use Poisson and multinomial losses. Input BigWig/W5 tracks are scaled so that one fragment is counted as one event, with each bp position of the fragment contributing 1/(frag length). As a result, the total coverage across the genome should sum to the read depth of the sample.

- If you start with BAM files, you can make BigWig files with option `--normalizeUsing None` in `bamCoverage`. Find out the fragment length by `samtools stats x.bam|grep "average length"` And then set the scaling factor to 1/(frag length).
- For standard BigWig tracks that are TPM normalized, it sums up to (frag length) * 1e6. When fragment length and library size are unknown for your RNA-seq data (e.g. when you only have RPM normalized BigWig data), we typically assume fragment length of 100, and library size of 30 million reads. Thus, for RPM normalized BigWig files, we set a scaling factor of 30/100 = 0.3.


### Step 4. Create TFRecords

```bash
cp ${baskerville_path}/docs/transfer_human/make_tfr.sh ./
# Update data_path variable in make_tfr.sh to match your data directory
./make_tfr.sh
```

Note: Make sure to edit the `data_path` variable in `make_tfr.sh` to match your directory structure before running.

### Step 5. Parameter Json File

Similar to Borzoi training, arguments for transfer learning are specified in the params.json file. Add an additional `transfer` section in the parameter json file to allow transfer learning. For the transfer learning rate, we suggest lowering the lr to 1e-5 for full fine-tuning, and keeping the original lr for other methods. For batch size, we suggest a batch size of 1 to reduce GPU memory for linear probing or adapter-based methods. Here's the `transfer` arguments for different transfer methods. 

Example params.json files for transfer learning of Borzoi-lite are located: baskerville-tf/tests/data/transfer/json/borzoilite_\*.json

Example params.json files for transfer learning of full Borzoi are located: baskerville-tf/tests/data/transfer/json/borzoi_\*.json


**Full fine-tuning**:
```
    "transfer": {
        "mode": "full"
    },
```

**Linear probing**:
```
    "transfer": {
        "mode": "linear"
    },    
```

**LoRA**:
```
    "transfer": {
        "mode": "adapter",
        "adapter": "lora",
        "adapter_latent": 8
    },
```

**Locon4**:
```
    "transfer": {
        "mode": "adapter",
        "adapter": "locon",
        "conv_select": 4
    },
```

**Houlsby**:
```
    "transfer": {
        "mode": "adapter",
        "adapter": "houlsby",
        "adapter_latent": 8
    },
```

**Houlsby_se4**:
```
    "transfer": {
        "mode": "adapter",
        "adapter": "houlsby_se",
        "adapter_latent": 8,
        "conv_select": 4,
        "conv_latent": 16
    },
```
### Step 6. Train model

Run setup_folds.py to setup directory structures:

```bash
./setup_folds.py \
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

Note: we recommend loading the model trunk only. While it is possible to load full Borzoi model and ignore last dense layer by model.load_weights(weight_file, skip_mismatch=True, by_name=True), Tensorflow requires loading layer weight by name in this way. If layer name don't match, weights of the layer will not be loaded and no warning message will be given.

### Step 7. Load models

We apply weight merging for lora, ia3, and locon weights, and so there is no architecture changes once the model is trained. You can use the same params.json file, and load the train_rep0/model_best.mergeW.h5 weight file.

For houlsby and houlsby_se, model architectures change due to the insertion of adapter modules. New architecture json file is auto-generated in train_rep0/params.json;
