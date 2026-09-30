import h5py
import subprocess

import numpy as np

stat_keys = ["logSUM", "logD2"]
fasta_file = "tests/data/hg38_1m.fa.gz"
targets_file = "tests/data/tiny/hg38/targets.txt"
params_file = "tests/data/eval/params.json"
model_file = "tests/data/eval/model.h5"
snp_out_dir = "tests/data/ism/snp_out"
bed_out_dir = "tests/data/ism/bed_out"


def test_snp():
    cmd = [
        "src/baskerville/scripts/hound_ism_snp.py",
        "-f",
        fasta_file,
        "-l",
        "6",
        "-o",
        snp_out_dir,
        "--stats",
        ",".join(stat_keys),
        "-t",
        targets_file,
        params_file,
        model_file,
        "tests/data/ism/snp.vcf",
    ]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

    with h5py.File(f"{snp_out_dir}/scores.h5", "r") as scores_h5:
        for sk in stat_keys:
            score = scores_h5[sk][:]
            score_var = score.var(axis=2, dtype="float32")

            # verify shape
            assert score.shape == (2, 6, 4, 47)

            # verify not NaN
            assert not np.isnan(score).any()

            # verify variance
            assert (score_var > 0).all()


def test_bed():
    cmd = [
        "src/baskerville/scripts/hound_ism_bed.py",
        "-f",
        fasta_file,
        "-l",
        "6",
        "-o",
        bed_out_dir,
        "--stats",
        ",".join(stat_keys),
        "-t",
        targets_file,
        params_file,
        model_file,
        "tests/data/ism/seqs.bed",
    ]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

    with h5py.File(f"{bed_out_dir}/scores.h5", "r") as scores_h5:
        for sk in stat_keys:
            score = scores_h5[sk][:]
            score_var = score.var(axis=2, dtype="float32")

            # verify shape
            assert score.shape == (2, 6, 4, 47)

            # verify not NaN
            assert not np.isnan(score).any()

            # verify variance
            assert (score_var > 0).all()
