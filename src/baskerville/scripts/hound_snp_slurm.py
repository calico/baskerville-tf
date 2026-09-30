#!/usr/bin/env python
# Copyright 2017 Calico LLC

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
from optparse import OptionParser
import os
import pickle
import subprocess
import sys

import h5py
import numpy as np

import slurm

"""
hound_snp_slurm.py

Compute variant effect predictions for SNPs in a VCF file,
parallelized across a slurm cluster.
"""


################################################################################
# main
################################################################################
def main():
    usage = "usage: %prog [options] <params_file> <model_file> <vcf_file>"
    parser = OptionParser(usage)

    # snp
    parser.add_option(
        "-c",
        dest="cluster_snps_pct",
        default=0,
        type="float",
        help="Cluster SNPs within a %% of the seq length to make a single ref pred [Default: %default]",
    )
    parser.add_option(
        "-f",
        dest="genome_fasta",
        default=None,
        help="Genome FASTA for sequences [Default: %default]",
    )
    parser.add_option(
        "--float16",
        dest="float16",
        default=False,
        action="store_true",
        help="Use mixed float16 precision [Default: %default]",
    )
    parser.add_option(
        "-o",
        dest="out_dir",
        default="snp_out",
        help="Output directory for tables and plots [Default: %default]",
    )
    parser.add_option(
        "--rc",
        dest="rc",
        default=False,
        action="store_true",
        help="Average forward and reverse complement predictions [Default: %default]",
    )
    parser.add_option(
        "--shifts",
        dest="shifts",
        default="0",
        type="str",
        help="Ensemble prediction shifts [Default: %default]",
    )
    parser.add_option(
        "--stats",
        dest="snp_stats",
        default="logSAD",
        help="Comma-separated list of stats to save. [Default: %default]",
    )
    parser.add_option(
        "-t",
        dest="targets_file",
        default=None,
        type="str",
        help="File specifying target indexes and labels in table format",
    )
    parser.add_option(
        "-u",
        dest="untransform_old",
        default=False,
        action="store_true",
        help="Untransform old models [Default: %default]",
    )

    # multi
    parser.add_option(
        "-e",
        dest="conda_env",
        default="tf12",
        help="Anaconda environment [Default: %default]",
    )
    parser.add_option(
        "--name",
        dest="name",
        default="snp",
        help="SLURM name prefix [Default: %default]",
    )
    parser.add_option(
        "--max_proc",
        dest="max_proc",
        default=None,
        type="int",
        help="Maximum concurrent processes [Default: %default]",
    )
    parser.add_option(
        "-p",
        dest="processes",
        default=None,
        type="int",
        help="Number of processes, passed by multi script",
    )
    parser.add_option(
        "-q",
        dest="queue",
        default="geforce",
        help="SLURM queue on which to run the jobs [Default: %default]",
    )
    parser.add_option(
        "-r",
        dest="restart",
        default=False,
        action="store_true",
        help="Restart a partially completed job [Default: %default]",
    )
    (options, args) = parser.parse_args()

    if len(args) != 3:
        parser.error("Must provide parameters and model files and VCF file")

    #######################################################
    # prep work

    # output directory
    if not options.restart:
        if os.path.isdir(options.out_dir):
            print("Please remove %s" % options.out_dir, file=sys.stderr)
            exit(1)
        os.mkdir(options.out_dir)

    # pickle options
    options_pkl_file = "%s/options.pkl" % options.out_dir
    options_pkl = open(options_pkl_file, "wb")
    pickle.dump(options, options_pkl)
    options_pkl.close()

    #######################################################
    # launch worker threads
    jobs = []
    for pi in range(options.processes):
        if not options.restart or not job_completed(options, pi):
            cmd = ". /home/drk/anaconda3/etc/profile.d/conda.sh;"
            cmd += " conda activate %s;" % options.conda_env

            cmd += " time hound_snp.py %s %s %d" % (
                options_pkl_file,
                " ".join(args),
                pi,
            )

            name = "%s_p%d" % (options.name, pi)
            outf = "%s/job%d.out" % (options.out_dir, pi)
            errf = "%s/job%d.err" % (options.out_dir, pi)

            j = slurm.Job(
                cmd,
                name,
                outf,
                errf,
                queue=options.queue,
                gpu=1,
                mem=22000,
                time="14-0:0:0",
            )
            jobs.append(j)

    slurm.multi_run(
        jobs, max_proc=options.max_proc, verbose=True, launch_sleep=10, update_sleep=60
    )

    #######################################################
    # collect output

    collect_h5("scores.h5", options.out_dir, options.processes)

    # for pi in range(options.processes):
    #     shutil.rmtree('%s/job%d' % (options.out_dir,pi))


def collect_table(file_name, out_dir, num_procs):
    os.rename("%s/job0/%s" % (out_dir, file_name), "%s/%s" % (out_dir, file_name))
    for pi in range(1, num_procs):
        subprocess.call(
            "tail -n +2 %s/job%d/%s >> %s/%s"
            % (out_dir, pi, file_name, out_dir, file_name),
            shell=True,
        )


def collect_h5(file_name, out_dir, num_procs):
    # count variants
    num_variants = 0
    for pi in range(num_procs):
        # open job
        job_h5_file = "%s/job%d/%s" % (out_dir, pi, file_name)
        job_h5_open = h5py.File(job_h5_file, "r")
        num_variants += len(job_h5_open["snp"])
        job_h5_open.close()

    # initialize final h5
    final_h5_file = "%s/%s" % (out_dir, file_name)
    final_h5_open = h5py.File(final_h5_file, "w")

    # keep dict for string values
    final_strings = {}

    job0_h5_file = "%s/job0/%s" % (out_dir, file_name)
    job0_h5_open = h5py.File(job0_h5_file, "r")
    for key in job0_h5_open.keys():
        if key in ["percentiles", "target_ids", "target_labels"]:
            # copy
            final_h5_open.create_dataset(key, data=job0_h5_open[key])

        elif key[-4:] == "_pct":
            values = np.zeros(job0_h5_open[key].shape)
            final_h5_open.create_dataset(key, data=values)

        elif job0_h5_open[key].dtype.char == "S":
            final_strings[key] = []

        elif job0_h5_open[key].ndim == 1:
            final_h5_open.create_dataset(
                key, shape=(num_variants,), dtype=job0_h5_open[key].dtype
            )

        else:
            num_targets = job0_h5_open[key].shape[1]
            final_h5_open.create_dataset(
                key, shape=(num_variants, num_targets), dtype=job0_h5_open[key].dtype
            )

    job0_h5_open.close()

    # set values
    vi = 0
    for pi in range(num_procs):
        # open job
        job_h5_file = "%s/job%d/%s" % (out_dir, pi, file_name)
        job_h5_open = h5py.File(job_h5_file, "r")

        # append to final
        for key in job_h5_open.keys():
            if key in ["percentiles", "target_ids", "target_labels"]:
                # once is enough
                pass

            elif key[-4:] == "_pct":
                # average
                u_k1 = np.array(final_h5_open[key])
                x_k = np.array(job_h5_open[key])
                final_h5_open[key][:] = u_k1 + (x_k - u_k1) / (pi + 1)

            else:
                if job_h5_open[key].dtype.char == "S":
                    final_strings[key] += list(job_h5_open[key])
                else:
                    job_variants = job_h5_open[key].shape[0]
                    try:
                        final_h5_open[key][vi : vi + job_variants] = job_h5_open[key]
                    except TypeError as e:
                        print(e)
                        print(
                            f"{job_h5_file} ${key} has the wrong shape. Remove this file and rerun"
                        )
                        exit()

        vi += job_variants
        job_h5_open.close()

    # create final string datasets
    for key in final_strings:
        final_h5_open.create_dataset(key, data=np.array(final_strings[key], dtype="S"))

    final_h5_open.close()


def job_completed(options, pi):
    """Check whether a specific job has generated its
    output file."""
    out_file = "%s/job%d/scores.h5" % (options.out_dir, pi)
    return os.path.isfile(out_file) or os.path.isdir(out_file)


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
