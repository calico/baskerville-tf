import h5py
import numpy as np


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


def collect_h5_borzoi(out_dir, num_procs, sad_stat) -> None:
    h5_file = "scores_f0c0.h5"

    # count sequences
    num_seqs = 0
    for pi in range(num_procs):
        # open job
        job_h5_file = "%s/job%d/%s" % (out_dir, pi, h5_file)
        job_h5_open = h5py.File(job_h5_file, "r")
        num_seqs += job_h5_open[sad_stat].shape[0]
        job_h5_open.close()

    # initialize final h5
    final_h5_file = "%s/%s" % (out_dir, h5_file)
    final_h5_open = h5py.File(final_h5_file, "w")

    # keep dict for string values
    final_strings = {}

    job0_h5_file = "%s/job0/%s" % (out_dir, h5_file)
    job0_h5_open = h5py.File(job0_h5_file, "r")
    for key in job0_h5_open.keys():
        key_shape = list(job0_h5_open[key].shape)
        key_shape[0] = num_seqs
        key_shape = tuple(key_shape)
        if job0_h5_open[key].dtype.char == "S":
            final_strings[key] = []
        else:
            final_h5_open.create_dataset(
                key, shape=key_shape, dtype=job0_h5_open[key].dtype
            )

    # set values
    si = 0
    for pi in range(num_procs):
        # open job
        job_h5_file = "%s/job%d/%s" % (out_dir, pi, h5_file)
        job_h5_open = h5py.File(job_h5_file, "r")

        # append to final
        for key in job_h5_open.keys():
            job_seqs = job_h5_open[key].shape[0]
            if job_h5_open[key].dtype.char == "S":
                final_strings[key] += list(job_h5_open[key])
            else:
                final_h5_open[key][si : si + job_seqs] = job_h5_open[key]

        job_h5_open.close()
        si += job_seqs

    # create final string datasets
    for key in final_strings:
        final_h5_open.create_dataset(key, data=np.array(final_strings[key], dtype="S"))

    final_h5_open.close()
