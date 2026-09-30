import os
import pickle
import sys
import subprocess
import time


def exec_par(cmds, max_proc=None, verbose=False):
    """
    Execute the commands in the list 'cmds' in parallel, but
    only running 'max_proc' at a time.
    Args:
        cmds: list of commands to execute
        max_proc: maximum number of processes to run in parallel
        verbose: print command to stderr
    """
    total = len(cmds)
    finished = 0
    running = 0
    p = []

    if max_proc is None:
        max_proc = len(cmds)

    if max_proc == 1:
        while finished < total:
            if verbose:
                print(cmds[finished], file=sys.stderr)
            op = subprocess.Popen(cmds[finished], shell=True)
            os.waitpid(op.pid, 0)
            finished += 1

    else:
        while finished + running < total:
            # launch jobs up to max
            while running < max_proc and finished + running < total:
                if verbose:
                    print(cmds[finished + running], file=sys.stderr)
                p.append(subprocess.Popen(cmds[finished + running], shell=True))
                # print 'Running %d' % p[running].pid
                running += 1

            # are any jobs finished
            new_p = []
            for i in range(len(p)):
                if p[i].poll() is not None:
                    running -= 1
                    finished += 1
                else:
                    new_p.append(p[i])

            # if none finished, sleep
            if len(new_p) == len(p):
                time.sleep(1)
            p = new_p

        # wait for all to finish
        for i in range(len(p)):
            p[i].wait()


def load_extra_options(options_pkl_file, options):
    """
    Args:
        options_pkl_file: option file
        options: existing options from command line
    Returns:
        options: updated options
    """
    options_pkl = open(options_pkl_file, "rb")
    new_options = pickle.load(options_pkl)
    new_option_attrs = vars(new_options)
    # Assuming 'options' is the existing options object
    # Update the existing options with the new attributes
    for attr_name, attr_value in new_option_attrs.items():
        setattr(options, attr_name, attr_value)
    options_pkl.close()
    return options
