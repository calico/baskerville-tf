#!/usr/bin/env python
# Copyright 2019 Calico LLC

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
import glob
import json
import os
import shutil

from natsort import natsorted

"""
setup_folds.py

Setup directory structure for baskerville model training on cross folds.
"""

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <params_file> <data1_dir> ...'
  parser = OptionParser(usage)

  # setup options
  parser.add_option('-c', dest='crosses',
      default=1, type='int',
      help='Number of cross-fold rounds [Default:%default]')
  parser.add_option('-f', dest='fold_subset',
      default=None, type='int',
      help='Run a subset of folds [Default:%default]')
  parser.add_option('-o', dest='out_dir',
      default='train_out',
      help='Output directory [Default: %default]')

  (options, args) = parser.parse_args()

  if len(args) < 2:
    parser.error('Must provide parameters and data directory.')
  else:
    params_file = os.path.abspath(args[0])
    data_dirs = [os.path.abspath(arg) for arg in args[1:]]

  #######################################################
  # prep work
  
  if os.path.isdir(options.out_dir):
    print('Output directory %s exists. Please remove.' % options.out_dir)
    exit(1)
  os.makedirs(options.out_dir, exist_ok=True)

  # read model parameters
  with open(params_file) as params_open:
    params = json.load(params_open)

  # copy params into output directory
  shutil.copy(params_file, '%s/params.json' % options.out_dir)

  # read data parameters
  num_data = len(data_dirs)
  data_stats_file = '%s/statistics.json' % data_dirs[0]
  with open(data_stats_file) as data_stats_open:
    data_stats = json.load(data_stats_open)

  # count folds
  num_folds = len([dkey for dkey in data_stats if dkey.startswith('fold')])

  # subset folds
  if options.fold_subset is not None:
    num_folds = min(options.fold_subset, num_folds)

  # arrange data
  for ci in range(options.crosses):
    for fi in range(num_folds):
      rep_dir = '%s/f%dc%d' % (options.out_dir, fi, ci)
      os.makedirs(rep_dir, exist_ok=True)

      # make data directories
      for di in range(num_data):
        rep_data_dir = '%s/data%d' % (rep_dir, di)
        if not os.path.isdir(rep_data_dir):
          make_rep_data(data_dirs[di], rep_data_dir, fi, ci)

  print('Setup complete! Directory structure created at: %s' % options.out_dir)


def make_rep_data(data_dir, rep_data_dir, fi, ci): 
  # read data parameters
  data_stats_file = '%s/statistics.json' % data_dir
  with open(data_stats_file) as data_stats_open:
    data_stats = json.load(data_stats_open)

  # sequences per fold
  fold_seqs = []
  dfi = 0
  while 'fold%d_seqs'%dfi in data_stats:
    fold_seqs.append(data_stats['fold%d_seqs'%dfi])
    del data_stats['fold%d_seqs'%dfi]
    dfi += 1
  num_folds = dfi

  # split folds into train/valid/test
  test_fold = fi
  valid_fold = (fi+1+ci) % num_folds
  train_folds = [fold for fold in range(num_folds) if fold not in [valid_fold,test_fold]]

  # clear existing directory
  if os.path.isdir(rep_data_dir):
    shutil.rmtree(rep_data_dir)

  # make data directory
  os.makedirs(rep_data_dir, exist_ok=True)

  # dump data stats
  data_stats['test_seqs'] = fold_seqs[test_fold]
  data_stats['valid_seqs'] = fold_seqs[valid_fold]
  data_stats['train_seqs'] = sum([fold_seqs[tf] for tf in train_folds])
  with open('%s/statistics.json'%rep_data_dir, 'w') as data_stats_open:
    json.dump(data_stats, data_stats_open, indent=4)

  # set sequence tvt
  try:
    seqs_bed_out = open('%s/sequences.bed'%rep_data_dir, 'w')
    for line in open('%s/sequences.bed'%data_dir):
      a = line.split()
      sfi = int(a[-1].replace('fold',''))
      if sfi == test_fold:
        a[-1] = 'test'
      elif sfi == valid_fold:
        a[-1] = 'valid'
      else:
        a[-1] = 'train'
      print('\t'.join(a), file=seqs_bed_out)
    seqs_bed_out.close()
  except (ValueError, FileNotFoundError):
    pass

  # copy targets
  shutil.copy('%s/targets.txt'%data_dir, '%s/targets.txt'%rep_data_dir)

  # sym link tfrecords
  rep_tfr_dir = '%s/tfrecords' % rep_data_dir
  os.mkdir(rep_tfr_dir)

  # test tfrecords
  ti = 0
  test_tfrs = natsorted(glob.glob('%s/tfrecords/fold%d-*.tfr' % (data_dir, test_fold)))
  for test_tfr in test_tfrs:
    test_tfr = os.path.abspath(test_tfr)
    test_rep_tfr = '%s/test-%d.tfr' % (rep_tfr_dir, ti)
    os.symlink(test_tfr, test_rep_tfr)
    ti += 1

  # valid tfrecords
  ti = 0
  valid_tfrs = natsorted(glob.glob('%s/tfrecords/fold%d-*.tfr' % (data_dir, valid_fold)))
  for valid_tfr in valid_tfrs:
    valid_tfr = os.path.abspath(valid_tfr)
    valid_rep_tfr = '%s/valid-%d.tfr' % (rep_tfr_dir, ti)
    os.symlink(valid_tfr, valid_rep_tfr)
    ti += 1

  # train tfrecords
  ti = 0
  train_tfrs = []
  for tfi in train_folds:
    train_tfrs += natsorted(glob.glob('%s/tfrecords/fold%d-*.tfr' % (data_dir, tfi)))
  for train_tfr in train_tfrs:
    train_tfr = os.path.abspath(train_tfr)
    train_rep_tfr = '%s/train-%d.tfr' % (rep_tfr_dir, ti)
    os.symlink(train_tfr, train_rep_tfr)
    ti += 1


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
