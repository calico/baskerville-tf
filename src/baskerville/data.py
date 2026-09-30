# Copyright 2023 Calico LLC

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
import collections
import heapq
import math
import subprocess
import sys
import tempfile

import numpy as np
import pysam

"""
data.py

Helper methods for hound_data*
"""


def annotate_unmap(mseqs, unmap_bed, seq_length, pool_width):
    """Intersect the sequence segments with unmappable regions
         and annoate the segments as NaN to possible be ignored.

    Args:
      mseqs: list of ModelSeq's
      unmap_bed: unmappable regions BED file
      seq_length: sequence length (after cropping)
      pool_width: pooled bin width

    Returns:
      seqs_unmap: NxL binary NA indicators
    """

    # print sequence segments to file
    seqs_temp = tempfile.NamedTemporaryFile()
    seqs_bed_file = seqs_temp.name
    write_seqs_bed(seqs_bed_file, mseqs)

    # hash segments to indexes
    chr_start_indexes = {}
    for i in range(len(mseqs)):
        chr_start_indexes[(mseqs[i].chr, mseqs[i].start)] = i

    # initialize unmappable array
    pool_seq_length = seq_length // pool_width
    seqs_unmap = np.zeros((len(mseqs), pool_seq_length), dtype="bool")

    # intersect with unmappable regions
    p = subprocess.Popen(
        "bedtools intersect -wo -a %s -b %s" % (seqs_bed_file, unmap_bed),
        shell=True,
        stdout=subprocess.PIPE,
    )
    for line in p.stdout:
        line = line.decode("utf-8")
        a = line.split()

        seq_chrom = a[0]
        seq_start = int(a[1])
        seq_end = int(a[2])
        seq_key = (seq_chrom, seq_start)

        unmap_start = int(a[4])
        unmap_end = int(a[5])

        overlap_start = max(seq_start, unmap_start)
        overlap_end = min(seq_end, unmap_end)

        pool_seq_unmap_start = math.floor((overlap_start - seq_start) / pool_width)
        pool_seq_unmap_end = math.ceil((overlap_end - seq_start) / pool_width)

        # skip minor overlaps to the first
        first_start = seq_start + pool_seq_unmap_start * pool_width
        first_end = first_start + pool_width
        first_overlap = first_end - overlap_start
        if first_overlap < 0.1 * pool_width:
            pool_seq_unmap_start += 1

        # skip minor overlaps to the last
        last_start = seq_start + (pool_seq_unmap_end - 1) * pool_width
        last_overlap = overlap_end - last_start
        if last_overlap < 0.1 * pool_width:
            pool_seq_unmap_end -= 1

        seqs_unmap[
            chr_start_indexes[seq_key], pool_seq_unmap_start:pool_seq_unmap_end
        ] = True
        assert (
            seqs_unmap[
                chr_start_indexes[seq_key], pool_seq_unmap_start:pool_seq_unmap_end
            ].sum()
            == pool_seq_unmap_end - pool_seq_unmap_start
        )

    return seqs_unmap


################################################################################
def break_large_contigs(contigs, break_t, verbose=False):
    """Break large contigs in half until all contigs are under
    the size threshold."""

    # initialize a heapq of contigs and lengths
    contig_heapq = []
    for ctg in contigs:
        ctg_len = ctg.end - ctg.start
        heapq.heappush(contig_heapq, (-ctg_len, ctg))

    ctg_len = break_t + 1
    while ctg_len > break_t:
        # pop largest contig
        ctg_nlen, ctg = heapq.heappop(contig_heapq)
        ctg_len = -ctg_nlen

        # if too large
        if ctg_len > break_t:
            if verbose:
                print(
                    "Breaking %s:%d-%d (%d nt)" % (ctg.chr, ctg.start, ctg.end, ctg_len)
                )

            # break in two
            ctg_mid = ctg.start + ctg_len // 2
            ctg_left = Contig(ctg.genome, ctg.chr, ctg.start, ctg_mid)
            ctg_right = Contig(ctg.genome, ctg.chr, ctg_mid, ctg.end)

            # add left
            ctg_left_len = ctg_left.end - ctg_left.start
            heapq.heappush(contig_heapq, (-ctg_left_len, ctg_left))

            # add right
            ctg_right_len = ctg_right.end - ctg_right.start
            heapq.heappush(contig_heapq, (-ctg_right_len, ctg_right))

    # return to list
    contigs = [len_ctg[1] for len_ctg in contig_heapq]

    return contigs


def contig_sequences(contigs, seq_length, stride, snap=1, label=None):
    """Break up a list of Contig's into a list of model length
    and stride sequence contigs."""
    mseqs = []

    for ctg in contigs:
        seq_start = int(np.ceil(ctg.start / snap) * snap)
        seq_end = seq_start + seq_length

        while seq_end < ctg.end:
            # record sequence
            mseqs.append(ModelSeq(ctg.genome, ctg.chr, seq_start, seq_end, label))

            # update
            seq_start += stride
            seq_end += stride

    return mseqs


def load_chromosomes(genome_file):
    """Load genome segments from either a FASTA file or
    chromosome length table."""

    # is genome_file FASTA or (chrom,start,end) table?
    file_fasta = open(genome_file).readline()[0] == ">"

    chrom_segments = {}

    if file_fasta:
        fasta_open = pysam.Fastafile(genome_file)
        for i in range(len(fasta_open.references)):
            chrom_segments[fasta_open.references[i]] = [(0, fasta_open.lengths[i])]
        fasta_open.close()

    else:
        for line in open(genome_file):
            a = line.split()
            chrom_segments[a[0]] = [(0, int(a[1]))]

    return chrom_segments


def rejoin_large_contigs(contigs):
    """Rejoin large contigs that were broken up before alignment comparison."""

    # split list by genome/chromosome
    gchr_contigs = {}
    for ctg in contigs:
        gchr = (ctg.genome, ctg.chr)
        gchr_contigs.setdefault(gchr, []).append(ctg)

    contigs = []
    for gchr in gchr_contigs:
        # sort within chromosome
        gchr_contigs[gchr].sort(key=lambda x: x.start)
        # gchr_contigs[gchr] = sorted(gchr_contigs[gchr], key=lambda ctg: ctg.start)

        ctg_ongoing = gchr_contigs[gchr][0]
        for i in range(1, len(gchr_contigs[gchr])):
            ctg_this = gchr_contigs[gchr][i]
            if ctg_ongoing.end == ctg_this.start:
                # join
                # ctg_ongoing.end = ctg_this.end
                ctg_ongoing = ctg_ongoing._replace(end=ctg_this.end)
            else:
                # conclude ongoing
                contigs.append(ctg_ongoing)

                # move to next
                ctg_ongoing = ctg_this

        # conclude final
        contigs.append(ctg_ongoing)

    return contigs


def split_contigs(chrom_segments, gaps_file):
    """Split the assembly up into contigs defined by the gaps.

    Args:
      chrom_segments: dict mapping chromosome names to lists of (start,end)
      gaps_file: file specifying assembly gaps

    Returns:
      chrom_segments: same, with segments broken by the assembly gaps.
    """

    chrom_events = {}

    # add known segments
    for chrom in chrom_segments:
        if len(chrom_segments[chrom]) > 1:
            print(
                "I've made a terrible mistake...regarding the length of chrom_segments[%s]"
                % chrom,
                file=sys.stderr,
            )
            exit(1)
        cstart, cend = chrom_segments[chrom][0]
        chrom_events.setdefault(chrom, []).append((cstart, "Cstart"))
        chrom_events[chrom].append((cend, "cend"))

    # add gaps
    for line in open(gaps_file):
        a = line.split()
        chrom = a[0]
        gstart = int(a[1])
        gend = int(a[2])

        # consider only if its in our genome
        if chrom in chrom_events:
            chrom_events[chrom].append((gstart, "gstart"))
            chrom_events[chrom].append((gend, "Gend"))

    for chrom in chrom_events:
        # sort
        chrom_events[chrom].sort()

        # read out segments
        chrom_segments[chrom] = []
        for i in range(len(chrom_events[chrom]) - 1):
            pos1, event1 = chrom_events[chrom][i]
            pos2, event2 = chrom_events[chrom][i + 1]

            event1 = event1.lower()
            event2 = event2.lower()

            shipit = False
            if event1 == "cstart" and event2 == "cend":
                shipit = True
            elif event1 == "cstart" and event2 == "gstart":
                shipit = True
            elif event1 == "gend" and event2 == "gstart":
                shipit = True
            elif event1 == "gend" and event2 == "cend":
                shipit = True
            elif event1 == "gstart" and event2 == "gend":
                pass
            else:
                print(
                    "I'm confused by this event ordering: %s - %s" % (event1, event2),
                    file=sys.stderr,
                )
                exit(1)

            if shipit and pos1 < pos2:
                chrom_segments[chrom].append((pos1, pos2))

    return chrom_segments


def write_seqs_bed(bed_file, seqs, labels=False):
    """Write sequences to BED file."""
    bed_out = open(bed_file, "w")
    for i in range(len(seqs)):
        line = "%s\t%d\t%d" % (seqs[i].chr, seqs[i].start, seqs[i].end)
        if labels:
            line += "\t%s" % seqs[i].label
        print(line, file=bed_out)
    bed_out.close()


################################################################################
Contig = collections.namedtuple("Contig", ["genome", "chr", "start", "end"])
ModelSeq = collections.namedtuple(
    "ModelSeq", ["genome", "chr", "start", "end", "label"]
)
