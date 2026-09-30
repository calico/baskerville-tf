# Baskerville Hound Scripts Documentation

This document describes the main functionality and differences between key scripts in the Baskerville suite.

## Overview

The Baskerville package includes several "hound" scripts for making genomic predictions using trained deep learning models. Each script serves a specific purpose in the genomic prediction pipeline, from basic sequence prediction to variant effect analysis and mutagenesis studies.

| Script | Input Type | Analysis Focus | Output Type | Parallelization |
|--------|------------|----------------|-------------|-----------------|
| `hound_predbed.py` | BED regions | Sequence prediction | Per-region predictions | Single process |
| `hound_predvcf.py` | VCF variants | Full window prediction | Ref/alt predictions | Single process |
| `hound_snp_slurm.py` | VCF variants | Variant effects | Effect statistics | SLURM cluster |
| `hound_snp.py` | VCF variants | Variant effects | Effect statistics | Single/multi-worker |
| `hound_snpgene.py` | VCF + GTF | Gene-context effects | Gene-level statistics | Single/multi-worker |
| `hound_ism_bed.py` | BED regions | Saturation mutagenesis | Position × nucleotide effects | Single process |
| `hound_ism_snp.py` | VCF variants | Saturation mutagenesis | Position × nucleotide effects | Single process |
| `hound_isd_bed.py` | BED regions + GTF | Deletion mutagenesis | Gene-context deletion effects | Single process |


## Script Descriptions

### 1. `hound_predbed.py` - BED File Sequence Prediction

**Purpose**: Predicts sequences from genomic regions defined in a BED file.

**Key Features**:
- Takes genomic coordinates from a BED file as input
- Generates model predictions for specified genomic regions
- Supports both full predictions and summed predictions across sequence length
- Can generate BigWig files for genome browser visualization
- Supports embedding layer extraction for sequence representations

**Input**:
- BED file with genomic coordinates
- Model parameters and weights
- Genome FASTA file
- Optional targets file for specific prediction tracks

**Output**:
- HDF5 file with predictions for each BED region
- Optional BigWig files for visualization
- Predictions can be per-position or summed across regions

---

### 2. `hound_predvcf.py` - VCF Variant Window Prediction

**Purpose**: Predicts full genomic windows for variants from a VCF file, generating predictions for both reference and alternate alleles.

**Key Features**:
- Processes SNPs from VCF files (INDELs use reference sequence for alternate allele)
- Generates full-window predictions at 32bp-bin resolution
- Provides separate predictions for reference and alternate alleles
- Focuses on complete genomic context around variants

**Input**:
- VCF file with genetic variants
- Model parameters and weights
- Genome FASTA file
- Targets file (required for stranded datasets)

**Output**:
- HDF5 file with separate reference and alternate predictions
- Full prediction windows for each variant
- Variant metadata (chromosome, position, ref/alt alleles)

---

### 3. `hound_snp.py` - SNP Variant Effect Prediction

**Purpose**: Computes variant effect predictions for SNPs in a VCF file using various statistical measures.

**Key Features**:
- Calculates multiple variant effect statistics (logSUM, SAD, D1, D2, JS, etc.)
- metric summarized along the sequence length axis
- Supports SNP clustering to reduce redundant predictions
- Handles both single-worker and multi-worker execution modes
- Comprehensive suite of distance and information-theory based metrics

**Available Statistics**:
- **Basic Sum-based**: SUM, logSUM, logSED, sqrtSUM
- **Maximum Effect**: SAX (finds position of maximum difference)
- **Distance-based**: D1/D2 (L1/L2 distances), logD1/logD2, sqrtD1/sqrtD2
- **Information Theory**: JS (Jensen-Shannon divergence), logJS

**Input**:
- VCF file with SNPs
- Model parameters and weights
- Genome FASTA file
- Targets file

**Output**:
- HDF5 file with variant effect scores
- Statistical summaries for each SNP across all prediction targets
- Configurable output statistics

---

### 4. `hound_snp_slurm.py` - Cluster-Parallelized SNP Analysis

**Purpose**: Orchestrates parallel computation of variant effect predictions across a SLURM cluster.

**Key Features**:
- Distributes SNP analysis across multiple compute nodes
- Manages job submission and monitoring via SLURM
- Handles result aggregation from parallel workers
- Optimized for large-scale variant effect studies

**Input**:
- Same inputs as `hound_snp.py`
- Additional SLURM configuration parameters
- Number of parallel processes to spawn

**Output**:
- Aggregated results from all parallel workers
- Job management and monitoring files
- Same final output format as `hound_snp.py`

---

### 5. `hound_snpgene.py` - Gene-Context SNP Analysis

**Purpose**: Computes variant effect predictions for SNPs with respect to gene exons defined in a GTF file.

**Key Features**:
- Focuses variant analysis on gene regions (exons)
- metric computed for each variant-gene pair
- Integrates GTF gene annotations
- Supports gene span aggregation
- Gene clustering to optimize predictions
- Same statistical framework as `hound_snp.py`

**Input**:
- VCF file with SNPs
- GTF file with gene annotations
- Model parameters and weights
- Genome FASTA file
- Targets file

---

### 6. `hound_ism_bed.py` - In Silico Saturation Mutagenesis (BED)

**Purpose**: Performs systematic in silico saturation mutagenesis of sequences from BED file regions.

**Key Features**:
- Mutates every position in a specified region to all 4 possible nucleotides (A, C, G, T)
- Configurable mutation window around sequence center
- Measures regulatory effects of all possible single nucleotide changes
- Supports ensemble predictions with multiple shifts

**Input**:
- BED file with genomic regions to mutate
- Model parameters and weights
- Genome FASTA file
- Targets file for prediction tracks

**Output**:
- HDF5 file with mutation effect scores
- 4D array: [sequences × positions × nucleotides × targets]
- Identifies critical regulatory positions and nucleotides

---

### 7. `hound_ism_snp.py` - In Silico Saturation Mutagenesis (VCF)

**Purpose**: Performs systematic in silico saturation mutagenesis around variants from VCF files.

**Key Features**:
- Centers mutagenesis around known variants from VCF
- Systematically tests all nucleotide changes in variant neighborhoods
- Default 200bp mutation window around variants
- Same mutagenesis approach as `hound_ism_bed.py` but variant-focused

**Input**:
- VCF file with variants of interest
- Model parameters and weights
- Genome FASTA file
- Targets file for prediction tracks

**Output**:
- HDF5 file with mutation effect scores
- 4D array: [variants × positions × nucleotides × targets]
- Variant metadata and sequence labels

---

### 8. `hound_isd_bed.py` - In Silico Deletion Mutagenesis (Gene-Context)

**Purpose**: Performs systematic in silico deletion mutagenesis of sequences with gene annotation context.

**Key Features**:
- Systematically deletes nucleotides (default 1bp) across BED regions
- Integrates gene annotations to focus on gene-relevant regions
- Supports both exon-only and gene span analysis
- Uses stitching to handle deletion compensation shifts
- Gene-aware analysis with strand-specific effects

**Input**:
- BED file with genomic regions
- GTF file with gene annotations
- Model parameters and weights
- Genome FASTA file
- Targets file and optional target genes list

**Output**:
- Separate HDF5 files for each BED entry
- Deletion effect scores focused on gene regions
- SED/logSED statistics for measuring regulatory impact

## Choosing the Right Script

- **Use `hound_predbed.py`** when you want predictions for specific genomic regions
- **Use `hound_predvcf.py`** when you need full prediction windows around variants
- **Use `hound_snp.py`** for variant effect analysis with statistical summaries
- **Use `hound_snp_slurm.py`** for large-scale variant studies requiring cluster computing
- **Use `hound_snpgene.py`** when focusing on variant effects within gene contexts
- **Use `hound_ism_bed.py`** for comprehensive mutagenesis analysis of regulatory regions
- **Use `hound_ism_snp.py`** for detailed mutagenesis analysis around specific variants
- **Use `hound_isd_bed.py`** for deletion-based mutagenesis with gene context
