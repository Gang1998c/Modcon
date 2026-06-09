# ModCon: An Integrated Multimodal Pipeline for RNA Modification Conservation Scoring

[![Python](https://img.shields.io/badge/Python-3.8%252B-blue.svg)](https://www.python.org/)
[![R](https://img.shields.io/badge/R-4.2.2%252B-seafoam.svg)](https://www.r-project.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.2-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-yellow.svg)](https://huggingface.co/)

**ModCon** is a high-performance, cross-language software tool designed to assess and score RNA modification site conservation across species (human, mouse, and pig). By combining advanced sequence tokenization with multimodal deep learning architecture, this tool provides a comprehensive and interpretable solution for quantifying conservation features of RNA modifications across different species.

------------------------------------------------------------------------

## 🧬 Architectural Overview

This pipeline executes an automated 5-stage data processing workflow that seamlessly bridges Python and R environments:

1.  **Base Feature Mapping:**
    Maps site coordinates with a base reference directory to inherit empirical statistical anchors (`noes` and `pm` metrics).

2.  **Cross-Species Coordinate Liftover:**
    Uses precise and strand-aware logic to project human `hg38` modification sites to target reference networks (`mm39` and `susScr11`).

3.  **High-Dimensional Topology Profiling:**
    Executes R language subprocess via `m6ALogisticModel` package, lifts coordinates to `hg19`, isolates 40 distinct genomic features, and extracts centered 800bp reference sequence mappings.

4.  **Base-Specific Multimodal Inference:**
    Employs automated sequence routing gate-forwarding data arrays to query specialized deep learning architectures (fusing MLP networks with DNABERT-2 Transformer embeddings).

5.  **Ensemble Aggregation:**
    Aggregates cross-species tracking results, empirical database scores, and deep learning classification probabilities into a single holistic metric: **ModCon Score**.

### Pipeline Workflow

![ModCon Pipeline Architecture](https://github.com/Gang1998c/Modcon/raw/main/Modcon_Framework.jpg)

This diagram shows the complete analysis workflow:
- **Data Collection**: RNA modification data from multiple species (human, mouse, pig)
- **Data Processing**: Base calling, ELIGOS modification detection, and site mapping
- **Model Construction**: Homologous modification consistency, intra-species reproducibility, sequence context conservation analysis
- **Fine-tuned Large Language Model**: MLP classifier and DNABERT-2 Transformer integration
- **Model Validation and Application**: NGS validation, variant density validation, functional analysis, and web service prediction

------------------------------------------------------------------------

## 🛠️ Prerequisites & Installation

### 1. Python Environment Setup

Ensure you activate an Anaconda environment running Python 3.8+:

```bash
# Create new conda environment
conda create -n modcon python=3.8 -y

# Activate environment
conda activate modcon

# Install dependencies
pip install -r requirements.txt
```

### 2. R Environment Setup

Execute the following commands in R to install required packages:

```r
# 1. Install standard CRAN dependency packages
install.packages(c("dplyr", "caret", "e1071", "ROCR", "pROC", "devtools"))

# 2. Install Bioconductor core bioinformatics packages and large genomic datasets
if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
BiocManager::install(c("GenomicRanges", "rtracklayer", "SummarizedExperiment", 
                       "TxDb.Hsapiens.UCSC.hg19.knownGene", 
                       "BSgenome.Hsapiens.UCSC.hg19", 
                       "BSgenome.Hsapiens.UCSC.hg38",
                       "fitCons.UCSC.hg19", 
                       "phastCons100way.UCSC.hg19"))

# 3. Install lab-specific multimodal feature extraction package
devtools::install_github("m6ALogisticModel")
```

### 3. Clone Repository

```bash
git clone https://github.com/Gang1998c/Modcon.git
cd Modcon
```

------------------------------------------------------------------------

## 📖 Quick Start

### Basic Usage

```bash
python master_pipeline.py \
    --user_df_path "server_example_data.csv" \
    --ontdata_path "data/ontdata.csv" \
    --r_script_path "server_R.R" \
    --hg38_fa "fasta_and_chain/hg38.fa" \
    --mm39_fa "mm39.fa" \
    --sus11_fa "susScr11.fa" \
    --mm39_chain "hg38ToMm39.over.chain" \
    --sus11_chain "hg38ToSusScr11.over.chain" \
    --mouse_raw_path "mouse_raw.csv" \
    --pig_raw_path "pig_raw.csv" \
    --dnabert_model_path "DNABERT-2-117M/" \
    --base_model_dir "model/" \
    --output_path "results.csv"
```

Users only need to provide single-base site data in DNA format (1-based). Refer to `server_example_data.csv` for format reference (reference genome: hg38). Other required files and trained models have been uploaded to Hugging Face and can be downloaded from there.

`ontdata.csv`, `mouse_raw.csv`, and `pig_raw.csv` are the original datasets used in this study and are also required for model execution. Due to the large file size, these datasets have been uploaded to Hugging Face and can be downloaded from: https://huggingface.co/datasets/Gang1998c/Modcon_data. The repository also includes the reference genomes and corresponding `over.chain` files.

The four trained base-specific models can be downloaded from: https://huggingface.co/Gang1998c/Modcon</sub>

For downloading and using the `DNABERT-2` model, please refer to:
• https://github.com/MAGICS-LAB/DNABERT_2
• https://huggingface.co/zhihan1996/DNABERT-2-117M

#📦 Data & Model Availability Download Guide:
·Essential Raw Datasets: The files ontdata.csv, mouse_raw.csv, and pig_raw.csv constitute the foundational raw data and are strictly required to run the model pipeline. Due to their large sizes, these files—along with the required reference genomes and over.chain files—have been hosted on Hugging Face. Users must download them manually from the Modcon Data Repository.
### Parameter Illustration

```
optional arguments:
  -h, --help            show this help message and exit
  --user_df_path USER_DF_PATH
                        Path to user-uploaded modification sites file (CSV format) (default: None)
  --r_script_path R_SCRIPT_PATH
                        Path to companion R feature extraction engine script (default: server_R.R)
  --ontdata_path ONTDATA_PATH
                        Path to base reference library (ontdata.csv) (default: None)
  --output_path OUTPUT_PATH
                        Output file destination path (default: master_pipeline_output.csv)
  --mouse_raw_path MOUSE_RAW_PATH
                        Path to mouse known conserved coordinates catalog (default: mouse_raw.csv)
  --pig_raw_path PIG_RAW_PATH
                        Path to pig known conserved coordinates catalog (default: pig_raw.csv)
  --hg38_fa HG38_FA     Path to Human hg38 reference genome FASTA (default: None)
  --mm39_fa MM39_FA     Path to Mouse mm39 reference genome FASTA (default: None)
  --sus11_fa SUS11_FA   Path to Pig susScr11 reference genome FASTA (default: None)
  --mm39_chain MM39_CHAIN
                        Path to hg38ToMm39 conversion over.chain file (default: None)
  --sus11_chain SUS11_CHAIN
                        Path to hg38ToSusScr11 conversion over.chain file (default: None)
  --dnabert_model_path DNABERT_MODEL_PATH
                        Path to local pre-trained DNABERT-2 workspace directory (default: ./embedded_model)
  --base_model_dir BASE_MODEL_DIR
                        Parent workspace directory containing model_A/C/G/U folders (default: ./)
```

------------------------------------------------------------------------

## 📊 Input Data Format

### Required Columns

| Column Name | Type | Description |
|-------------|------|-------------|
| `seqnames` | str | Chromosome identifier (e.g., `chr1`) |
| `position` | int | Modification site coordinate (hg38, 1-based) |
| `strand` | str | Strand direction (`+` or `-`) |

### Example Input

```csv
seqnames,position,strand
chr17,42775424,+
chr1,165743188,-
chr1,44776696,+
chr7,76327559,-
chrM,3812,+
chr2,232894801,-
chr19,16185627,+
chr11,64238581,+
chr1,1407207,-
chrY,57211982,+
```

------------------------------------------------------------------------

## 📈 Output Format

| Column Name | Description |
|-------------|-------------|
| `modcon_score` | Final ModCon conservation score (0-1) |

### Example Output

```
seqnames	position	strand	modcon_score
chr17	42775424	+	0.7047
chr1	165743188	-	0.2386
chr1	44776696	+	0.6802
chr7	76327559	-	0.0043
chrM	3812	+	0.5442
chr2	232894801	-	0.1426
chr19	16185627	+	0.2492
chr11	64238581	+	0.0354
chr1	1407207	-	0.3815
chrY	57211982	+	0.2118
```

------------------------------------------------------------------------

## 🤝 Contributing

We welcome bug reports and pull requests!

------------------------------------------------------------------------

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

------------------------------------------------------------------------

## ✉️ Contact

For questions or suggestions, please contact the [project maintainer](https://github.com/Gang1998c).
