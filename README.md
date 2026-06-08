# ModCon: An Integrated Multimodal Pipeline for RNA Modification Conservation Scoring

[![Python](https://img.shields.io/badge/Python-3.8%252B-blue.svg)](https://www.python.org/)
[![R](https://img.shields.io/badge/R-4.2.2%252B-seafoam.svg)](https://www.r-project.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.2-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-yellow.svg)](https://huggingface.co/)

**ModCon** 是一款高性能、跨语言的软件工具，旨在评估和打分跨物种（人类、小鼠和猪）的 RNA 修饰位点保守性。通过将先进的序列标记化（Sequence Tokenization）与多模态的深度学习架构相结合，该工具提供了一套全面且可解释的解决方案来量化不同物种间 RNA 修饰的保守性特征。

------------------------------------------------------------------------

## 🧬 架构概述 (Architectural Overview)

该流水线执行一个自动化的 5 阶段流处理，在 Python 和 R 运行环境之间无缝衔接：

1.  **基础特征映射 (Base Feature Mapping):**
    将位点坐标与基础参考目录进行映射，以继承经验统计锚点（`noes` 和 `pm` 指标）。

2.  **跨物种坐标转换 (Cross-Species Coordinate Liftover):**
    利用精准且感知链方向（strand-aware）的逻辑，将人类 `hg38` 修饰位点投射到目标参考网络（`mm39` 和 `susScr11`）。

3.  **高维拓扑特征谱分析 (High-Dimensional Topology Profiling):**
    通过 `m6ALogisticModel` 包执行 R 语言子进程，将坐标提升至 `hg19`，分离出 40 个不同的基因组特征，并提取中心化的 800bp 参考序列映射图。

4.  **碱基特异性多模态推理 (Base-Specific Multimodal Inference):**
    通过自动化的序列路由门控转发数据阵列，以查询专门的深度学习架构（融合了 MLP 网络与 DNABERT-2 Transformer 嵌入）。

5.  **集成聚合 (Ensemble Aggregation):**
    将跨物种追踪结果、经验库得分和深度学习分类概率聚合为一个单一的整体指标：**ModCon Score**。

### 流程图 (Pipeline Workflow)

![ModCon Pipeline Architecture](https://github.com/Gang1998c/Modcon/raw/main/Modcon_Framework.jpg)

该图展示了完整的分析流程：
- **数据收集** (Data Collection): 来自多物种（人类、小鼠、猪）的 RNA 修饰数据
- **数据处理** (Data Processing): 碱基调用、ELIGOS 修饰检测和位点映射
- **模型构建** (Model Construction): 同源修饰一致性、物种内重复性、序列上下文保守性分析
- **微调大语言模型** (Fine-tuned Large Language Model): MLP 分类器和 DNABERT-2 Transformer 集成
- **模型验证与应用** (Model Validation and Application): NGS 验证、变异密度验证、功能分析和 Web 服务预测

------------------------------------------------------------------------

## 🛠️ 环境准备与安装 (Prerequisites & Installation)

### 1. Python 环境配置 (Python Environment Setup)

请确保你激活了一个运行 Python 3.8+ 的 Anaconda 环境：

```bash
# 创建新的 conda 环境
conda create -n modcon python=3.8 -y

# 激活环境
conda activate modcon

# 安装所需依赖
pip install -r requirements.txt
```

### 2. R 环境配置 (R Environment Setup)

在 R 中执行以下命令安装所需的包：

```r
# 1. 安装常规 CRAN 依赖包
install.packages(c("dplyr", "caret", "e1071", "ROCR", "pROC", "devtools"))

# 2. 安装 Bioconductor 核心生物信息包及巨型基因组数据集
if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
BiocManager::install(c("GenomicRanges", "rtracklayer", "SummarizedExperiment", 
                       "TxDb.Hsapiens.UCSC.hg19.knownGene", 
                       "BSgenome.Hsapiens.UCSC.hg19", 
                       "BSgenome.Hsapiens.UCSC.hg38",
                       "fitCons.UCSC.hg19", 
                       "phastCons100way.UCSC.hg19"))

# 3. 安装实验室专用多模态特征提取包
devtools::install_github("m6ALogisticModel")
```

### 3. 克隆仓库 (Clone Repository)

```bash
git clone https://github.com/Gang1998c/Modcon.git
cd Modcon
```

------------------------------------------------------------------------

## 📖 快速开始 (Quick Start)

### 基本用法 (Basic Usage)

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

<sub>用户只需提供 DNA 格式的单碱基位点数据（1-based），格式参考 `server_example_data.csv`（参考基因组：hg38）。其他需要的文件以及训练完成的模型已上传到 Hugging Face，用户可在该处下载。</sub>

### 参数说明 (Parameter Illustration)

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

## 📊 输入数据格式 (Input Data Format)

### 必需列 (Required Columns)

| 列名 | 类型 | 描述 |
|------|------|------|
| `seqnames` | str | 染色体编号 (如 `chr1`) |
| `position` | int | 修饰位点坐标 (hg38, 1-based) |
| `strand` | str | 链方向 (`+` 或 `-`) |

### 示例输入 (Example Input)

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

## 📈 输出结果格式 (Output Format)

| 列名 | 描述 |
|------|------|
| `modcon_score` | 最终的 ModCon 保守性评分 (0-1) |

### 示例输出 (Example Output)

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

## 🤝 贡献 (Contributing)

欢迎提交问题报告和拉取请求！

------------------------------------------------------------------------

## 📄 许可证 (License)

本项目采用 MIT 许可证。详见 [LICENSE](./LICENSE) 文件。

------------------------------------------------------------------------

## ✉️ 联系方式 (Contact)

如有问题或建议，请联系 [项目维护者](https://github.com/Gang1998c)。
