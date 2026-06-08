---
editor_options: 
  markdown: 
    wrap: 72
---

# ModCon: An Integrated Multimodal Pipeline for RNA Modification Conservation Scoring

[![Python](https://img.shields.io/badge/Python-3.8%252B-blue.svg)](https://www.python.org/)
[![R](https://img.shields.io/badge/R-4.0%252B-seafoam.svg)](https://www.r-project.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%252B-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-yellow.svg)](https://huggingface.co/)

**ModCon**
是一款高性能、跨语言的软件工具，旨在评估和打分跨物种（人类、小鼠和猪）的
RNA 修饰位点保守性。通过将先进的序列标记化（Sequence
Tokenization）与高维基因组拓扑学相结合，该流水线提供了一个代表修饰保守性的强健指标。

------------------------------------------------------------------------

## 🧬 架构概述 (Architectural Overview)

该流水线执行一个自动化的 5 阶段流处理，在 Python 和 R
运行环境之间无缝衔接：

1.  **基础特征映射 (Base Feature Mapping):**
    将位点坐标与基础参考目录进行映射，以继承经验统计锚点（`noes` 和 `pm`
    指标）。
2.  **跨物种坐标转换 (Cross-Species Coordinate Liftover):**
    利用精准且感知链方向（strand-aware）的逻辑，将人类 `hg38`
    修饰位点投射到目标参考网络（`mm39` 和 `susScr11`）。
3.  **高维拓扑特征谱分析 (High-Dimensional Topology Profiling):** 通过
    `m6ALogisticModel` 包执行 R 语言子进程，将坐标提升至 `hg19`，分离出
    40 个不同的基因组特征，并提取中心化的 800bp 参考序列映射图[cite:
    2]。
4.  **碱基特异性多模态推理 (Base-Specific Multimodal Inference):**
    通过自动化的序列路由门控转发数据阵列，以查询专门的深度学习架构（融合了
    MLP 网络与 DNABERT-2 Transformer 嵌入）[cite: 4]。
5.  **集成聚合 (Ensemble Aggregation):**
    将跨物种追踪结果、经验库得分和深度学习分类概率聚合为一个单一的整体指标：**ModCon
    Score**。

------------------------------------------------------------------------

## 🛠️ 环境准备与安装 (Prerequisites & Installation)

### 1. Python 环境配置 (Python Environment Setup)

请确保你激活了一个运行 Python 3.8+ 的 Anaconda 环境：

\`\`\`bash conda create -n dnabert2 python=3.8 -y conda activate
dnabert2 pip install torch transformers pandas numpy biopython
python-liftover joblib tqdm
