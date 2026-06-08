# ModCon: An Integrated Multimodal Pipeline for RNA Modification Conservation Scoring

[![Python](https://img.shields.io/badge/Python-3.8%252B-blue.svg)](https://www.python.org/)
[![R](https://img.shields.io/badge/R-4.2.2%252B-seafoam.svg)](https://www.r-project.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.2-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-yellow.svg)](https://huggingface.co/)

**ModCon** 是一款高性能、跨语言的软件工具，旨在评估和打分跨物种（人类、小鼠和猪）的 RNA 修饰位点保守性。通过将先进的序列标记化（Sequence Tokenization）与高维基因组拓扑学相结合，该流水线提供了一个代表修饰保守性的强健指标。

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

在 R 中安装所需的包（如适用）：

```r
# 安装所需的 R 包
install.packages(c("data.table", "dplyr", "ggplot2"))
# 如需特定的生物信息学包
if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
BiocManager::install("GenomicRanges")
```

### 3. 克隆仓库 (Clone Repository)

```bash
git clone https://github.com/Gang1998c/Modcon.git
cd Modcon
```

### 4. 安装 ModCon 包 (Install ModCon Package)

```bash
pip install -e .
# 或者
python setup.py install
```

------------------------------------------------------------------------

## 📖 快速开始 (Quick Start)

### 基本用法 (Basic Usage)

```python
from modcon import ModConPipeline

# 初始化管道
pipeline = ModConPipeline(config_file="config.yaml")

# 加载 RNA 修饰位点数据
modifications = pipeline.load_modifications("path/to/modifications.csv")

# 执行完整的分析流程
results = pipeline.run(modifications, target_species=["mm39", "susScr11"])

# 获取保守性评分
conservation_scores = results.get_modcon_scores()
print(conservation_scores)
```

### 命令行调用 (Command-line Interface)

```bash
# 运行完整的分析流程
python master_pipeline.py \
  --user_df_path modifications.csv \
  --ontdata_path ontdata.csv \
  --hg38_fa hg38.fa \
  --mm39_fa mm39.fa \
  --sus11_fa susScr11.fa \
  --mm39_chain hg38ToMm39.over.chain \
  --sus11_chain hg38ToSusScr11.over.chain \
  --output_path results.csv

# 获取帮助信息
python master_pipeline.py --help
```

------------------------------------------------------------------------

## 📊 输入数据格式 (Input Data Format)

### 必需列 (Required Columns)

| 列名 | 类型 | 描述 |
|------|------|------|
| `chromosome` | str | 染色体编号 (如 `chr1`) |
| `position` | int | 修饰位点坐标 (hg38) |
| `strand` | str | 链方向 (`+` 或 `-`) |
| `modification_type` | str | 修饰类型 (如 `m6A`, `m1A`, `Ψ`) |
| `score` | float | 初始评分 (可选) |

### 示例输入 (Example Input)

```csv
chromosome,position,strand,modification_type,score
chr1,1000,+,m6A,0.85
chr1,2000,-,m1A,0.92
chr2,5000,+,m6A,0.78
```

------------------------------------------------------------------------

## 📈 输出结果格式 (Output Format)

管道生成包含以下信息的结果文件：

| 列名 | 描述 |
|------|------|
| `modcon_score` | 最终的 ModCon 保守性评分 (0-1) |
| `conservation_level` | 保守性等级 (`High`, `Medium`, `Low`) |
| `cross_species_tracking` | 跨物种追踪结果 |
| `topology_features` | 高维拓扑特征 |
| `deep_learning_probability` | 深度学习模型预测概率 |
| `ensemble_confidence` | 集成模型置信度 |

------------------------------------------------------------------------

## 🔧 高级配置 (Advanced Configuration)

### 配置文件示例 (Configuration File Example)

```yaml
# config.yaml
modcon:
  reference_genome: "hg38"
  target_species:
    - "mm39"
    - "susScr11"
  
  # 特征提取参数
  feature_extraction:
    window_size: 800  # bp
    num_features: 40
    
  # 深度学习模型参数
  deep_learning:
    model_name: "dnabert2"
    batch_size: 32
    device: "cuda"  # 或 "cpu"
    
  # 集成聚合权重
  ensemble:
    cross_species_weight: 0.3
    empirical_score_weight: 0.3
    deep_learning_weight: 0.4
```

### 自定义参数调整 (Custom Parameter Tuning)

```python
pipeline = ModConPipeline(
    reference_genome="hg38",
    window_size=800,
    num_features=40,
    batch_size=32,
    device="cuda"
)
```

------------------------------------------------------------------------

## 📚 详细文档 (Documentation)

- [架构设计文档](./docs/architecture.md)
- [API 参考](./docs/api_reference.md)
- [常见问题](./docs/faq.md)
- [教程与示例](./docs/tutorials/)

------------------------------------------------------------------------

## 🧪 测试 (Testing)

运行测试套件确保安装正确：

```bash
# 运行所有测试
pytest tests/

# 运行特定测试
pytest tests/test_coordinate_liftover.py -v

# 生成覆盖率报告
pytest --cov=modcon tests/
```

------------------------------------------------------------------------

## 📝 使用示例 (Usage Examples)

### 示例 1：分析单个修饰位点 (Analyze Single Site)

```python
from modcon import ModConPipeline

pipeline = ModConPipeline()

# 单个位点
site = {
    "chromosome": "chr1",
    "position": 10000,
    "strand": "+",
    "modification_type": "m6A"
}

result = pipeline.analyze_single_site(site)
print(f"ModCon Score: {result['modcon_score']}")
print(f"Conservation Level: {result['conservation_level']}")
```

### 示例 2：批量分析 (Batch Analysis)

```python
import pandas as pd
from modcon import ModConPipeline

# 读取输入数据
input_data = pd.read_csv("modifications.csv")

pipeline = ModConPipeline()
results = pipeline.analyze_batch(input_data, n_jobs=4)

# 保存结果
results.to_csv("conservation_scores.csv", index=False)
```

### 示例 3：可视化结果 (Visualize Results)

```python
from modcon.visualization import plot_conservation_distribution

# 绘制保守性分布
plot_conservation_distribution(
    results,
    output_file="conservation_distribution.png"
)
```

------------------------------------------------------------------------

## 🤝 贡献 (Contributing)

我们欢迎社区的贡献！请遵循以下步骤：

1. Fork 该仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

请确保你的代码遵循 [PEP 8](https://www.python.org/dev/peps/pep-0008/) 风格指南。

------------------------------------------------------------------------

## 📋 许可证 (License)

该项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

------------------------------------------------------------------------

## 📧 联系方式 (Contact)

- 作者：Gang1998c
- 问题报告：[GitHub Issues](https://github.com/Gang1998c/Modcon/issues)
- 邮件：[您的邮箱]

------------------------------------------------------------------------

## 📖 引用 (Citation)

如果你在研究中使用了 ModCon，请引用：

```bibtex
@software{modcon2024,
  author = {Gang, Author},
  title = {ModCon: An Integrated Multimodal Pipeline for RNA Modification Conservation Scoring},
  year = {2024},
  url = {https://github.com/Gang1998c/Modcon}
}
```

------------------------------------------------------------------------

## ⚠️ 免责声明 (Disclaimer)

本工具用于研究目的。在使用结果做出重要决策前，请进行充分的验证和评估。

------------------------------------------------------------------------

## 🙏 致谢 (Acknowledgments)

感谢所有贡献者和使用者的支持。本项目基于最新的生物信息学和深度学习技术。
