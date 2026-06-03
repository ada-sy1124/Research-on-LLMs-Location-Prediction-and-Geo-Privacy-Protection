# Research on LLMs Location Prediction and Geo-Privacy Protection

本项目聚焦一个核心问题：**大模型可以从街景图片中“猜到你在哪”吗？如果可以，它到底依赖了哪些视觉线索？**

我们设计了一套可复现的数据流水线，系统评估 LLM 的地理定位能力，并通过“类别级遮挡实验”寻找影响定位精度的关键目标（如建筑、路面标线、交通标志等），为地理隐私防护提供实证基础。

## 研究目标

1. 评估多模态 LLM 对图片地理位置的预测能力。  
2. 定位模型推理时依赖的关键视觉类别。  
3. 在“隐私增益”与“信息损失”之间寻找可用的遮挡策略（Pareto 最优）。  
4. 产出可用于后续微调的数据集与训练样本。  

## 方法流程

1. `YES/NO` 初筛  
调用 Gemini 对街景图进行坐标预测，计算与真实坐标的距离误差 `d`。  
若 `d <= 5(km)` 记为 `YES`（模型可定位），否则为 `NO`。

2. 类别级物理遮挡（afterSAM）  
对 `YES` 样本，用 SAM 按类别生成掩码并将对应像素置黑，构造多组遮挡图。

3. 遮挡后重测（YES_Mask）  
对每个遮挡版本再次预测坐标，得到新误差 `d'`，计算 `d_diff = d' - d`，衡量隐私增益。

4. Pareto 选择最优遮挡类别  
在“隐私增益高”和“遮挡面积小”之间筛选非支配解，输出每张图的推荐类别标签。

5. 构建训练集并微调  
导出多模态 `jsonl`，可用于 QLoRA（示例脚本基于 Qwen2.5-VL + Swift）。

## 仓库结构

- `README.md`：项目总览（当前文件）
- `code/README.md`：代码重构说明
- `code/src/geoai_pipeline/`：主流程实现（配置、常量、工具、pipeline）
- `code/*.py`：兼容原始命名的脚本入口
- `code/辅助功能/`：数据预览、统计、合并等辅助脚本
- `code/模型训练/QLoRA.py`：训练示例
- `data/`：默认数据输出目录

## 快速开始

### 1) 环境准备

建议使用 Python 3.10+，并按实际需要安装依赖（至少包含）：

- `datasets`
- `google-genai`
- `python-dotenv`
- `numpy`
- `pillow`
- `tqdm`
- `torch`
- `sam3`（用于遮挡流程）
- `swift`（用于 QLoRA 训练）

### 2) 配置环境变量

```bash
cp code/.env.example code/.env
```

按你的本地环境修改 `code/.env`（尤其是 `GEMINI_API_KEY`、输入输出路径、起止索引等）。

### 3) 运行主流程（在 `code/` 目录）

```bash
cd code

# Step 1: 从原始街景数据筛选 YES/NO
python 最新筛数据集_gemini从0构建YESandNO.py

# Step 2: 对 YES 数据做类别级遮挡
python 从YES构建afterSAM.py

# Step 3: 对遮挡图重测定位误差
python gemini_sam从afterSAM构建YES_Mask.py

# Step 4: 导出训练用 JSONL
python 训练集文本构建.py
```

如需统计、合并、预览数据，可使用 `code/辅助功能/` 下脚本。

## 主要输出

- `data/YES` / `data/NO`：初筛结果  
- `data/YES_NEW_afterSAM`：遮挡后样本  
- `data/YES_Mask_*`：遮挡重测结果（含 `d_prime`、`d_diff`）  
- `data/trainset.jsonl`：训练数据  

## 许可

本项目采用 [MIT License](./LICENSE)。
