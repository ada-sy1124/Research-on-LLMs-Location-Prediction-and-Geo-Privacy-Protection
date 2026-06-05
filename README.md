# PCI-CAP: 基于投影器空间对比反演的因果对抗保护框架

**(Projector-space Contrastive Inversion for Causal Adversarial Protection)**

---

## 核心摘要 (Abstract Highlights)

PCI-CAP 是一种针对多模态大语言模型（MLLM）的零监督、低损耗隐私保护框架。本框架突破了传统像素级对抗攻击中“计算昂贵”与“视觉噪音过载”的瓶颈，首次提出将防御阵地转移至 **LLM 语义咽喉处（Post-Projector Space）**。通过负向掩码反演（Learned Negative Masking）精准定位隐私泄露的语义元凶，并结合 SAM3 与物理距离阈值，实现了“恰好致盲”的极限低损耗图像保护。

**三大核心优势：**

1. **从像素到语义 (From Pixels to Semantics)**：在高维 Token 空间过滤低级视觉噪音，实现极高精度的因果归因。
2. **极小化物理污染 (Minimal-Damage Protection)**：SAM3 物理结界结合真实地理距离裁判，将对抗噪声严格限制在致死目标内。
3. **零真实标签依赖 (Zero-Ground-Truth Defense)**：利用模型自身的“第一直觉（伪标签）”作为靶点，完全适配用户本地部署的真实场景。

---

## 阶段一：系统初始化与靶标锚定 (System Initialization & Target Anchoring)

本阶段在完全无监督（无真实 GPS 坐标）的前提下，获取代理模型的内部认知基准。

* **输入**：原始待保护图像 $X$。
* **流程**：将 $X$ 输入本地开源代理模型（Surrogate MLLM），进行纯前向自回归推理。
* **输出**：提取模型输出的坐标序列作为**伪标签 (Pseudo-label)** $Y_{target}$（例如：`43.6426, -79.3870`）。该坐标在物理世界的绝对真伪并不重要，它是后续摧毁模型认知的唯一数学锚点。

---

## 阶段二：语义空间因果反演 (Semantic-Space Causal Inversion)

这是框架的核心创新层。抛弃像素级的正向加噪，在 LLM 接收视觉信号的“咽喉”处进行负向擦除。

### 1. 特征截获 (Feature Interception)

图像 $X$ 穿过 Vision Encoder 与 Projector 后，截获即将送入 LLM Decoder 的视觉 Token 矩阵：


$$Z = \{z_1, z_2, ..., z_N\}$$


*(注：此时的 $Z$ 已转化为 LLM 可理解的高度抽象语义碎片)*

### 2. 连续掩码初始化 (Mask Initialization)

定义一个可学习的连续掩码向量 $M \in [0, 1]^N$，初始值全设为 1。将模型的实际视觉输入重构为门控状态：


$$Z_{masked} = Z \odot M$$

### 3. 反演优化 (The Inversion Objective)

冻结大模型所有权重，使用 Adam 优化器仅更新掩码 $M$，最小化以下反演目标函数：


$$\mathcal{L}_{inv} = -\text{CrossEntropy}(Y_{target} | Z_{masked}) + \lambda \|1 - M\|_1$$

* **对抗项 (交叉熵)**：强迫模型无法输出伪标签 $Y_{target}$，瓦解其初始地理认知。
* **稀疏正则项 (L1 惩罚)**：强迫 $M$ 尽可能保持为 1（即遮挡面积最小化），逼迫优化器寻找极其稀少的“致死 Token”。

### 4. 死穴提取 (Fatal Token Extraction)

优化收敛后，提取 $M$ 中权重逼近于 0 的索引位置（Index），即为导致地理隐私泄露的核心语义词元。

---

## 阶段三：跨模态降维与物理结界 (Cross-Modal Grounding & Physical Masking)

将抽象的语义特征打回物理世界，建立精确的防泄漏隔离区。

### 1. 网格逆映射 (Grid Back-projection)

利用 Vision Transformer (ViT) 严格的空间序列守恒特性，将“致死 Token”的序列索引，通过下采样比例逆向映射回原图 $X$ 的二维像素坐标点 $(x, y)$。

### 2. SAM3 语义切割 (Semantic Segmentation via SAM3)

将上述 $(x, y)$ 坐标集合作为提示点（Prompt Points）输入 Segment Anything Model 3 (SAM3)。SAM3 输出包含该坐标的精确物理边界，生成**二维二值化物理掩码矩阵 $M_{phys}$**（例如：精确剥离出图像中的某块路牌）。

---

## 阶段四：掩码门控毒药与物理裁判 (Masked Poisoning & Physical Early-Stopping)

回到传统的像素端对抗攻击，利用物理裁判实现一击脱离。

### 1. 掩码定向下毒 (Mask-Gated Adversarial Perturbation)

启动 PGD (Projected Gradient Descent) 对抗迭代。每次计算出的对抗梯度 $\nabla_X \mathcal{L}$ 必须受到物理结界的严格限制：


$$\delta_{t+1} = \Pi_{\epsilon} \left( \delta_t + \alpha \cdot \text{sign}(\nabla_X \mathcal{L}) \odot M_{phys} \right)$$


确保对抗噪声 100% 倾泻于目标物体，绝对保护背景像素（天空、建筑）的画质。

### 2. 物理距离裁判 (Physical Haversine Early-Stopping)

* **监控机制**：在迭代过程中，每注入一次噪声，代理模型重新生成一次预测坐标 $Y_{current}$。
* **悬崖触发**：调用半正矢公式（Haversine Formula）计算 $Y_{current}$ 与初始伪标签 $Y_{target}$ 之间的地球球面距离 $\Delta \text{Distance}$。
* **一击脱离**：一旦 $\Delta \text{Distance} > \tau$（例如 $\tau = 50\text{km}$），立刻强制终止迭代循环（Break）。

### 3. 黑盒致盲迁移 (Black-box Transferability)

输出最终的受保护图像 $X_{adv}$。由于主流大模型（如 GPT-4o, Claude 3.5）底层共享相似的视觉注意力和 OCR 处理机制，$X_{adv}$ 将成功引发跨模型的定位认知崩溃。
