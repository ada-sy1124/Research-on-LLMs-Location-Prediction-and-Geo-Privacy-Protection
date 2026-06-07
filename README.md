
# DCSD-2.0: 物理感知的可微因果子集发现

**Physically-Aware Differentiable Causal Subset Discovery for VLM Geolocation**

## 1. 架构总览 (Architecture Overview)

本算法旨在解答：“在多模态大模型（VLM）进行地理定位时，图像中的哪些特定物理目标构成了模型预测经纬度的**核心因果锚点**？”
为突破离散组合搜索的算力瓶颈与传统交叉熵无视物理距离的缺陷，本架构将目标遮挡过程**连续化**，并将地球大圆距离测量**可微化**，实现端到端的黑盒特征寻根。

---

## 2. 第一阶段：离散目标提取与连续化松弛 (Initialization & Relaxation)

### 2.1 先验目标池化 (Object Grounding)

1. **启发式发现：** 给定原图 $I$，利用 VLM 的思维链（CoT）输出定位依据，提取出 $N$ 个显著的地理实体名词。
2. **掩码生成：** 将名词输入 Segment Anything (SAM)，获取 $N$ 个绝对离散的二进制掩码矩阵集合：

$$\mathcal{M} = \{M_1, M_2, \dots, M_N\}, \quad M_i \in \{0, 1\}^{H \times W}$$



### 2.2 遮挡权重的连续化映射 (Continuous Relaxation)

为每个掩码 $M_i$ 声明一个可学习的标量参数 $\alpha_i$。在整个优化过程中，**大模型的全部参数被冻结，仅有 $\alpha$ 参与梯度更新**。
将 $\alpha_i$ 经过 Sigmoid 激活，生成 $[0, 1]$ 之间的连续遮挡概率 $p_i$：


$$p_i = \text{Sigmoid}(\alpha_i)$$


利用广播机制（Broadcasting），将原始图像 $I$ 与加权后的掩码组合进行融合，生成具有半透明/灰色遮挡层的合成输入图像 $I_{input}$：


$$I_{input} = I \odot (1 - \sum_{i=1}^{N} p_i \cdot M_i)$$

---

## 3. 第二阶段：物理感知的期望标记损失 (Physically-Aware Token Loss)

将 $I_{input}$ 与提问提示词（Prompt）喂给 VLM。采用 **Teacher Forcing** 机制，强制将真实的经纬度坐标字符串 $Y_{true} = [y_1, y_2, \dots, y_m]$ 喂入解码器（Decoder）。

### 3.1 暴力逻辑阉割 (Logit Masking)

在 VLM 输出每一位字符的 Logits 时，由于非数字 Token 会破坏距离计算，我们在 Softmax 之前进行掩码拦截。
定义合法字符集 $\mathcal{V}_{num} = \{0..9, ., -, ,\}$。对于词表中的第 $k$ 个 Token $v_k$：

$$\text{Logit}'_{k} = 
\begin{cases} 
\text{Logit}_{k}, & \text{if } v_k \in \mathcal{V}_{num} \\
-\infty, & \text{otherwise}
\end{cases}$$

经过 Softmax 后，所有非数字字符的概率被绝对锁死为 0。模型被迫输出长度为 13（或等同于 $\mathcal{V}_{num}$ 大小）的概率向量 $\mathbf{P}_t$。

### 3.2 物理距离矩阵构建 (The 111km/Cosine Distance Matrix)

针对真实序列中的第 $t$ 个位置（假设其真实数字为 $y_t$，科学指数/位数权重为 $E_t$），我们根据 **WGS-84 测绘标准**，构建一个固定常数构成的距离惩罚向量 $D_t$。
对于合法的数字候选词 $v_j \in \{0..9\}$：

* **如果当前预测的是纬度 (Latitude)：**

$$D_{t, j}^{(lat)} = |v_j - y_t| \times 10^{E_t} \times 111$$


* **如果当前预测的是经度 (Longitude)：**
引入高纬度经线收缩的余弦惩罚机制（其中 $\text{Lat}_{true}$ 为真实纬度）：

$$D_{t, j}^{(lon)} = |v_j - y_t| \times 10^{E_t} \times 111 \times \cos(\text{Lat}_{true})$$



*(对于小数点、逗号等符号位，若模型猜错，设定一个统一的常量惩罚值 $M$)*

### 3.3 可微期望距离点积 (Differentiable Expected Distance)

将模型输出的概率分布 $\mathbf{P}_t$ 与写死的物理距离矩阵 $D_t$ 进行点积，计算出该位置上产生的**预期物理偏差（公里数）**：


$$L_{dist\_t} = \sum_{j \in \mathcal{V}_{num}} P(v_j \mid I_{input}, y_{<t}) \cdot D_{t, j}$$

---

## 4. 第三阶段：目标函数与离散化 (Optimization & Binarization)

### 4.1 最终损失函数的设计

为了寻找能够导致最大定位破坏的**最少**目标组合，总损失函数定义为：


$$L_{total} = \lambda \sum_{i=1}^{N} p_i - \sum_{t=1}^{m} L_{dist\_t}$$

* **第一项（L1 稀疏正则化）：** $\sum p_i$ 迫使非必要目标的遮挡概率趋近于 0（即保留原图）。
* **第二项（负期望距离和）：** 我们通过**最小化负距离**（等同于最大化正距离），驱使梯度去寻找让模型坐标预测偏离最离谱的遮挡组合。

### 4.2 反向传播与最终阈值切割

1. 调用 `L_total.backward()`。误差梯度从坐标距离出发，穿过冻结的 VLM 神经网络，落回图像并精确更新 $\alpha$ 参数。
2. 使用 Adam 优化器迭代 30~50 步，直到 Loss 收敛。
3. **二值化输出：** 设定硬阈值 $\tau = 0.5$。若 $p_i > \tau$，则将其视为“核心因果目标”，在输出图中将其涂为 100% 纯黑；若 $p_i \le \tau$，则恢复原图。

---

> **核心工程优势：** 本架构无需修改任何 VLM 底层源码，无需预训练任何分类头。仅需一次前向传播即可完成极其复杂的地球物理感知梯度计算，将 $O(2^N)$ 的搜索复杂度降维打击至 $O(1)$ 的连续优化，速度极快且极具物理可解释性。

---
