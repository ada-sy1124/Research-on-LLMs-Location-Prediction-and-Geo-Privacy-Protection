对，你刚才的理解和我真正想表达的方法之间有一个差别：

> **不是“给词表里每个 token 都预存一个 Haversine 距离，然后和每一步概率向量点积”。**
> 更准确的是：**先定义一批合法的坐标 token 序列，每条序列对应一个完整坐标；再给每条完整序列预计算 Haversine 距离；优化时用模型给这些序列分配的概率去加权距离。**

也就是说，Haversine 不是作用在“单个 token”上，而是作用在“完整坐标序列”上。

---

# 1. 先看为什么不能按单 token 算

假设模型输出坐标：

```text
51.5074, -0.1278
```

它可能被 tokenizer 切成：

```text
["51", ".", "507", "4", ",", "-", "0", ".", "127", "8"]
```

单独看 token `"51"` 没有完整地理意义。
单独看 token `"507"` 也不知道它是纬度小数部分，还是经度小数部分，还是普通文本里的数字。

所以如果你给每个 token 一个固定地理距离：

[
\text{token} \rightarrow \text{distance}
]

这是不合理的。

真正有地理意义的是完整序列：

```text
"51.5074,-0.1278"
```

它可以解析成：

[
(51.5074, -0.1278)
]

然后才能和目标坐标算 Haversine 距离。

---

# 2. 我说的方法：序列级 Geodesic Token Risk

核心思想是：

> 把“解码后算距离”改成“在合法坐标序列概率分布上算期望距离”。

假设我们有一个候选坐标序列集合：

[
\mathcal{B}={y^{(1)}, y^{(2)}, ..., y^{(K)}}
]

每个 (y^{(j)}) 都是一条合法坐标 token 序列。

例如：

```text
y(1) = "51.5074,-0.1278"
y(2) = "51.5075,-0.1278"
y(3) = "51.5074,-0.1279"
y(4) = "48.8566,+002.3522"
...
```

每条序列都可以解析成坐标：

[
g(y^{(j)})=(\phi_j,\lambda_j)
]

然后你提前计算它和目标坐标 (g^*) 的 Haversine 距离：

[
D_j = d_{hav}(g(y^{(j)}), g^*)
]

这里 (D_j) 是常数。

接下来，VLM 对这条序列的概率是：

[
p_\theta(y^{(j)}|x_m)
=====================

\prod_{t=1}^{T}
p_\theta(y_t^{(j)}|x_m, y_{<t}^{(j)})
]

其中：

* (x_m)：被连续实体 mask 修改后的图像；
* (y_t^{(j)})：第 (j) 条坐标序列的第 (t) 个 token；
* (p_\theta)：冻结 VLM 的 token 概率。

于是定义：

[
\mathcal{R}_{geo}(x_m)
======================

\sum_{j=1}^{K}
p_\theta(y^{(j)}|x_m)
D_j
]

这就是我说的 **Geodesic Token Risk**。

它的意思是：

> 如果模型把高概率分给地理距离很远的坐标序列，loss 就大；
> 如果模型把高概率分给地理距离很近的坐标序列，loss 就小。

---

# 3. 为什么这个方法不断梯度？

因为你没有走这条断梯度路径：

```text
argmax token → decode string → parse float → haversine
```

而是走这条路径：

```text
entity mask → VLM logits → 坐标序列概率 → 加权 Haversine 距离
```

Haversine 距离 (D_j) 是提前算好的常数，不需要对它求导。

梯度来自：

[
\frac{\partial \mathcal{R}_{geo}}{\partial x_m}
===============================================

\sum_j
D_j
\frac{\partial p_\theta(y^{(j)}|x_m)}{\partial x_m}
]

也就是说，梯度不是从 Haversine 公式穿回来，而是从**坐标序列概率**穿回来。

所以它不会断。

---

# 4. 那普通词 token 怎么办？

比如模型在某一步可能输出：

```text
"London"
"street"
"I"
"cannot"
```

这些 token 没有坐标意义。我的方法不是给它们算 Haversine，而是用一个**坐标语法 grammar**把它们排除。

例如你强制模型输出固定格式：

```text
LAT=+51.5074;LON=-000.1278
```

那么每个位置都有合法 token 集合。

例如：

| 位置     | 合法 token             |
| ------ | -------------------- |
| `LAT=` | 只能是 `LAT=` 或对应 token |
| 纬度符号   | `+` 或 `-`            |
| 纬度整数位  | 数字                   |
| 小数点    | `.`                  |
| 纬度小数位  | 数字                   |
| 分隔符    | `;LON=`              |
| 经度符号   | `+` 或 `-`            |
| 经度整数位  | 数字                   |
| 经度小数位  | 数字                   |

普通词 `"London"` 不属于合法转移，所以不进入 (\mathcal{B})。

但是我们还要防止模型把概率大量分给这些普通词。因此加一个有效性损失：

[
\mathcal{L}_{valid}
===================

-\log
\sum_{y \in \mathcal{B}}
p_\theta(y|x_m)
]

它的含义是：

> 模型应该把概率质量分配给合法坐标序列，而不是输出普通句子或无效文本。

最终 loss 可以写成：

[
\mathcal{L}
===========

\alpha \mathcal{R}*{geo}
+
\beta \mathcal{L}*{valid}
+
\lambda \mathcal{L}_{mask}
]

其中：

* (\mathcal{R}_{geo})：合法坐标序列上的 Haversine 期望风险；
* (\mathcal{L}_{valid})：惩罚非坐标输出；
* (\mathcal{L}_{mask})：实体子集稀疏性和面积正则。

---

# 5. 一个具体小例子

假设目标坐标是伦敦：

[
g^*=(51.5074,-0.1278)
]

我们构造 4 条候选坐标序列：

| 序列        |                             坐标 | Haversine 距离 |
| --------- | -----------------------------: | -----------: |
| (y^{(1)}) |      London: (51.5074,-0.1278) |         0 km |
| (y^{(2)}) | near London: (51.6000,-0.1000) |        10 km |
| (y^{(3)}) |        Paris: (48.8566,2.3522) |       343 km |
| (y^{(4)}) |      Tokyo: (35.6762,139.6503) |      9558 km |

模型在当前 masked image 上给这些序列的概率是：

| 序列         |    概率 |      |
| ---------- | ----: | ---- |
| (p(y^{(1)} | x_m)) | 0.50 |
| (p(y^{(2)} | x_m)) | 0.20 |
| (p(y^{(3)} | x_m)) | 0.20 |
| (p(y^{(4)} | x_m)) | 0.10 |

那么：

[
\mathcal{R}_{geo}
=================

0.50 \cdot 0
+
0.20 \cdot 10
+
0.20 \cdot 343
+
0.10 \cdot 9558
]

# [

# 0 + 2 + 68.6 + 955.8

1026.4
]

如果你遮掉某些实体后，模型对 Tokyo 坐标序列的概率上升，那么 (\mathcal{R}*{geo}) 变大。
如果你只保留某些实体后，模型仍然把概率给 London 附近，那么 (\mathcal{R}*{geo}) 变小。

于是这个 loss 就可以告诉你：

> 当前实体 mask 是否让模型的地理预测远离目标位置。

---

# 6. 这和 token CE 有什么区别？

普通 token CE 是：

[
\mathcal{L}_{CE}
================

-\sum_t \log p_\theta(y_t^*|x_m,y_{<t}^*)
]

它只知道目标 token 是什么。
它不知道地理距离。

比如目标是：

```text
51.5074,-0.1278
```

模型输出：

```text
51.5075,-0.1278
```

和输出：

```text
35.6762,139.6503
```

从 token CE 角度看，二者都只是 token 不匹配。
但从地理角度看，前者几乎还是伦敦，后者已经到东京了。

你的 Haversine token risk 可以区分这两者。

这就是创新点。

---

# 7. 候选坐标序列 (\mathcal{B}) 从哪里来？

这里有几种实现方式。

## 方式 A：围绕目标坐标构造局部网格

如果目标是原始预测坐标 (g^*)，可以在它附近构造不同半径的候选点：

* 10 m；
* 100 m；
* 1 km；
* 10 km；
* 100 km；
* 1000 km。

然后把这些候选坐标格式化成 token 序列。

优点：简单稳定。
缺点：候选空间有限。

---

## 方式 B：beam search 获取模型自己可能输出的坐标

对原图或 masked image 做 constrained beam search，只允许输出合法坐标格式。得到 top-K 坐标序列：

[
\mathcal{B} = \text{BeamSearch}*{coord}(f*\theta,x)
]

然后对这些 beam candidates 算 Haversine 距离。

优点：候选都是模型真实可能输出的。
缺点：每次优化都 beam search 会贵。

---

## 方式 C：固定 geocell / coordinate lattice

把地球离散成网格或 geocell，每个格点中心坐标对应一条 token 序列。

例如：

* 国家级；
* 城市级；
* 0.1° 网格；
* S2 cell；
* H3 cell。

然后：

[
\mathcal{B}
]

就是这些 geocell center 的坐标 token 序列。

优点：全局覆盖。
缺点：候选数量可能很大，需要采样或 coarse-to-fine。

---

## 方式 D：混合方案

最推荐的是混合：

[
\mathcal{B}
===========

\mathcal{B}*{beam}
\cup
\mathcal{B}*{local}
\cup
\mathcal{B}_{negative}
]

包括：

* 模型原始预测附近坐标；
* ground truth 附近坐标；
* beam search 里模型高概率坐标；
* 随机远距离负样本；
* 同国家/不同城市样本；
* 不同国家样本。

这样 loss 既稳定，又能表达地理距离层级。

---

# 8. 这个 loss 怎么和实体 mask 连接？

你有候选实体：

[
\mathcal{C}={e_1,e_2,\dots,e_N}
]

每个实体有一个连续 gate：

[
z_i \in [0,1]
]

构造 masked image：

[
x_m
===

x \odot \left(1-\sum_i z_i e_i\right)
+
x_{blur} \odot \left(\sum_i z_i e_i\right)
]

直观地说：

* (z_i=0)：实体保留；
* (z_i=1)：实体被遮挡或模糊；
* 中间值：连续混合，便于求导。

然后把 (x_m) 输入冻结 VLM，得到坐标 token 概率，再计算 (\mathcal{R}_{geo})。

如果你的目标是找“最小破坏子集”，就优化：

[
\max_z
\mathcal{R}_{geo}(x_m)
----------------------

## \lambda_1 \sum_i z_i

\lambda_2 \sum_i z_i Area(e_i)
]

含义：

> 遮挡尽可能少的实体，但让模型地理预测尽可能远离原始位置或真实位置。

如果你的目标是找“最小保留子集”，就优化：

[
\min_z
\mathcal{R}*{geo}(x*{keep})
+
\lambda_1 \sum_i z_i
+
\lambda_2 \sum_i z_i Area(e_i)
]

含义：

> 只保留尽可能少的实体，但模型仍然能预测原位置。

这两个分别对应 necessity 和 sufficiency。

---

# 9. 你可以把方法理解成三层

## 第一层：实体 gate 层

决定遮挡哪些实体：

[
z_1,z_2,\dots,z_N
]

这是你真正要优化的变量。

---

## 第二层：VLM token 概率层

冻结 VLM，给每条合法坐标序列一个概率：

[
p_\theta(y^{(j)}|x_m)
]

这是梯度通道。

---

## 第三层：地理距离风险层

每条坐标序列有一个预计算 Haversine 距离：

[
D_j
]

最终：

[
\mathcal{R}_{geo}
=================

\sum_j p_\theta(y^{(j)}|x_m)D_j
]

这是地理语义。

三层连起来就是：

```text
实体 gate → masked image → VLM coordinate token probability → Haversine-weighted risk
```

---

# 10. 最容易误解的地方

你之前理解成：

> 每个 token 都有一个距离分数，然后和全词表概率向量点积。

这个是**单步 token-level risk**，不够严谨。

我建议的方法是：

> 每条合法坐标 token 序列有一个距离分数，然后和这条序列的概率相乘。

也就是：

[
\text{token-level}
\quad \rightarrow \quad
\text{sequence-level}
]

因为地理坐标必须作为完整序列才有意义。

---

# 11. 但有没有简化版可以先跑？

有。MVP 阶段可以先做简化版。

## 简化版 1：teacher-forced coordinate NLL + distance weighted negatives

选一批候选坐标序列：

* 原始预测坐标；
* ground truth；
* 近邻坐标；
* 远距离坐标。

对每条候选坐标序列计算 teacher-forced log probability：

[
\log p_\theta(y^{(j)}|x_m)
==========================

\sum_t \log p_\theta(y_t^{(j)}|x_m,y_{<t}^{(j)})
]

然后做：

[
\mathcal{R}_{geo}
=================

\sum_j
\text{softmax}(\log p_\theta(y^{(j)}|x_m))
D_j
]

这很好实现。

---

## 简化版 2：contrastive geodesic token loss

你不直接算所有候选期望，而是让模型远离原位置、靠近远位置，或者相反。

例如 deletion 目标：

[
\mathcal{L}_{del}
=================

## \log p_\theta(y_{near}|x_m)

\log p_\theta(y_{far}|x_m)
+
\lambda \mathcal{L}_{mask}
]

其中 (y_{near}) 是原始预测附近坐标，(y_{far}) 是远距离负样本。

这个没有完整期望风险优雅，但很容易跑通。

---

# 12. 最终一句话

我的 Haversine 方法不是“对单个数字 token 算距离”，而是：

> **把合法坐标输出看成一组 token 序列，每条序列对应一个真实坐标和一个 Haversine 距离；然后用冻结 VLM 对这些序列的概率加权这些距离，从而构造一个不断梯度的地理 token 风险，并把梯度回传到连续实体 mask。**

这就是完整思路。
