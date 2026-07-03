下面是最终版加权规则，可以直接作为方法部分的核心定义。

GeoTF 最终加权规则：WGS84-aware Structured Coordinate Weighting

模型先自然输出坐标，例如：

y^* = \text{``-35.1510, +128.9830''}

然后对这个原始输出做 teacher forcing。非坐标内容允许存在，但权重为 0。

最终 loss：

\mathcal{L}_{GeoTF}(I_m, y^*)
=
-\sum_t w_t \log p(y_t^* \mid I_m, y_{<t}^*)

其中 I_m 是遮挡后的图像，y^* 是原图上模型自然输出的参考坐标。

⸻

1. 先解析坐标字符

对模型输出 y^* 提取第一个合法经纬度：

(\phi, \lambda)

其中：

\phi = \text{latitude}

\lambda = \text{longitude}

例如：

The location is probably -35.1510, +128.9830.

只解析：

-35.1510, +128.9830

其他文字权重全部为 0。

⸻

2. 字符分为五类

A. 非坐标文本

例如：

The location is probably

权重：

w_t = 0

⸻

B. 分隔符

包括：

空格、逗号、括号、冒号、单位符号

权重：

w_t = 0

注意：逗号权重为 0，因为它只分隔纬度和经度，不表达数值量级。

⸻

C. 符号位

包括：

+ 或 -

符号位参与加权，因为：

-120 \neq +120

符号位代表半球翻转，是最高层级地理判断。

⸻

D. 小数点

包括：

.

小数点参与加权，因为它决定整数度和小数度的边界，是量级结构的一部分。

⸻

E. 坐标数字

包括纬度和经度中的所有数字：

0,1,2,3,4,5,6,7,8,9

数字位根据其十进制度位次赋权。

⸻

3. WGS84 局部尺度

使用 WGS84 椭球，而不是简单 111\text{ km}。

WGS84 参数：

a = 6378137

f = \frac{1}{298.257223563}

e^2 = f(2-f)

设纬度为弧度制：

\phi_r = \phi \cdot \frac{\pi}{180}

子午圈曲率半径：

M(\phi)
=
\frac{a(1-e^2)}
{(1-e^2\sin^2\phi_r)^{3/2}}

卯酉圈曲率半径：

N(\phi)
=
\frac{a}
{\sqrt{1-e^2\sin^2\phi_r}}

纬度 1 度对应地表距离：

D_{lat}(\phi)
=
\frac{\pi}{180}M(\phi)

经度 1 度对应地表距离：

D_{lon}(\phi)
=
\frac{\pi}{180}N(\phi)\cos\phi_r

其中 D_{lat} 和 D_{lon} 单位为米。

⸻

4. 数字位权重

对每个坐标数字字符，先确定它代表的十进制度位次 k_t。

例如纬度：

-35.1510

字符	位次	k_t
3	十位度	1
5	个位度	0
1	小数第 1 位	-1
5	小数第 2 位	-2
1	小数第 3 位	-3
0	小数第 4 位	-4

经度：

+128.9830

字符	位次	k_t
1	百位度	2
2	十位度	1
8	个位度	0
9	小数第 1 位	-1
8	小数第 2 位	-2
3	小数第 3 位	-3
0	小数第 4 位	-4

数字位的物理尺度：

纬度数字：

s_t = D_{lat}(\phi) \cdot 10^{k_t}

经度数字：

s_t = D_{lon}(\phi) \cdot 10^{k_t}

⸻

5. 小数点权重

小数点是 radix boundary，表示整数度和小数度的分界。

因此小数点的物理尺度设为对应坐标轴的 1^\circ 距离。

纬度小数点：

s_t = D_{lat}(\phi)

经度小数点：

s_t = D_{lon}(\phi)

也就是相当于 k_t = 0 的尺度。

⸻

6. 符号位权重

符号位表示半球翻转，需要参与，但必须做上限限制，避免压死所有数字位。

纬度符号

从 \phi 变成 -\phi，近似纬向位移：

s_{\text{lat-sign}}
=
2|\phi| \cdot D_{lat}(\phi)

经度符号

从 \lambda 变成 -\lambda，经度差取短弧：

\Delta \lambda_{\text{sign}}
=
\min(2|\lambda|,\ 360 - 2|\lambda|)

经度符号尺度：

s_{\text{lon-sign}}
=
\Delta \lambda_{\text{sign}} \cdot D_{lon}(\phi)

然后符号位做 cap：

s_{\text{sign}}
=
\min(s_{\text{sign}},\ s_{\text{cap}})

推荐：

s_{\text{cap}}
=
\max_{j \in \text{integer digits}} s_j

也就是说，符号位最多和最高整数位同量级，不能无限支配 loss。

⸻

7. log 压缩

所有参与项先得到物理尺度 s_t，然后统一做 log 压缩：

\tilde{w}_t = \log(1+s_t)

其中 s_t 单位是米。

这样可以保留物理量级差异，但避免高位权重过度支配低位。

⸻

8. 均值归一化

令 \mathcal{C} 是所有参与坐标权重的字符集合：

\mathcal{C}
=
\{
\text{sign}, \text{dot}, \text{coordinate digits}
\}

不包括：

\text{comma}, \text{space}, \text{text}

最终权重：

w_t =
\begin{cases}
0, & t \notin \mathcal{C} \\
\frac{\tilde{w}_t}
{\frac{1}{|\mathcal{C}|}\sum_{j\in \mathcal{C}}\tilde{w}_j}, & t \in \mathcal{C}
\end{cases}

这样所有有效坐标字符的平均权重为 1，不会改变整体 loss 尺度。

⸻

9. tokenizer 聚合规则

如果 tokenizer 是字符级，直接使用字符权重。

如果一个 token 覆盖多个字符，例如：

"128"

则 token 权重取字符权重的平均：

w_{\text{token}}
=
\frac{1}{|A(t)|}
\sum_{c \in A(t)} w_c

其中 A(t) 是该 token 覆盖的字符集合。

如果 token 混合了数字和小数点：

"128."

也正常平均。

如果 token 是非坐标文本：

"location"

则：

w_{\text{token}} = 0

⸻

10. 最终规则一句话

最终方法是：

模型自然输出坐标；提取其中合法经纬度；非坐标文本、空格、逗号权重为 0；坐标数字按 WGS84 下对应十进制度位次的地表距离赋权；经度使用 \cos\phi 修正；小数点按 1^\circ radix boundary 赋权；正负号按半球翻转位移赋权但 capped 到最高整数位尺度；所有有效项经过 \log(1+s) 压缩并均值归一化；token 权重由字符权重聚合得到。

最终公式：

\boxed{
\mathcal{L}_{GeoTF}(I_m, y^*)
=
-\sum_t
w_t^{WGS84}
\log p(y_t^* \mid I_m, y_{<t}^*)
}

其中：

\boxed{
w_t^{WGS84}
=
\frac{
\log(1+s_t)
}{
\text{mean}_{j\in\mathcal{C}}\log(1+s_j)
}
}

非坐标项：

\boxed{
w_t = 0
}

这就是最终版 WGS84-aware structured coordinate weighting。
