# 手写图归一化与消息传递

本项目对应工程验收任务：用 PyTorch 手写归一化、消息传递和节点更新，与 DGL 参考实现逐项核对。交付代码、测试和可复现说明，不生成预测 CSV。本地测试通过不等于导师已验收或已获得平台分数。

## 文件与运行

只有三个文件：graphconv.py 是可导入的 nn.Module，test_graphconv.py 是必要的验收测试，README.md 同时作为环境说明、实验报告和运行记录。不生成模型、缓存、额外日志或环境。

沿用已有环境 E:\dev\miniforge3\envs\pytorch；没有安装、升级依赖或新建环境。PowerShell 中运行：

```powershell
& 'E:\dev\miniforge3\envs\pytorch\python.exe' -B 'E:\dev\code\gamma培养计划\消息传递\test_graphconv.py'
```

运行条件：Python 3.11.16、PyTorch 2.1.2+cu121、DGL 2.2.1+cu121。实测设备为 CPU 和 NVIDIA GeForce RTX 4050 Laptop GPU。种子 42，CPU 线程数 1；CPU 使用 float64，CUDA 使用 float32。CUDA 聚合可能存在浮点累加顺序差异，不保证跨设备或运行逐位一致。

## 语义约定

- 输入是同构、有向、无权多重图；节点索引为 0 到 N-1。edges[0] 是源节点，edges[1] 是接收节点，不自动添加反向边。
- 重复的非自环边保留为多条消息，并计入度数。这样与 DGL 多重图语义一致，不能先去重再与未去重的参考图比较。
- self_loops=True 为默认设置：先移除已有全部自环，再为每节点添加恰好一个自环；归一化使用处理后的图。
- self_loops=False：完全保留原始边及原始自环数量。没有入边的节点聚合为零，输出只有偏置；偏置初始化为零。对应 DGL allow_zero_in_degree=True。
- “空图”同时覆盖 N>0、E=0 和 N=0、E=0。开启自环时，前者会转为每节点一个自环，后者仍为空。
- 不支持边权、异构图、二部图和额外批维度。接口聚焦当前任务，没有扩展这些功能。

## 公式与矩阵形状

令处理后的边为 j → i，A[i,j] 为该方向边的条数。对有向图，d_out(j) 为源节点出度，d_in(i) 为接收节点入度。使用 DGL GraphConv(norm="both") 的归一化：

```text
alpha(j,i) = 1 / sqrt(max(d_out(j),1) * max(d_in(i),1))
H' = D_in^(-1/2) A D_out^(-1/2) H W + b
```

图若为双向对称图，出入度相同；对一般有向图不能把两种度混用。clamp_min(1) 用来避免零度导致无穷值，不会凭空添加邻居。没有实际边时也没有该边的消息。

| 阶段 | 代码对应 | 形状 |
| --- | --- | --- |
| 节点特征 | x | N × F_in |
| 输入边索引 | edges | 2 × E |
| 自环处理后的边 | normalize 返回的 edges | 2 × E' |
| 归一化系数 | norm | E' |
| 节点线性变换 | hidden = x @ weight | N × F_out |
| 按源节点取消息并归一化 | hidden[src] * norm.unsqueeze(1) | E' × F_out |
| 按接收节点求和 | aggregated.index_add_(0, dst, messages) | N × F_out |
| 节点更新 | aggregated + bias | N × F_out |

W 的形状为 F_in × F_out，b 为 F_out。实现先投影再聚合，与线性聚合后乘 W 数学等价；没有建立 N×N 稠密邻接矩阵。层本身不内置激活函数，多层网络在层间调用 relu()，便于与 DGL 默认 activation=None 对齐。

GraphConv 是标准 nn.Module，支持 .to(device)、.double()、parameters()、state_dict() 和 PyTorch 自动求导。使用方法如下：

```python
import torch
from graphconv import GraphConv

x = torch.randn(3, 4)
edges = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
layer = GraphConv(4, 2)
outputs = layer(x, edges)  # [3, 2]
```

## 验收方法

参考层为 DGL GraphConv(norm="both", allow_zero_in_degree=True)，设置完全相同的权重与非零偏置。自环策略在两边独立应用；随机上游梯度相同，分别检查输入、权重、偏置的梯度，不仅检查最终输出。

| 测试 | 可核查内容 |
| --- | --- |
| test_reference_cases | 有向图和孤立点、无向图、自环、重复边、无边图、零节点图、单节点图；两种自环策略；1→1、2→5、5→2 维度；CPU/GPU |
| test_known_values | 小图手算结果、重复边度归一化、自环唯一性；避免两个实现共同犯错仍通过 |
| test_gradcheck | CPU 双精度下以数值差分核对输入梯度 |
| test_invalid_input | 越界/负索引、错误边类型或形状、错误特征形状、NaN 和无穷值必须明确失败 |
| test_existing_graph | 直接读取已有 GNN 入门/nodes.csv 和 edges.csv，240 节点、888 边，检查输出及梯度，不读取标签 |
| test_two_layers | 同样初始参数的两层网络 4→8→3，ReLU、交叉熵、SGD；连续三步输出及参数更新与 DGL 一致 |

参考对比共 85 组：7 种图 × 2 种自环策略 × 3 种维度 × 2 种设备，加 1 组已有图数据。CPU 容差为 atol=1e-10、rtol=1e-9；GPU 容差为 atol=2e-5、rtol=2e-4。

两层集成测试只用前三个节点计算监督损失，后两个节点不参与监督。没有进行真实标签训练或调参，因此没有训练/验证分类划分、准确率或隐藏测试成绩可报告。

## 实测结果与限制

2026-11-17，本机运行全部 6 个测试方法通过。关键输出记录：

```text
Python 3.11.16 PyTorch 2.1.2+cu121 DGL 2.2.1+cu121
CUDA True NVIDIA GeForce RTX 4050 Laptop GPU
数据复用：240 节点，888 边；不读取标签。
test_existing_graph ... ok
test_gradcheck ... ok
test_invalid_input ... ok
test_known_values ... ok
test_reference_cases ... ok
参考对比 85 组；最大输出绝对误差 2.384e-07；最大梯度绝对误差 2.384e-07
test_two_layers ... ok
Ran 6 tests
OK
```

上述记录省略 unittest 模块全名和运行耗时；执行命令会重新输出完整结果，不需要额外日志文件。误差为全部参考对比中观测到的最大绝对误差，不是对任意输入的理论误差上界。

CPU/GPU 的精度不同，这里仅报告正确性，不作性能快慢结论。没有用这一实现覆盖旧的节点分类项目；在新任务中使用该层时，应确认归一化、自环、方向与目标模型一致，不能因为名字都叫 GCN 就直接混用。

## 数据、来源与提交

已有图数据来自平台原创教学合成数据，复用原 GNN 入门项目的 nodes.csv 和 edges.csv；本仓库保留相邻目录关系及原始数据内容，供验收复现。当前材料未给出明确的数据再分发许可，不声称拥有额外的数据授权，也未对数据另行附加许可证。测试图由代码和种子 42 生成，不依赖网络下载。

本实现由 Codex 辅助编写与验证。参考 [DGL 2.2.1 GraphConv 官方文档](https://www.dgl.ai/dgl_docs/en/2.2.x/generated/dgl.nn.pytorch.conv.GraphConv.html) 及本机已安装的 dgl/nn/pytorch/conv/graphconv.py；手写核心仅依赖 PyTorch，DGL 仅用于测试对照。

平台需提交代码链接、环境说明和实验报告，可用本 README 作为报告。代码仓库为 https://github.com/baiyang431/gamma-message-passing ，验收时可填写对应 PR 的实际链接。提交号以 GitHub 仓库历史为准；没有将工程材料自动提交到题目平台，也不声称已经过导师验收。
