# 手写图归一化与消息传递

使用 PyTorch 手写图卷积，并与 DGL `GraphConv(norm="both")` 对照验证。覆盖归一化、消息传递、节点更新、边界行为及自动求导。

## 项目结构

```text
消息传递/
  graphconv.py
  test_graphconv.py
  README.md
GNN 入门/
  nodes.csv
  edges.csv
```

完整公式、矩阵形状、实验环境及验收报告见 [消息传递/README.md](消息传递/README.md)。两份图数据直接供测试读取，保留原项目相对目录关系。没有上传环境、训练标签或预测结果。

## 复现

已验证环境：Python 3.11.16、PyTorch 2.1.2+cu121、DGL 2.2.1+cu121；GPU 为 NVIDIA GeForce RTX 4050 Laptop GPU。

在仓库根目录，使用已有的兼容环境执行：

```powershell
python -B '.\消息传递\test_graphconv.py'
```

本机使用的具体解释器：

```powershell
& 'E:\dev\miniforge3\envs\pytorch\python.exe' -B '.\消息传递\test_graphconv.py'
```

随机种子为 42。CPU 使用 float64，CUDA 可用时额外执行 float32 对照；无 GPU 时只运行 CPU 测试，因此参考对比数量不同。CPU 与 GPU 同时启用时，全部 6 个测试方法、85 组参考对比通过，最大输出与梯度绝对误差均约为 `2.384e-7`。测试输出实际数量与误差；不作性能快慢结论。

## 数据与工具

图数据来自平台原创教学合成数据，用于本次工程验收；来源及许可信息的已知范围详见实验报告。本实现由 Codex 辅助编写与验证。参考 [DGL 2.2.1 GraphConv 官方文档](https://www.dgl.ai/dgl_docs/en/2.2.x/generated/dgl.nn.pytorch.conv.GraphConv.html)。
