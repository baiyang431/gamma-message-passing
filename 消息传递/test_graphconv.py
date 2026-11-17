"""核对手写图卷积的输出、梯度、边界行为与 DGL 集成。"""

import csv
import platform
import unittest
from pathlib import Path

import dgl
import torch
from dgl.nn import GraphConv as DGLGraphConv

from graphconv import GraphConv, normalize

ROOT = Path(__file__).resolve().parent


class GraphConvTest(unittest.TestCase):
    max_output_error = 0.0
    max_grad_error = 0.0
    comparisons = 0

    def compare(self, x, edges, out_feats, self_loops):
        manual = GraphConv(x.shape[1], out_feats, self_loops).to(x)
        reference = DGLGraphConv(
            x.shape[1], out_feats, norm="both", allow_zero_in_degree=True
        ).to(x)
        with torch.no_grad():
            manual.bias.uniform_(-0.2, 0.2)
            reference.weight.copy_(manual.weight)
            reference.bias.copy_(manual.bias)
        graph = dgl.graph((edges[0], edges[1]), num_nodes=len(x), device=x.device)
        if self_loops:
            graph = dgl.add_self_loop(dgl.remove_self_loop(graph))
        a = x.detach().clone().requires_grad_()
        b = x.detach().clone().requires_grad_()
        actual, expected = manual(a, edges), reference(graph, b)
        self.assertEqual(actual.shape, (len(x), out_feats))
        atol, rtol = (1e-10, 1e-9) if x.dtype == torch.float64 else (2e-5, 2e-4)
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        error = (actual - expected).abs().max().item() if actual.numel() else 0.0
        type(self).max_output_error = max(type(self).max_output_error, error)
        # 使用相同非均匀上游梯度，避免求和损失掩盖节点顺序错误。
        upstream = torch.randn_like(actual)
        grads_a = torch.autograd.grad(actual, (a, manual.weight, manual.bias), upstream)
        grads_b = torch.autograd.grad(expected, (b, reference.weight, reference.bias), upstream)
        for ga, gb in zip(grads_a, grads_b):
            torch.testing.assert_close(ga, gb, atol=atol, rtol=rtol)
            error = (ga - gb).abs().max().item() if ga.numel() else 0.0
            type(self).max_grad_error = max(type(self).max_grad_error, error)
        type(self).comparisons += 1

    def test_reference_cases(self):
        cases = {
            "directed_isolated": (5, [[0, 0, 1, 3], [1, 2, 2, 0]]),
            "undirected": (4, [[0, 1, 1, 2], [1, 0, 2, 1]]),
            "self_loops": (3, [[0, 0, 1, 2], [0, 1, 1, 2]]),
            "duplicates": (4, [[0, 0, 0, 1, 2, 2], [1, 1, 0, 2, 2, 2]]),
            "no_edges": (3, [[], []]),
            "no_nodes": (0, [[], []]),
            "one_node": (1, [[0, 0], [0, 0]]),
        }
        devices = [("cpu", torch.float64)]
        if torch.cuda.is_available():
            devices.append(("cuda", torch.float32))
        for device, dtype in devices:
            for name, (n, pairs) in cases.items():
                for loops in [False, True]:
                    for in_feats, out_feats in [(1, 1), (2, 5), (5, 2)]:
                        with self.subTest(device=device, case=name, loops=loops,
                                          shape=(in_feats, out_feats)):
                            edges = torch.tensor(pairs, dtype=torch.long, device=device)
                            x = torch.randn(n, in_feats, dtype=dtype, device=device)
                            self.compare(x, edges, out_feats, loops)

    def test_known_values(self):
        layer = GraphConv(1, 1, self_loops=False).double()
        with torch.no_grad():
            layer.weight.fill_(2)
            layer.bias.fill_(0.5)
        edges = torch.tensor([[0, 0, 1], [1, 1, 2]])
        x = torch.tensor([[3.], [4.], [5.], [8.]], dtype=torch.float64)
        torch.testing.assert_close(layer(x, edges), torch.tensor(
            [[0.5], [6.5], [8.5], [0.5]], dtype=torch.float64))
        clean, norm = normalize(edges, 4, torch.float64, False)
        self.assertTrue(torch.equal(clean, edges))
        torch.testing.assert_close(norm, torch.tensor([.5, .5, 1.], dtype=torch.float64))
        loops = torch.tensor([[0, 0, 0, 1], [0, 0, 1, 1]])
        result, _ = normalize(loops, 3, torch.float64, True)
        self.assertEqual((result[0] == result[1]).sum().item(), 3)

    def test_gradcheck(self):
        layer = GraphConv(2, 3).double()
        x = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
        edges = torch.tensor([[0, 0, 1], [1, 1, 2]])
        self.assertTrue(torch.autograd.gradcheck(lambda values: layer(values, edges), (x,)))

    def test_invalid_input(self):
        layer = GraphConv(2, 3)
        x = torch.randn(3, 2)
        for edges in [torch.tensor([[0], [3]]), torch.tensor([[-1], [0]]),
                      torch.zeros(2, 1), torch.zeros(3, 1, dtype=torch.long)]:
            with self.subTest(edges=edges):
                with self.assertRaises(ValueError):
                    layer(x, edges)
        edges = torch.empty(2, 0, dtype=torch.long)
        for values in [torch.randn(3, 4), torch.randn(2),
                       torch.full((3, 2), float("nan")), torch.full((3, 2), float("inf"))]:
            with self.subTest(shape=values.shape):
                with self.assertRaises(ValueError):
                    layer(values, edges)

    def test_existing_graph(self):
        data = ROOT.parent / "GNN 入门"
        with (data / "nodes.csv").open(encoding="utf-8-sig", newline="") as f:
            nodes = list(csv.DictReader(f))
        with (data / "edges.csv").open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        indices = {row["id"]: i for i, row in enumerate(nodes)}
        x = torch.tensor([[float(row[k]) for k in ["f1", "f2", "f3", "f4"]]
                          for row in nodes], dtype=torch.float64)
        edges = torch.tensor([[indices[row["source"]], indices[row["target"]]]
                              for row in rows]).t().contiguous()
        self.compare(x, edges, 3, True)
        print(f"数据复用：{len(nodes)} 节点，{len(rows)} 边；不读取标签。", flush=True)

    def test_two_layers(self):
        x = torch.randn(5, 4, dtype=torch.float64)
        edges = torch.tensor([[0, 1, 2, 2], [1, 2, 0, 3]])
        a = torch.nn.ModuleList([GraphConv(4, 8), GraphConv(8, 3)]).double()
        b = torch.nn.ModuleList([
            DGLGraphConv(4, 8, allow_zero_in_degree=True),
            DGLGraphConv(8, 3, allow_zero_in_degree=True),
        ]).double()
        b.load_state_dict(a.state_dict())
        graph = dgl.add_self_loop(dgl.graph((edges[0], edges[1]), num_nodes=5))
        optimizer_a = torch.optim.SGD(a.parameters(), lr=0.01)
        optimizer_b = torch.optim.SGD(b.parameters(), lr=0.01)
        target = torch.tensor([0, 1, 2, 0, 1])
        for _ in range(3):
            optimizer_a.zero_grad()
            optimizer_b.zero_grad()
            out_a = a[1](a[0](x, edges).relu(), edges)
            out_b = b[1](graph, b[0](graph, x).relu())
            torch.testing.assert_close(out_a, out_b, atol=1e-10, rtol=1e-9)
            torch.nn.functional.cross_entropy(out_a[:3], target[:3]).backward()
            torch.nn.functional.cross_entropy(out_b[:3], target[:3]).backward()
            optimizer_a.step()
            optimizer_b.step()
            for pa, pb in zip(a.parameters(), b.parameters()):
                torch.testing.assert_close(pa, pb, atol=1e-10, rtol=1e-9)

    @classmethod
    def tearDownClass(cls):
        print(f"参考对比 {cls.comparisons} 组；最大输出绝对误差 {cls.max_output_error:.3e}；"
              f"最大梯度绝对误差 {cls.max_grad_error:.3e}", flush=True)


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.manual_seed(42)
    print(f"Python {platform.python_version()}；PyTorch {torch.__version__}；DGL {dgl.__version__}",
          flush=True)
    print(f"CUDA：{torch.cuda.is_available()}；种子：42", flush=True)
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0), flush=True)
    unittest.main(verbosity=2)
