from dataclasses import dataclass, field
from typing import Any

from cupyx.scipy.sparse import csr_matrix
from scipy.sparse import lil_array
import cupy as cp
import numpy as np


@dataclass
class TreeFormSequentialDecisionProcess:
    """
    树形式序列决策过程 (Tree-Form Sequential Decision Process, TFSDP) 类。
    
    核心功能:
    将博弈树结构转换为适合 GPU 并行计算的矩阵和向量形式。
    这对应于论文第 3.1 节 "IMPLEMENTATION - SETUP" 中描述的内容。
    它不仅存储树的结构（节点、边），还处理策略从行为形式到序列形式的转换，以及反事实效用的计算。

    主要理论对应:
    该类实现了在树结构上进行动态规划 (Dynamic Programming) 的基础，通过邻接矩阵 (Adjacency Matrix) 表示博弈树 $G \in \mathbb{R}^{\mathbb{V}^2}$。
    这使得计算期望效面 (Eq. 1) 和到达概率 (Eq. 2, Eq. 5) 可以转化为矩阵向量运算。
    """
    # 原始的 CPU 端博弈树对象 (通常来自 noregret 库)，包含节点、边、信息集等拓扑结构。
    tree_form_sequential_decision_process: Any
    
    # 层级遍历的节点索引列表 (List[cupy.ndarray])。
    # level_sources[k] 存储第 k 层的所有节点索引。
    # 用于在 GPU 上进行并行化的层级运算（如反向归纳）。
    level_sources: Any = field(init=False, default_factory=list)
    
    # 层级遍历的序列 (Sequence) 索引列表 (List[cupy.ndarray])。
    # 对应于 level_sources 中决策节点产生的动作序列索引。
    level_sequences: Any = field(init=False, default_factory=list)
    
    # 层级遍历的父序列索引列表 (List[cupy.ndarray])。
    # 用于从父序列概率计算当前序列概率 (应用链式法则: pi(h) = pi(parent) * sigma(h))。
    level_parent_sequences: Any = field(init=False, default_factory=list)
    
    # 博弈树的邻接矩阵 (Sparse Matrix, N_nodes x N_nodes)。
    # graph[u, v] = 1 (或者策略概率) 表示从节点 u 到 v 的转换。
    # 在计算时，会填入策略概率。
    graph: Any = field(init=False)
    
    # 节点到序列的映射矩阵 (Sparse Matrix, N_nodes x N_sequences)。
    # graph2[u, seq_id] = 1 表示节点 u 执行了对应 seq_id 的动作。
    # 用于将节点上的计算关联到具体的序列（动作）。
    graph2: Any = field(init=False)
    
    # 行为策略更新索引 (Tuple[cupy.ndarray, cupy.ndarray])。
    # 存储了 (source_nodes, target_nodes) 的索引对。
    # 用于快速将当前的 behavior strategy 填入 self.graph 矩阵中。
    behavioral: Any = field(init=False)
    
    # 行为策略更新索引2 (Tuple[cupy.ndarray, cupy.ndarray])。
    # 存储了 (source_nodes, sequence_indices) 的索引对。
    # 用于快速将 behavior strategy 填入 self.graph2 矩阵中。
    behavioral2: Any = field(init=False)
    
    # 反事实效用计算索引 (cupy.ndarray)。
    # 存储了子节点的索引，用于在计算 counterfactual utility 时快速查找后继节点的值。
    counterfactual: Any = field(init=False)

    def __post_init__(self):
        """
        初始化 TFSDP 的矩阵表示。
        
        功能:
        1. 遍历博弈树，构建层级结构 (level_sources, level_sequences 等)。
        2. 构建邻接矩阵 graph 和 graph2，用于后续的矩阵乘法运算。
        3. 初始化行为策略和序列索引的映射。
        
        这部分逻辑是为了支持论文 3.1 节提到的为了高效并行化 CFR，避免递归树遍历，而是使用稠密/稀疏矩阵操作。
        """
        # 确保根序列是空的元组 ()
        assert self.sequences[0] == ()

        # 初始化 BFS 队列，从根节点开始
        queue = [self.nodes[0]]

        # 开始层级遍历 (BFS)
        while queue:
            sources = []          # 当前层的节点索引
            sequences = []        # 当前层产生的序列索引
            parent_sequences = [] # 当前层序列对应的父序列索引
            next_queue = []       # 下一层的节点

            for p in queue:
                # 记录当前节点索引
                sources.append(self.nodes.index(p))

                match self.node_types[p]:
                    case self.NodeType.DECISION_POINT:
                        # 如果是决策节点
                        # 1. 找到该节点的父序列索引
                        parent_sequence = self.sequences.index(
                            self.parent_sequences[p],
                        )

                        # 2. 遍历该节点的所有动作
                        for a in self.actions[p]:
                            # 记录 (节点, 动作) 构成的序列索引
                            sequences.append(self.sequences.index((p, a)))
                            # 记录父序列索引
                            parent_sequences.append(parent_sequence)
                            # 将子节点加入下一层队列
                            next_queue.append(self.transitions[p, a])
                    case self.NodeType.OBSERVATION_POINT:
                        # 如果是观察节点 (或机会节点 Chance Node)
                        # 仅遍历其信号/结果，将子节点加入队列
                        for s in self.signals[p]:
                            next_queue.append(self.transitions[p, s])

            # 将收集到的当前层数据转换为 GPU 数组 (CuPy array) 并存储
            self.level_sources.append(cp.array(sources, np.long))
            self.level_sequences.append(cp.array(sequences, np.long))
            self.level_parent_sequences.append(
                cp.array(parent_sequences, np.long),
            )

            # 移动到下一层
            queue = next_queue

        # 初始化稀疏矩阵构建器 (LIL format 适合增量构建)
        self.graph = lil_array((len(self.nodes), len(self.nodes)))
        self.graph2 = lil_array((len(self.nodes), len(self.sequences)))

        # 再次遍历所有节点，构建邻接关系
        for source, p in enumerate(self.nodes):
            match self.node_types[p]:
                case self.NodeType.DECISION_POINT:
                    # 对于决策节点
                    for a in self.actions[p]:
                        sequence = self.sequences.index((p, a))
                        # 标记 graph2: 节点 source 关联到 sequence
                        self.graph2[source, sequence] = 1

                    events = self.actions
                case self.NodeType.OBSERVATION_POINT:
                    events = self.signals

            # 对于所有类型的节点，构建 graph: 节点 source 到 target 的连接
            for e in events[p]:
                target = self.nodes.index(self.transitions[p, e])
                self.graph[source, target] = 1

        # 转换为 CSR 格式，适合高效的矩阵向量乘法
        self.graph = csr_matrix(self.graph)
        self.graph2 = csr_matrix(self.graph2)
        
        # 准备构建快速更新策略所需的索引映射
        sources = []
        targets = []
        sequences = []

        # 遍历所有序列 (除了根序列)
        for sequence, j_a in enumerate(self.sequences[1:]):
            j, _ = j_a
            source = self.nodes.index(j)
            target = self.nodes.index(self.transitions[j_a])
            sequence += 1 # 偏移索引，因为跳过了根序列

            sources.append(source)
            targets.append(target)
            sequences.append(sequence)

        # behavioral: 存储 (源节点, 目标节点) 索引
        # 用于在 self.graph[sources, targets] = strategy 这种操作中快速赋值
        self.behavioral = (
            cp.array(sources, np.long),
            cp.array(targets, np.long),
        )
        
        # behavioral2: 存储 (源节点, 序列索引) 索引
        # 用于在 self.graph2[sources, sequences] = strategy 中快速赋值
        self.behavioral2 = (
            cp.array(sources, np.long),
            cp.array(sequences, np.long),
        )
        
        # counterfactual: 存储所有非根序列对应的转换后节点索引
        # 用于根据序列找到其指向的下一个节点，方便计算反事实值
        self.counterfactual = cp.array(
            list(
                map(
                    self.nodes.index,
                    map(self.transitions.get, self.sequences[1:]),
                ),
            ),
        )

    def __getattr__(self, name):
        return getattr(self.tree_form_sequential_decision_process, name)

    def behavioral_uniform_strategy(self):
        """
        生成均匀行为策略。
        
        功能:
        为每个信息集生成一个均匀分布的策略，即每个动作的概率为 1/|A(h)|。
        这是 CFR 算法初始迭代或需要重置策略时常用的起点。
        
        对应公式:
        参考 Eq. 9 中的一种边界情况：当总后悔值为 0 时，策略通过均匀分布给出。
        """
        strategy = [None] * (len(self.sequences) - 1)

        for sequence, j_a in enumerate(self.sequences[1:]):
            j, _ = j_a
            strategy[sequence] = 1 / len(self.actions[j])

        return cp.array(strategy)

    def behavioral_best_response(self, utility):
        """
        计算针对给定效用的最佳响应 (Best Response)。
        
        功能:
        使用反向归纳法 (Backward Induction) 在树上计算每个节点的最大期望效用 V。
        对于决策节点 (Decision Point)，选择期望效用最大的动作。
        对于观察节点 (Observation Point)/机会节点，计算期望值。
        
        这通常用于计算 Exploitability (可利用度)，即纳什均衡的近似程度。
        """
        V = [0] * len(self.nodes)
        node = len(self.nodes)

        for p in reversed(self.nodes):
            node -= 1

            match self.node_types[p]:
                case self.NodeType.DECISION_POINT:
                    V[node] = max(
                        (
                            utility[self.sequences.index((p, a))]
                            + V[self.nodes.index(self.transitions[p, a])]
                        )
                        for a in self.actions[p]
                    )
                case self.NodeType.OBSERVATION_POINT:
                    V[node] = sum(
                        V[self.nodes.index(self.transitions[p, s])]
                        for s in self.signals[p]
                    )

        return NotImplemented, V[0]

    def sequence_form_best_response(self, utility):
        _, value = self.behavioral_best_response(utility)

        return NotImplemented, value

    def behavioral_to_sequence_form(self, behavioral_strategy):
        strategy = cp.empty(len(self.sequences))
        strategy[0] = 1
        strategy[1:] = behavioral_strategy

        for sequences, parent_sequences in zip(
                self.level_sequences,
                self.level_parent_sequences,
        ):
            strategy[sequences] *= strategy[parent_sequences]

        return strategy

    def counterfactual_utilities(self, behavioral_strategy, utility):
        graph = self.graph.copy()
        graph[self.behavioral] = behavioral_strategy
        graph2 = self.graph2.copy()
        graph2[self.behavioral2] = behavioral_strategy
        V = cp.zeros(len(self.nodes))

        for sources in reversed(self.level_sources):
            V[sources] = graph[sources] @ V + graph2[sources] @ utility

        return utility[1:] + V[self.counterfactual]
