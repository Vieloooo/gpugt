from dataclasses import dataclass, field, KW_ONLY
from typing import Any

from cupyx.scipy.sparse import csr_matrix
from noregret.regret_minimizers import SequenceFormPolytopeRegretMinimizer
from scipy.sparse import lil_array
import cupy as cp


@dataclass
class CounterfactualRegretMinimization(SequenceFormPolytopeRegretMinimizer):
    """
    反事实后悔最小化 (Counterfactual Regret Minimization, CFR) 算法实现。
    
    核心功能:
    在博弈的每次迭代中，根据累积的“反事实后悔”值来更新策略。
    通过大量的自我博弈迭代，平均策略将收敛到纳什均衡 (Nash Equilibrium)。
    
    对应理论:
    实现 doc_gpupt.md 中描述的 CFR 核心逻辑。
    - 维护累积后悔 Eq. 8: $\tilde{r}^{(T)}$
    - 策略更新 Eq. 9: Regret Matching
    """
    tree_form_sequential_decision_process: Any
    _: KW_ONLY
    dimension: Any = field(init=False)
    is_time_symmetric: Any = True
    mask: Any = field(init=False)
    counterfactual_regrets: Any = field(init=False)
    behavioral_uniform_strategy: Any = field(init=False)
    behavioral_strategy: Any = field(init=False)

    def __post_init__(self):
        """
        初始化 CFR 的状态。
        
        功能:
        1. 初始化 mask 矩阵，用于将每个节点对应的动作映射回信息集。
        2. 初始化累积后悔值 counterfactual_regrets (对应 Eq. 8 的分子部分)。
        3. 初始化策略为均匀策略。
        """
        tfsdp = self.tree_form_sequential_decision_process
        self.dimension = len(tfsdp.sequences)

        super().__post_init__()

        self.previous_strategy = cp.array(self.previous_strategy)
        self.average_strategy = cp.array(self.average_strategy)
        self.previous_utility = cp.array(self.previous_utility)
        self.cumulative_utility = cp.array(self.cumulative_utility)

        self.mask = lil_array(
            (len(tfsdp.decision_points), len(tfsdp.sequences) - 1),
        )

        for node, j in enumerate(tfsdp.decision_points):
            for a in tfsdp.actions[j]:
                sequence = tfsdp.sequences.index((j, a)) - 1
                self.mask[node, sequence] = 1

        self.mask = csr_matrix(self.mask)
        self.counterfactual_regrets = cp.zeros(len(tfsdp.sequences) - 1)
        self.behavioral_uniform_strategy = tfsdp.behavioral_uniform_strategy()
        self.behavioral_strategy = self.behavioral_uniform_strategy.copy()

    @property
    def _floored_counterfactual_regrets(self):
        return self.counterfactual_regrets.clip(0)

    def next_strategy(self, prediction=False):
        """
        生成下一次迭代的策略 (Regret Matching)。
        
        功能:
        根据累积的正反事实后悔值，按比例分配概率给各个动作。
        如果某个信息集的所有动作后悔值都非正，则使用均匀策略。
        
        对应公式:
        Eq. 9 (Strategy Profile for Iteration T+1):
        $\sigma^{(T+1)}(h, a) = \frac{R^{(T)}(h,a)^+}{\sum R^{(T)}(h,a')^+}$
        
        代码对应:
        numerator = self._floored_counterfactual_regrets  (即 R^+)
        denominator = self.mask.T @ (self.mask @ numerator) (即 sum(R^+))
        numerator / normalized_denominator (归一化得到概率)
        """
        if prediction is not False:
            raise NotImplementedError

        numerator = self._floored_counterfactual_regrets
        denominator = self.mask.T @ (self.mask @ numerator)
        normalized_denominator = denominator.copy()
        normalized_denominator[normalized_denominator == 0] = 1
        """
        观察效用并更新反事实后悔值。
        
        功能:
        1. 接收从博弈树计算回传的效用 (utility)。
        2. 计算反事实效用 (counterfactual utilities)。
        3. 计算瞬时后悔 (instantaneous regret) 并累加到累积后悔中。
        
        对应公式:
        Eq. 7 (Instantaneous Counterfactual Regret):
        $\tilde{r}(\sigma, h, a) = \tilde{u}(\sigma|_{h \to a}, h) - \tilde{u}(\sigma, h)$
        
        代码对应:
        counterfactual_utilities (即 \tilde{u}(\sigma|_{h \to a}))
        self.mask.T @ ... (计算当前策略下的期望效用 \tilde{u}(\sigma, h))
        两者相减即为后悔值，然后累加 (+=)。
        """
        self.behavioral_strategy = cp.where(
            denominator == 0,
            self.behavioral_uniform_strategy,
            numerator / normalized_denominator,
        )
        strategy = (
            self
            .tree_form_sequential_decision_process
            .behavioral_to_sequence_form(self.behavioral_strategy)
        )

        self.strategies.append(strategy)

        return strategy

    """
    CFR+ 算法实现。
    
    核心功能:
    CFR 的一个变体，通常收敛速度更快。
    主要区别在于对累积后悔值的处理：
    CFR+ 将累积后悔值与 0 取最大值 (max(0, R))，不允许后悔值长期为负。
    也就是论文中提到的 "CFR+ by Tammelin (2014) ... eliminates the averaging step while improving the convergence rate".
    (注：这里的实现保留了 observe_utility 中的 clip(0) 操作)
    """
    def observe_utility(self, utility):
        super().observe_utility(utility)

        counterfactual_utilities = (
            self
            .tree_form_sequential_decision_process
            .counterfactual_utilities(self.behavioral_strategy, utility)
        )
        self.counterfactual_regrets += (
            counterfactual_utilities
            - (
                self.mask.T
                @ (
                    self.mask
                    @ (self.behavioral_strategy * counterfactual_utilities)
                )
            )
        )


@dataclass
class CounterfactualRegretMinimizationPlus(CounterfactualRegretMinimization):
    _: KW_ONLY
    gamma: Any = 1

    @property
    def _floored_counterfactual_regrets(self):
        return self.counterfactual_regrets

    def observe_utility(self, utility):
        super().observe_utility(utility)

        self.counterfactual_regrets = self.counterfactual_regrets.clip(0)
