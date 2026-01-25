from dataclasses import dataclass

from cupyx.scipy.sparse import csr_matrix
from noregret.games import (
    ExtensiveFormGame,
    TwoPlayerExtensiveFormGame,
    TwoPlayerZeroSumExtensiveFormGame,
)

from gpugt.utilities import TreeFormSequentialDecisionProcess


@dataclass
class ExtensiveFormGame(ExtensiveFormGame):
    """
    扩展形式博弈 (Extensive-Form Game) 类。
    
    核心作用:
    对基础 ExtensiveFormGame 进行扩展，以支持 GPU 加速的数据结构 (TreeFormSequentialDecisionProcess)。
    对应论文 2.1 节 "FINITE EXTENSIVE-FORM GAMES" 的定义 1。
    """
    @classmethod
    def deserialize(cls, raw_data):
        game = super().deserialize(raw_data)
        game.tree_form_sequential_decision_processes = list(
            map(
                TreeFormSequentialDecisionProcess,
                game.tree_form_sequential_decision_processes,
            ),
        )

        return game


@dataclass
class TwoPlayerExtensiveFormGame(
        ExtensiveFormGame,
        TwoPlayerExtensiveFormGame,
):
    """
    双人扩展形式博弈。
    
    核心作用:
    专门处理双人博弈的数据结构，将效用矩阵转换为 SciPy/CuPy 稀疏矩阵格式，以便高效计算。
    """
    @classmethod
    def deserialize(cls, raw_data):
        game = super().deserialize(raw_data)
        game.utilities = tuple(map(csr_matrix, game.utilities))

        return game


@dataclass
class TwoPlayerZeroSumExtensiveFormGame(
        TwoPlayerExtensiveFormGame,
        TwoPlayerZeroSumExtensiveFormGame,
):
    """
    双人零和扩展形式博弈。
    
    核心作用:
    专门优化零和博弈 (Zero-Sum Games)。
    在零和博弈中，玩家1的收益 = -玩家2的收益。
    此类将效用矩阵进一步优化为 CSR 格式。
    
    对应理论:
    论文 2.3 节提到，对于 2-player zero-sum games，CFR 可以收敛到纳什均衡。
    """
    @classmethod
    def deserialize(cls, raw_data):
        game = super(TwoPlayerExtensiveFormGame, cls).deserialize(raw_data)
        game.utilities = csr_matrix(game.utilities)

        return game
