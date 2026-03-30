from dataclasses import dataclass

from cupyx.scipy.sparse import csr_matrix
import cupy as cp
from noregret.games import (
    ExtensiveFormGame,
    TwoPlayerExtensiveFormGame,
    TwoPlayerZeroSumExtensiveFormGame,
)
from noregret.utilities import TreeFormSequentialDecisionProcess as CpuTreeFormSequentialDecisionProcess

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


@dataclass
class MultiPlayerExtensiveFormGame(ExtensiveFormGame):
    """Multi-player (n-player) general-sum extensive-form game (EFG) on GPU.

    Utilities are stored in a sparse profile list:
    - `utilities['coords']`: (nnz, player_count) cp.int64 per-player seq indices
    - `utilities['values']`: list[player] of cp.float arrays of length nnz

    This supports sequence-form utility vector queries for each player:
      u_i[s_i] = sum_{s_{-i}} U_i[s_i,s_{-i}] * prod_{j!=i} x_j[s_j]
    """

    @classmethod
    def deserialize(cls, raw_data):
        # Deserialize CPU TFSDPs from the persisted bundle, then wrap for GPU.
        cpu_tfsdps = CpuTreeFormSequentialDecisionProcess.deserialize_all(
            raw_data['tree_form_sequential_decision_processes'],
        )
        tfsdps = list(map(TreeFormSequentialDecisionProcess, cpu_tfsdps))
        player_count = len(tfsdps)
        utilities = raw_data['utilities']

        def _unpack_csr_vector_payload(payload: dict, length: int):
            """Unpack an (nnz x 1) CSR payload into a dense cp.ndarray (length,)."""
            if payload.get('type') != 'csr':
                raise ValueError('unsupported sparse utility type')
            shape = tuple(payload['shape'])
            if shape != (length, 1):
                raise ValueError('unexpected sparse vector shape')
            indptr = cp.asarray(payload['indptr'], dtype=cp.int32)
            data = cp.asarray(payload['data'], dtype=cp.float32)
            out = cp.zeros(length, dtype=data.dtype)
            row_nnz = indptr[1:] - indptr[:-1]
            mask = row_nnz > 0
            if bool(mask.any()):
                positions = indptr[:-1][mask]
                out[mask] = data[positions]
            return out

        # Packed sparse profiles with per-player value vectors (recommended).
        if isinstance(utilities, dict) and utilities.get('kind') == 'scipy.sparse.profile_per_player':
            if int(utilities.get('player_count', player_count)) != player_count:
                raise ValueError('utilities player_count does not match tfsdps')
            if bool(utilities.get('zero_sum', False)):
                raise ValueError('expected general-sum utilities, got zero-sum')

            coords = cp.asarray(utilities['coords'], dtype=cp.int32)
            nnz = int(coords.shape[0])
            payloads = list(utilities['values'])
            if len(payloads) != player_count:
                raise ValueError('utilities values do not match player_count')
            values = [
                _unpack_csr_vector_payload(payloads[p], nnz).astype(cp.float32, copy=False)
                for p in range(player_count)
            ]
            return cls(tfsdps, {'coords': coords, 'values': values})

        # Legacy/raw list format (template compatible).
        if isinstance(utilities, list):
            seq_index = [
                {seq: i for i, seq in enumerate(tfsdps[p].sequences)}
                for p in range(player_count)
            ]
            nnz = len(utilities)
            coords_h = [[0] * player_count for _ in range(nnz)]
            values_h = [[0.0] * nnz for _ in range(player_count)]
            for k, raw_utility in enumerate(utilities):
                seqs = raw_utility.get('sequences')
                vals = raw_utility.get('values')
                if seqs is None or vals is None:
                    raise ValueError('expected raw utility entries with sequences/values')
                if len(seqs) != player_count or len(vals) != player_count:
                    raise ValueError('raw utility entry does not match player_count')
                for p in range(player_count):
                    coords_h[k][p] = int(seq_index[p][tuple(seqs[p])])
                    values_h[p][k] = float(vals[p])

            coords = cp.asarray(coords_h, dtype=cp.int32)
            values = [cp.asarray(v, dtype=cp.float32) for v in values_h]
            return cls(tfsdps, {'coords': coords, 'values': values})

        raise ValueError('unsupported utilities format for MultiPlayerExtensiveFormGame')

    def dimension(self, player):
        player = int(player)
        if not (0 <= player < self.player_count):
            raise ValueError(f'Player {player} does not exist')
        return len(self.tree_form_sequential_decision_processes[player].sequences)

    @property
    def utility_coords(self):
        return self.utilities['coords']

    @property
    def utility_values(self):
        return self.utilities['values']

    def all_player_utilities(self, *strategies):
        """Compute utility vectors for all players given a full strategy profile."""
        if len(strategies) != self.player_count:
            raise ValueError('expected strategies for all players')
        coords = self.utility_coords
        values = self.utility_values
        nnz = int(coords.shape[0])

        # Joint reach probability for each sparse profile row.
        reach = cp.ones(nnz, dtype=cp.float32)
        per_player_reach = []
        for p in range(self.player_count):
            s = strategies[p]
            r = s[coords[:, p]]
            per_player_reach.append(r)
            reach *= r

        utilities = []
        for p in range(self.player_count):
            denom = per_player_reach[p]
            opp_reach = cp.where(denom != 0, reach / denom, 0)
            weights = values[p] * opp_reach
            utilities.append(
                cp.bincount(
                    coords[:, p],
                    weights=weights,
                    minlength=self.dimension(p),
                ),
            )
        return utilities

    def utility(self, player, *opponent_strategies):
        player = int(player)
        if len(opponent_strategies) != self.player_count - 1:
            raise ValueError('expected opponent strategies for all other players')
        full = list(opponent_strategies)
        full.insert(player, None)

        coords = self.utility_coords
        nnz = int(coords.shape[0])
        reach = cp.ones(nnz, dtype=cp.float32)
        for p in range(self.player_count):
            if p == player:
                continue
            s = full[p]
            reach *= s[coords[:, p]]

        weights = self.utility_values[player] * reach
        return cp.bincount(
            coords[:, player],
            weights=weights,
            minlength=self.dimension(player),
        )

    def value(self, player, *strategies):
        player = int(player)
        if len(strategies) != self.player_count:
            raise ValueError('expected strategies for all players')
        coords = self.utility_coords
        nnz = int(coords.shape[0])
        reach = cp.ones(nnz, dtype=cp.float32)
        for p in range(self.player_count):
            s = strategies[p]
            reach *= s[coords[:, p]]
        return (self.utility_values[player] * reach).sum()

    def correlated_value(self, player, *strategies):
        raise NotImplementedError

    def best_response(self, player, *opponent_strategies):
        player = int(player)
        utility = self.utility(player, *opponent_strategies)
        tfsdp = self.tree_form_sequential_decision_processes[player]
        return tfsdp.sequence_form_best_response(utility)


# Backwards/explicit naming per doc/GPUPT_multi_player.md.
MultiPlayerGeneralSumExtensiveFormGame = MultiPlayerExtensiveFormGame
