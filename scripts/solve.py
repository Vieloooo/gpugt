from argparse import ArgumentParser, BooleanOptionalAction
from json import dump
from pathlib import Path
from resource import getrusage, RUSAGE_SELF
import sys
from sys import stdout
from time import time


def _bootstrap_local_submodules() -> None:
    """Allow running this script without installing sibling submodules.

    This repository vendors dependencies as git submodules under `projectroot/libs/*`.
    When executing a script by path (e.g. `python scripts/solve.py`), Python will not
    automatically add sibling submodules to `sys.path`, so imports like
    `from noregret...` may fail.
    """

    this_file = Path(__file__).resolve()
    project_root: Path | None = None
    for parent in this_file.parents:
        if parent.name == 'libs':
            project_root = parent.parent
            break

    if project_root is None:
        return

    for rel in (Path('libs/nogret'), Path('libs/gpugt')):
        candidate = (project_root / rel).resolve()
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))


_bootstrap_local_submodules()

from noregret.utilities import import_string
from tqdm import trange
import cupy as cp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_args():
    """
    解析命令行参数。
    
    参数说明:
    - game_path: 博弈文件路径 (通常是 json 或类似格式)。
    - game_import_string: 导入 Game 类的字符串 (例如 gpugt.games.TwoPlayerZeroSumExtensiveFormGame)。
    - regret_minimizer_import_string: 导入 RegretMinimizer 类的字符串 (例如 gpugt.regret_minimizers.CounterfactualRegretMinimization)。
    - iteration_count: CFR 迭代次数 T。
    - alternate: 是否交替更新策略 (Alternating Updates) 而不是同时更新。
    - exploitabilities: 记录可利用度 (Exploitability) 的输出路径。
    - values: 记录博弈价值 (Value) 的输出路径。
    - game_name: 博弈名称 (可选)。
    """
    parser = ArgumentParser()
    parser.add_argument('game_path', type=Path)
    parser.add_argument('game_import_string')
    parser.add_argument('regret_minimizer_import_string')
    parser.add_argument('iteration_count', type=int)
    parser.add_argument('-a', '--alternate', action=BooleanOptionalAction)
    parser.add_argument('-e', '--exploitabilities', type=Path)
    parser.add_argument('-v', '--values', type=Path)
    parser.add_argument('-n', '--game_name')

    return parser.parse_args()


def main():
    """
    主求解程序入口。
    
    核心流程:
    1. 加载博弈 (Game) 和 后悔最小化器 (Regret Minimizer, 如 CFR)。
    2. 初始化行玩家 (Row Player) 和列玩家 (Column Player) 的 CFR 求解器。
    3. 进行 T 次迭代 (args.iteration_count):
       - 生成当前策略 (next_strategy)。
       - 计算效用 (utility)。
       - 更新后悔值 (observe_utility)。
       - 如果设置了 alternate，则交替更新；否则同时更新 (Simultaneous Updates)。
    4. 定期计算可利用度 (Exploitability) 和博弈价值 (Value) 以监控收敛情况。
    5. 保存结果数据和绘图。
    
    理论对应:
    这是 doc_gpupt.md 中算法的顶层循环。
    Loop 对应 Eq. 8 (平均后悔的迭代累积) 和 Eq. 10 (平均策略的迭代累积) 的逐步过程。
    """
    args = parse_args()
    memory_pool = cp.get_default_memory_pool()
    pinned_memory_pool = cp.get_default_pinned_memory_pool()
    game_type = import_string(args.game_import_string)

    with open(args.game_path) as file:
        game = game_type.load(file)

    if args.game_name:
        game_name = args.game_name
    else:
        game_name = args.game_path.stem

    regret_minimizer_type = import_string(args.regret_minimizer_import_string)
    row_tfsdp = game.row_tree_form_sequential_decision_process
    row_cfr = regret_minimizer_type(row_tfsdp)
    column_tfsdp = game.column_tree_form_sequential_decision_process
    column_cfr = regret_minimizer_type(column_tfsdp)
    average_row_strategy = row_cfr.average_strategy
    average_column_strategy = column_cfr.average_strategy
    checkpoint = 1
    iterations = []
    times = []
    exploitabilities = []
    values = []

    for iteration in trange(1, args.iteration_count + 1):
        begin_time = time()

        if args.alternate:
            row_strategy = row_cfr.next_strategy()

            if iteration > 1:
                column_utility = game.column_utility(row_strategy)

                column_cfr.observe_utility(column_utility)

            column_strategy = column_cfr.next_strategy()
            row_utility = game.row_utility(column_strategy)

            row_cfr.observe_utility(row_utility)
        else:
            row_strategy = row_cfr.next_strategy()
            column_strategy = column_cfr.next_strategy()
            row_utility = game.row_utility(column_strategy)
            column_utility = game.column_utility(row_strategy)

            row_cfr.observe_utility(row_utility)
            column_cfr.observe_utility(column_utility)

        end_time = time()
        time_ = end_time - begin_time
        average_row_strategy = row_cfr.average_strategy
        average_column_strategy = column_cfr.average_strategy
        average_strategies = average_row_strategy, average_column_strategy

        if iteration == checkpoint and args.exploitabilities:
            exploitability = game.exploitability(*average_strategies)
            checkpoint *= 2
        else:
            exploitability = None

        value = game.row_value(*average_strategies)

        iterations.append(iteration)
        times.append(time_)
        exploitabilities.append(
            None if exploitability is None else exploitability.item(),
        )
        values.append(value.item())

    used_bytes = memory_pool.used_bytes()
    total_bytes = memory_pool.total_bytes()
    n_free_blocks = pinned_memory_pool.n_free_blocks()
    ru_maxrss = getrusage(RUSAGE_SELF).ru_maxrss
    data = {
        'Iteration': iterations,
        'Exploitability': exploitabilities,
        'Value': values,
    }
    df = pd.DataFrame(data)

    if args.exploitabilities:
        plt.clf()
        sns.lineplot(df, x='Iteration', y='Exploitability')
        plt.xscale('log')
        plt.yscale('log')
        plt.title(f'Exploitability of {game_name} in self-play')
        plt.savefig(args.exploitabilities)

    if args.values:
        plt.clf()
        sns.lineplot(df, x='Iteration', y='Value')
        plt.xscale('log')
        plt.title(f'Value of {game_name} in self-play')
        plt.savefig(args.values)

    row_sequences = row_tfsdp.sequences
    column_sequences = column_tfsdp.sequences
    data = {
        'game_path': str(args.game_path),
        'game_import_string': args.game_import_string,
        'regret_minimizer_import_string': args.regret_minimizer_import_string,
        'iteration_count': args.iteration_count,
        'alternate': args.alternate,
        'iterations': iterations,
        'times': times,
        'exploitabilities': exploitabilities,
        'values': values,
        'row_sequences': list(row_sequences),
        'average_row_strategy': average_row_strategy.tolist(),
        'column_sequences': list(column_sequences),
        'average_column_strategy': average_column_strategy.tolist(),
        'used_bytes': used_bytes,
        'total_bytes': total_bytes,
        'n_free_blocks': n_free_blocks,
        'ru_maxrss': ru_maxrss,
    }

    dump(data, stdout)


if __name__ == '__main__':
    main()
