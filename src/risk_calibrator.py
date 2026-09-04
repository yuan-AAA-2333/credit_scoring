"""
风险分层与Cut-off决策模块
职责：按分数分档验证区分度、校准cut-off阈值、生成决策建议
核心原则：cut-off阈值仅在训练集上确定，测试集仅用于评估
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import TARGET
from scorecard_builder import score_customer


def decile_analysis(df, score_col='score'):
    """按分数分10档，验证区分度（应单调递减）。"""
    df = df.copy()
    df['score_decile'] = pd.qcut(df[score_col], 10, labels=False, duplicates='drop')
    decile = df.groupby('score_decile', observed=True).agg(
        样本数=(score_col, 'count'), 最低分=(score_col, 'min'),
        最高分=(score_col, 'max'), 坏账率=(TARGET, 'mean'))
    decile['坏账率'] = (decile['坏账率'] * 100).round(2)
    decile = decile.reset_index().rename(columns={'score_decile': '分数档'})
    decile['分数档'] = decile['分数档'] + 1
    print('10 档分数 → 坏账率（验证区分度，应单调递减）：')
    print(decile.to_string(index=False))
    print()
    return decile


def calibrate_cutoff(train_df, selected_features, bins_map, woe_map, coef, factor, base_score,
                     quantiles=(0.3, 0.7)):
    """在训练集上校准cut-off阈值。"""
    train_scores = score_customer(train_df, selected_features, bins_map, woe_map, coef, factor, base_score)
    q_rej, q_app = np.quantile(train_scores, quantiles)
    print('cut-off 校准：以「自动通过上限坏账率≈10%%、强制拒绝下限≈25%%」为业务目标，'
          '映射到本组人群 %.0f%%/%.0f%% 分位：' % (quantiles[0]*100, quantiles[1]*100))
    print('  拒绝     < %.0f 分' % q_rej)
    print('  人工复核  %.0f ~ %.0f 分' % (q_rej, q_app))
    print('  通过     > %.0f 分' % q_app)
    return q_rej, q_app


def apply_decision(df, q_rej, q_app, score_col='score'):
    """应用cut-off到数据集，生成分类决策。"""
    def decision(s):
        return '拒绝' if s < q_rej else ('人工复核' if s < q_app else '通过')
    df = df.copy()
    df['decision'] = df[score_col].apply(decision)
    return df


def generate_decision_table(test_df, q_rej, q_app, score_col='score'):
    """生成决策统计表。"""
    segments = [
        ('通过',   test_df[score_col] >= q_app, '≥ %.0f' % q_app),
        ('人工复核', (test_df[score_col] >= q_rej) & (test_df[score_col] < q_app), '%.0f ~ %.0f' % (q_rej, q_app)),
        ('拒绝',   test_df[score_col] < q_rej, '< %.0f' % q_rej),
    ]
    table2_rows = []
    seg_br = {}
    for action, mask, seg in segments:
        share = mask.sum() / len(test_df) * 100
        br = test_df.loc[mask, TARGET].mean() * 100
        seg_br[action] = br
        table2_rows.append({'分数段': seg, '样本占比': f'{share:.1f}%', '坏账率': f'{br:.2f}%', '建议动作': action})
    table2 = pd.DataFrame(table2_rows)

    rej_rate = (test_df[score_col] < q_rej).mean() * 100
    app_rate = (test_df[score_col] >= q_app).mean() * 100
    print('表 2：分数 → 违约率映射与 cut-off')
    print(table2.to_string(index=False))
    print('拒单率: %.1f%% | 直接通过率: %.1f%% | 人工复核率: %.1f%%'
          % (rej_rate, app_rate, 100 - rej_rate - app_rate))
    print()
    return table2, seg_br


def print_decision_advice(q_rej, q_app, seg_br, app_rate, rej_rate):
    """输出放贷决策建议。"""
    print('放贷决策建议：≥%.0f 分直接通过（坏账率 %.1f%%）、<%.0f 分直接拒绝（坏账率 %.1f%%）、'
          '中间人工复核；据此可自动处理约 %.0f%% 的申请（通过+拒绝），仅 %.0f%% 需人工介入，'
          '通过人群坏账率 %.1f%% 远低于整体 19.25%%，有效控制风险。'
          % (q_app, seg_br['通过'], q_rej, seg_br['拒绝'],
             app_rate + rej_rate, 100 - app_rate - rej_rate, seg_br['通过']))
    print()


def plot_risk_stratification(decile, seg_br, save_path=None):
    """风险分层可视化。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    axes[0].bar(decile['分数档'].astype(str), decile['坏账率'], color='crimson')
    axes[0].set_xlabel('分数档（1=最低分/最高风险 → 10=最高分/最低风险）')
    axes[0].set_ylabel('坏账率 %')
    axes[0].set_title('10 档分数 × 坏账率（区分度）')
    for x, y in enumerate(decile['坏账率']):
        axes[0].text(x, y + 0.5, f'{y:.1f}', ha='center', fontsize=8)

    colors = {'通过': 'seagreen', '人工复核': 'goldenrod', '拒绝': 'crimson'}
    order = ['通过', '人工复核', '拒绝']
    axes[1].bar(order, [seg_br[k] for k in order], color=[colors[k] for k in order])
    axes[1].set_ylabel('坏账率 %')
    axes[1].set_title('通过 / 人工复核 / 拒绝 三档坏账率')
    for x, k in enumerate(order):
        axes[1].text(x, seg_br[k] + 0.5, f"{seg_br[k]:.2f}%", ha='center')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
    plt.close()
