"""
评分卡构建模块
职责：将逻辑回归系数转换为标准评分卡（600/50惯例）
"""

import numpy as np
import pandas as pd
from config import SCORE_P0, SCORE_ODDS0, SCORE_PDO
from binner import apply_bins


def calc_scoring_params(model):
    """计算评分卡参数（Factor/Offset/基础分）。

    600/50惯例：好客户odds=50对应600分，odds翻倍+50分
    """
    factor = SCORE_PDO / np.log(2)
    offset = SCORE_P0 - factor * np.log(SCORE_ODDS0)
    intercept = model.intercept_[0]
    base_score = offset - factor * intercept
    coef = dict(zip(model.feature_names_in_ if hasattr(model, 'feature_names_in_') else [], model.coef_[0]))
    return factor, offset, base_score, coef


def build_scorecard(selected_features, woe_detail, iv_map, coef, factor, base_score):
    """生成评分卡明细表。"""
    scorecard_rows = []
    for c in selected_features:
        grp = woe_detail[c]
        for bin_label, row in grp.iterrows():
            woe_val = row['WOE']
            # WOE高=风险高=扣分（带负号）
            score = -factor * coef[c] * woe_val
            scorecard_rows.append({
                '变量': c, '分箱': bin_label, '样本数': int(row['total']),
                'WOE': round(woe_val, 4), '得分': round(score, 1)
            })
    scorecard = pd.DataFrame(scorecard_rows)

    # 分箱标签美化
    scorecard.loc[scorecard['变量'] == 'term', '分箱'] = \
        scorecard.loc[scorecard['变量'] == 'term', '分箱'].astype(str) + ' 个月'

    # 变量按IV降序、箱内按得分降序
    var_order = sorted(selected_features, key=lambda c: -iv_map[c])
    scorecard['_o'] = scorecard['变量'].map({v: i for i, v in enumerate(var_order)})
    scorecard = scorecard.sort_values(['_o', '得分'], ascending=[True, False]).drop(columns='_o').reset_index(drop=True)

    # 格式化得分（正数加+号）
    def fmt_score(s):
        return ('+' if s >= 0 else '') + f'{s:.1f}'
    scorecard['得分'] = scorecard['得分'].apply(fmt_score)

    # 插入基础分行
    base_row = pd.DataFrame([{'变量': '基础分', '分箱': '—', '样本数': '—', 'WOE': '—', '得分': f'{base_score:.1f}'}])
    scorecard_table = pd.concat([base_row, scorecard], ignore_index=True)

    return scorecard_table


def score_customer(df, selected_features, bins_map, woe_map, coef, factor, base_score):
    """给一批客户打分。"""
    total = np.full(len(df), base_score)
    for c in selected_features:
        total -= factor * coef[c] * apply_bins(df[c], bins_map[c]).map(woe_map[c]).astype(float).fillna(0.0)
    return total


def verify_score_consistency(scores, prob, offset, factor):
    """评分一致性校验：评分卡求和 vs 概率反推。"""
    odds_good = (1 - prob) / prob
    diff = np.abs(scores - (offset + factor * np.log(odds_good))).max()
    print('评分一致性校验：评分卡求和 vs 概率反推，最大绝对差 = %.4f' % diff)
    return diff


def demo_scoring(df, selected_features, bins_map, woe_map, coef, factor, base_score):
    """新客户打分演示（取第1位客户为例）。"""
    demo = df.iloc[0]
    print('如何给一个新客户打分（以第 1 位客户为例）：')
    print('  ① 看每个特征落在哪个档位 → ② 查评分卡取该档得分 → ③ 总分 = 基础分 + 各档得分之和。')
    demo_rows = []
    for c in selected_features:
        bl = apply_bins(pd.Series([demo[c]]), bins_map[c]).iloc[0]
        w = woe_map[c].get(bl, 0.0)
        bl_disp = bl + ' 个月' if c == 'term' else bl
        score_val = -factor * coef[c] * w
        demo_rows.append({'特征': c, '命中档位': bl_disp, '得分': ('+' if score_val >= 0 else '') + f'{score_val:.1f}'})
    demo_df = pd.DataFrame(demo_rows)
    print(demo_df.to_string(index=False))
    total_score = demo.get('score')
    if total_score is None or pd.isna(total_score):
        scores_arr = score_customer(pd.DataFrame([demo]), selected_features, bins_map, woe_map, coef, factor, base_score)
        total_score = float(scores_arr[0])
    print('  基础分 %.1f + 各档得分 = 该客户总分 %.1f；再按阈值判定通过/复核/拒绝。' % (base_score, total_score))
    print('  基础分 %.1f + 各档得分 = 该客户总分 %.1f；再按阈值判定通过/复核/拒绝。' % (base_score, total_score))
    print()
