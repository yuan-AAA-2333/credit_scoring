"""
WOE编码与IV变量筛选模块
职责：计算WOE/IV、变量筛选、共线性诊断
核心原则：所有WOE/IV计算仅在训练集上进行
"""

import numpy as np
import pandas as pd
from config import TARGET, IV_THRESHOLD, CORR_THRESHOLD
from binner import apply_bins


def calc_woe_iv(df, bin_col, target=TARGET, smooth=0.5):
    """计算单特征的WOE和IV。

    WOE = ln(坏账占比/好账占比)，每箱计数 +0.5 平滑避免除零
    IV = Σ(坏占比-好占比)×WOE
    """
    g = df.groupby(bin_col, observed=True)[target].agg(['sum', 'count'])
    g.columns = ['bad', 'total']
    g['good'] = g['total'] - g['bad']
    bad_s = g['bad'] + smooth
    good_s = g['good'] + smooth
    g['bad_pct'] = bad_s / bad_s.sum()
    g['good_pct'] = good_s / good_s.sum()
    g['WOE'] = np.log(g['bad_pct'] / g['good_pct'])
    g['IV'] = (g['bad_pct'] - g['good_pct']) * g['WOE']
    return g, g['IV'].sum()


def compute_all_woe_iv(train_df, feature_cols, bins_map):
    """对所有候选特征计算WOE和IV，返回映射字典。"""
    woe_map = {}
    iv_map = {}
    woe_detail = {}
    for c in feature_cols:
        grp, iv = calc_woe_iv(
            train_df.assign(**{'__bin__': apply_bins(train_df[c], bins_map[c])}),
            '__bin__'
        )
        woe_map[c] = grp['WOE'].to_dict()
        iv_map[c] = iv
        woe_detail[c] = grp
    return woe_map, iv_map, woe_detail


def iv_filter(feature_cols, iv_map, threshold=IV_THRESHOLD):
    """IV筛选：剔除预测力弱的特征。"""
    selected = [c for c in feature_cols if iv_map[c] >= threshold]
    dropped = [c for c in feature_cols if iv_map[c] < threshold]
    return selected, dropped


def woe_encode(df, c, bins_map, woe_map):
    """对单个特征做WOE编码。"""
    return apply_bins(df[c], bins_map[c]).map(woe_map[c]).astype(float).fillna(0.0)


def collinearity_filter(selected, iv_map, train_woe_df, threshold=CORR_THRESHOLD):
    """共线性诊断：WOE相关性 > threshold 时剔除IV较低者。"""
    corr = train_woe_df.corr()
    drop_coll = set()
    for i in range(len(selected)):
        for j in range(i + 1, len(selected)):
            a, b = selected[i], selected[j]
            if abs(corr.loc[a, b]) > threshold and a not in drop_coll and b not in drop_coll:
                drop_coll.add(a if iv_map[a] < iv_map[b] else b)
    final_selected = [c for c in selected if c not in drop_coll]
    return final_selected, drop_coll


def print_iv_ranking(iv_map):
    """打印IV排序表。"""
    iv_df = (pd.DataFrame({'特征': list(iv_map.keys()), 'IV': list(iv_map.values())})
             .sort_values('IV', ascending=False).reset_index(drop=True))
    print('IV 排序表（全部候选特征）：')
    print(iv_df.round(4).to_string(index=False))
    print()
    return iv_df
