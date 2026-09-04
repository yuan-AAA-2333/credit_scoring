"""
特征分箱模块
职责：将连续/类别变量离散化为有序档位
核心原则：训练集定边界，测试集仅套用
"""

import numpy as np
import pandas as pd
from config import DERIVED_FEATURES, DEFAULT_N_BINS, MIN_BIN_FREQ


# 特征分类
def get_feature_kind_map():
    """返回每个特征的分箱类型映射。"""
    cat_cols = ['term', 'grade', 'home_ownership', 'verification_status',
                'purpose', 'addr_state', 'initial_list_status']
    cont_cols = ['loan_amnt', 'installment', 'emp_length', 'annual_inc', 'dti',
                 'open_acc', 'revol_bal', 'revol_util', 'total_acc',
                 'mths_since_earliest_cr_line', 'int_rate'] + DERIVED_FEATURES
    zero_cols = ['delinq_2yrs', 'inq_last_6mths', 'pub_rec', 'acc_now_delinq',
                 'collections_12_mths_ex_med', 'tot_coll_amt']
    special_cols = ['mths_since_last_delinq']  # -1 = 从未逾期

    kind_map = {}
    for c in cat_cols:
        kind_map[c] = 'cat'
    for c in cont_cols:
        kind_map[c] = 'cont'
    for c in zero_cols:
        kind_map[c] = 'zero'
    for c in special_cols:
        kind_map[c] = 'special'
    return kind_map


def _q_edges(s, n_bins):
    """等频分箱边界计算。"""
    qs = s.quantile([i / n_bins for i in range(1, n_bins)])
    return sorted(set(np.round(qs.values, 6)))


def build_bins(train_s, kind, n_bins=DEFAULT_N_BINS, min_freq=MIN_BIN_FREQ):
    """返回分箱规则字典。kind ∈ {'cont','zero','special','cat'}。

    - cont: 等频分箱
    - zero: 0单独成箱，正数再等频分箱
    - special: 特殊值(-1)单独成箱
    - cat: 稀有类合并为"其他"
    """
    if kind == 'cont':
        return {'edges': sorted(set([-np.inf] + _q_edges(train_s, n_bins) + [np.inf])), 'cat_map': None}
    if kind == 'zero':
        pos = train_s[train_s > 0]
        return {'edges': sorted(set([-np.inf, 0.0] + _q_edges(pos, max(n_bins - 1, 2)) + [np.inf])), 'cat_map': None}
    if kind == 'special':
        rest = train_s[train_s >= 0]
        return {'edges': sorted(set([-np.inf, -0.5] + _q_edges(rest, n_bins) + [np.inf])), 'cat_map': None}
    if kind == 'cat':
        s = train_s.astype(str)
        vc = s.value_counts()
        keep = set(vc[vc / len(s) >= min_freq].index)
        return {'edges': None, 'cat_map': {v: (v if v in keep else '其他') for v in vc.index}}


def apply_bins(s, rule):
    """把分箱规则套用到 Series 上，返回类别标签（字符串）。"""
    if rule['cat_map'] is not None:
        return s.astype(str).map(rule['cat_map']).fillna('其他')
    return pd.cut(s, rule['edges'], include_lowest=True).astype(str)


def build_all_bins(train_df, feature_cols, kind_map, no_merge=None):
    """为所有特征建立分箱规则。

    Args:
        train_df: 训练集DataFrame
        feature_cols: 特征列表
        kind_map: 特征类型映射
        no_merge: 不合并稀有类的类别特征集合（如 {'grade'}）

    Returns:
        dict: {特征名: 分箱规则}
    """
    if no_merge is None:
        no_merge = {'grade'}
    bins_map = {}
    for c in feature_cols:
        bins_map[c] = build_bins(
            train_df[c],
            kind_map[c],
            min_freq=(0.0 if c in no_merge else min_freq)
        )
    return bins_map
