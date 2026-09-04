"""
特征工程模块
职责：构造衍生特征、极端值处理
核心原则：fit/apply 分离，防数据泄漏
"""

import numpy as np
from config import DERIVED_FEATURES


def _derive(df):
    """构造5个衍生特征（偿债压力维度）。"""
    inc = df['annual_inc'].replace(0, np.nan)
    df['repay_burden'] = df['installment'] * 12 / inc         # 年还款负担率
    df['loan_to_income'] = df['loan_amnt'] / inc               # 贷款额收入比
    df['revol_bal_to_income'] = df['revol_bal'] / inc          # 循环负债收入比
    df['tot_bal_to_income'] = df['tot_cur_bal'] / inc          # 总余额收入比
    df['credit_line_util'] = df['revol_bal'] / df['total_rev_hi_lim'].replace(0, np.nan)  # 循环额度使用率
    return df


def feat_fit(df_train):
    """在训练集上计算衍生特征的截尾/填充参数（99%分位截尾+中位数填充）。"""
    tmp = _derive(df_train.copy())
    params = {}
    for c in DERIVED_FEATURES:
        s = tmp[c].replace([np.inf, -np.inf], np.nan)
        params[f'{c}_cap'] = s.quantile(0.99)
        params[f'{c}_med'] = s.median()
    return params


def feat_apply(df_in, params):
    """构造衍生特征，并用训练集参数做极端值处理。"""
    df = _derive(df_in.copy())
    for c in DERIVED_FEATURES:
        df[c] = df[c].replace([np.inf, -np.inf], np.nan)
        df[c] = df[c].clip(upper=params[f'{c}_cap'])
        df[c] = df[c].fillna(params[f'{c}_med'])
    return df


def print_derived_features():
    """打印衍生特征说明。"""
    print('衍生特征（5 个）及业务含义：')
    print('  repay_burden       年还款负担率 = 月供×12/年收入，越高还贷压力越大、越易违约')
    print('  loan_to_income     贷款额收入比 = 申请金额/年收入，衡量贷款杠杆，越高越激进')
    print('  revol_bal_to_income 循环负债收入比 = 循环欠款/年收入，存量循环负债的相对负担')
    print('  tot_bal_to_income  总余额收入比 = 账户总余额/年收入，总负债规模相对收入')
    print('  credit_line_util   循环额度使用率 = 循环欠款/授信总额度（交叉特征）')
    print()
