"""
数据清洗模块
职责：缺失值填充、异常值截尾
核心原则：fit/apply 分离，防数据泄漏
"""


def clean_fit(df_train):
    """在训练集上学习填充中位数与截尾阈值，返回参数字典。

    为什么只在训练集 fit：中位数/分位点是用数据「学」出来的量，只能在训练集计算，
    测试集只套用，防止数据泄漏。
    """
    return {
        'revol_util_med': df_train['revol_util'].median(),
        'tot_coll_amt_med': df_train['tot_coll_amt'].median(),
        'tot_cur_bal_med': df_train['tot_cur_bal'].median(),
        'total_rev_hi_lim_med': df_train['total_rev_hi_lim'].median(),
        'inc_cap': df_train['annual_inc'].quantile(0.99),
    }


def clean_apply(df_in, params):
    """套用训练集学到的参数做填充与截尾。

    处理策略：
    - mths_since_last_delinq 缺失56% → 业务含义"从未逾期" → 填-1（分箱时单独成档）
    - revol_util 缺失0.06% → 随机缺失 → 训练中位数填充
    - collections_12_mths_ex_med 缺失0.03% → 99.5%为0 → 填0
    - tot_coll_amt/tot_cur_bal/total_rev_hi_lim 同批缺失28% → 训练中位数填充
    - revol_util > 100% → 截尾至100%
    - annual_inc 极端值 → 99%分位截尾
    """
    df = df_in.copy()

    # ── 缺失值处理 ──
    df['mths_since_last_delinq'] = df['mths_since_last_delinq'].fillna(-1)
    df['revol_util'] = df['revol_util'].fillna(params['revol_util_med'])
    df['collections_12_mths_ex_med'] = df['collections_12_mths_ex_med'].fillna(0)
    df['tot_coll_amt'] = df['tot_coll_amt'].fillna(params['tot_coll_amt_med'])
    df['tot_cur_bal'] = df['tot_cur_bal'].fillna(params['tot_cur_bal_med'])
    df['total_rev_hi_lim'] = df['total_rev_hi_lim'].fillna(params['total_rev_hi_lim_med'])

    # ── 异常值处理 ──
    df['revol_util'] = df['revol_util'].clip(upper=100)
    df['annual_inc'] = df['annual_inc'].clip(upper=params['inc_cap'])

    return df


def audit_missing(df_train, df_test):
    """缺失值处理审计：断言清洗后训练/测试均无缺失值。"""
    assert df_train.isna().sum().sum() == 0 and df_test.isna().sum().sum() == 0, '清洗后仍存在缺失值！'
    print('审计通过：清洗后训练集与测试集均无缺失值（NaN=0）。')
