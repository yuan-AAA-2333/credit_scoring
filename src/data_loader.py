"""
数据加载与探索性数据分析（EDA）模块
职责：加载原始数据、输出数据概览、生成EDA图表
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from config import TARGET, EDA_NUMERIC, EDA_CATEGORICAL, FIGURES_DIR

# 设置中文字体
sns.set_theme(style='whitegrid')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def load_data(path):
    """加载原始CSV数据。"""
    return pd.read_csv(path)


def target_overview(df):
    """输出目标变量分布概览。"""
    print('数据形状:', df.shape)
    print('目标变量 y：1=违约(坏) 0=履约(好)')
    print(df[TARGET].value_counts().rename({0: '好客户', 1: '坏客户'}).to_string())
    print('坏账率: %.2f%%' % (df[TARGET].mean() * 100))
    print()


def missing_value_report(df):
    """全量缺失值诊断报告。"""
    miss = df.isna().sum()
    miss_table = (miss.rename('缺失数').to_frame()
                  .assign(**{'缺失率%': lambda x: (x['缺失数'] / len(df) * 100).round(2)}))
    print('缺失值一览（全部 %d 列的缺失率，含缺失为 0 的列）：' % len(miss_table))
    print(miss_table.to_string())
    print()
    return miss_table


def numeric_summary(df):
    """数值型变量描述统计。"""
    print('单变量分布（全量数值变量 describe）：')
    print(df[EDA_NUMERIC].describe().round(2).to_string())
    print()


def categorical_summary(df):
    """类别型变量频数统计。"""
    print('单变量分布（类别变量频数统计，含各档样本数）：')
    for c in EDA_CATEGORICAL:
        vc = df[c].astype(str).value_counts()
        top = vc.head(12)
        line = ', '.join(f'{k}={v}' for k, v in top.items())
        if len(vc) > 12:
            line += f', ...(共{len(vc)}类)'
        print(f'  {c}: {line}')
    print()


def plot_categorical_bars(df, save_path=None):
    """类别变量频数条形图。"""
    cat_plot = ['grade', 'term', 'home_ownership', 'verification_status', 'purpose', 'initial_list_status']
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    for ax, c in zip(axes.ravel(), cat_plot):
        vc = df[c].astype(str).value_counts()
        if len(vc) > 10:
            vc = vc.head(10)
        sns.barplot(x=vc.index, y=vc.values, ax=ax, color='steelblue')
        ax.set_title(c, fontsize=12)
        ax.set(ylabel='样本数', xlabel=None)
        ax.tick_params(axis='x', rotation=45, labelsize=8)
    plt.suptitle('类别变量频数分布', fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
    plt.close()


def outlier_report(df):
    """重点异常值识别。"""
    print('异常值识别：')
    print('  revol_util > 100% 的样本数:', (df['revol_util'] > 100).sum())
    print('  annual_inc 极端大值（99% 分位以上）:', (df['annual_inc'] > df['annual_inc'].quantile(0.99)).sum(),
          '，最大:', df['annual_inc'].max())
    print('  tot_coll_amt / tot_cur_bal / total_rev_hi_lim 三者同时缺失:',
          df[['tot_coll_amt', 'tot_cur_bal', 'total_rev_hi_lim']].isna().all(axis=1).sum())
    print()


def key_variable_analysis(df):
    """关键变量与违约率的初步关系。"""
    print('关键变量 × 违约率（初步关系）：')
    print('  term 36个月 坏账率: %.2f%% | term 60个月 坏账率: %.2f%%'
          % (df.loc[df['term'] == 36, TARGET].mean() * 100,
             df.loc[df['term'] == 60, TARGET].mean() * 100))
    grade_br = df.groupby('grade', observed=True)[TARGET].agg(['mean', 'count'])
    grade_br['mean'] = (grade_br['mean'] * 100).round(2)
    print('  grade × 坏账率：')
    print(grade_br.rename(columns={'mean': '坏账率%', 'count': '样本数'}).to_string())
    grade_map = {g: i for i, g in enumerate(sorted(df['grade'].unique()))}
    corr = round(df['int_rate'].corr(df['grade'].map(grade_map)), 4)
    print('  int_rate 与 grade 相关系数（有序映射后）:', corr)
    print()


def plot_boxplots(df, title, save_path=None):
    """箱线图可视化。"""
    _num_cols = ['loan_amnt', 'installment', 'annual_inc', 'dti', 'revol_bal',
                 'revol_util', 'total_acc', 'mths_since_earliest_cr_line']
    fig, axes = plt.subplots(2, 4, figsize=(18, 6))
    for ax, c in zip(axes.ravel(), _num_cols):
        sns.boxplot(y=df[c], ax=ax)
        ax.set_title(c, fontsize=10)
        ax.set(ylabel=None)
    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
    plt.close()


def run_eda(df, figures_dir=FIGURES_DIR):
    """执行完整EDA流程。"""
    import os
    os.makedirs(figures_dir, exist_ok=True)

    target_overview(df)
    missing_value_report(df)
    numeric_summary(df)
    categorical_summary(df)
    plot_categorical_bars(df, save_path=f'{figures_dir}/fig_categorical.png')
    outlier_report(df)
    key_variable_analysis(df)
    plot_boxplots(df, '清洗前箱线图（可见 annual_inc / revol_util / revol_bal 存在极端值）',
                  save_path=f'{figures_dir}/fig1_eda_boxplot_before.png')
