"""
全局配置模块
存放项目级常量、随机种子、路径等全局参数
"""

import numpy as np

# -------------------------------
# 随机种子（全项目统一，保证可复现）
# -------------------------------
SEED = 42
np.random.seed(SEED)

# -------------------------------
# 目标变量
# -------------------------------
TARGET = 'y'  # 1=违约(坏), 0=履约(好)

# -------------------------------
# 数据集划分参数
# -------------------------------
TEST_SIZE = 0.3          # 测试集比例
STRATIFY_COL = TARGET    # 分层依据

# -------------------------------
# 特征工程：衍生特征列表
# -------------------------------
DERIVED_FEATURES = [
    'repay_burden',        # 年还款负担率 = 月供×12/年收入
    'loan_to_income',      # 贷款额收入比 = 申请金额/年收入
    'revol_bal_to_income', # 循环负债收入比 = 循环欠款/年收入
    'tot_bal_to_income',   # 总余额收入比 = 账户总余额/年收入
    'credit_line_util',    # 循环额度使用率 = 循环欠款/授信总额度
]

# -------------------------------
# 评分卡标尺参数（600/50 行业标准）
# -------------------------------
SCORE_P0 = 600      # 基准分数
SCORE_ODDS0 = 50    # 基准分数对应的odds
SCORE_PDO = 50      # odds翻倍所增加的分数

# -------------------------------
# 分箱参数
# -------------------------------
DEFAULT_N_BINS = 5
MIN_BIN_FREQ = 0.02  # 类别变量稀有类合并阈值

# -------------------------------
# IV 筛选阈值
# -------------------------------
IV_THRESHOLD = 0.02

# -------------------------------
# 共线性阈值
# -------------------------------
CORR_THRESHOLD = 0.9

# -------------------------------
# 路径配置
# -------------------------------
RAW_DATA_PATH = 'data/raw/lc_loan_data.csv'
OUTPUT_DATA_DIR = 'data/output'
FIGURES_DIR = 'outputs/figures'

# -------------------------------
# EDA 专用字段列表
# -------------------------------
EDA_NUMERIC = [
    'loan_amnt', 'funded_amnt', 'int_rate', 'installment', 'emp_length',
    'annual_inc', 'dti', 'delinq_2yrs', 'inq_last_6mths', 'mths_since_last_delinq',
    'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc',
    'collections_12_mths_ex_med', 'acc_now_delinq', 'tot_coll_amt', 'tot_cur_bal',
    'total_rev_hi_lim', 'mths_since_earliest_cr_line'
]

EDA_CATEGORICAL = [
    'grade', 'term', 'home_ownership', 'verification_status',
    'purpose', 'initial_list_status', 'addr_state', 'sub_grade'
]
