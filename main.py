#!/usr/bin/env python3
"""
贷前信用评分卡建模 — 主入口脚本
================================
完整建模流程：
  原始数据 → EDA → 数据清洗 → 特征工程 → 数据划分 → 特征分箱
  → WOE编码 → IV筛选 → 共线性剔除 → 逻辑回归建模 → 模型评估
  → 评分卡转换 → 风险分层 → Cut-off决策

加分项：多模型对比、时间OOT验证、衍生特征消融实验

使用方法：
  cd credit_scoring
  python main.py
"""

import sys
import os

# 将 src 加入 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np

# -------------------------------
# 导入各功能模块
# -------------------------------
from config import (
    SEED, TARGET, TEST_SIZE, RAW_DATA_PATH,
    OUTPUT_DATA_DIR, FIGURES_DIR, DERIVED_FEATURES
)
from data_loader import load_data, run_eda
from data_cleaner import clean_fit, clean_apply, audit_missing
from feature_engineer import feat_fit, feat_apply, print_derived_features
from binner import get_feature_kind_map, build_bins, apply_bins, build_all_bins
from woe_iv import (
    compute_all_woe_iv, iv_filter, woe_encode,
    collinearity_filter, print_iv_ranking
)
from model_trainer import train_lr, evaluate_model, plot_roc_ks, print_coefficients
from scorecard_builder import (
    calc_scoring_params, build_scorecard,
    score_customer, verify_score_consistency, demo_scoring
)
from risk_calibrator import (
    decile_analysis, calibrate_cutoff, apply_decision,
    generate_decision_table, print_decision_advice, plot_risk_stratification
)
from evaluator import model_comparison, oot_validation, ablation_test

# sklearn
from sklearn.model_selection import train_test_split


def main():
    # ============================================================
    # 0. 环境准备
    # ============================================================
    os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    np.random.seed(SEED)
    pd.set_option('display.max_columns', 50)
    pd.set_option('display.width', 200)

    # ============================================================
    # ① EDA：业务理解与数据探索
    # ============================================================
    print('=' * 70)
    print('① 业务理解与数据探索（EDA）')
    print('=' * 70)

    df_raw = load_data(RAW_DATA_PATH)
    run_eda(df_raw)

    # ============================================================
    # ② 数据清洗（fit/apply 分离）
    # ============================================================
    print('=' * 70)
    print('② 数据清洗')
    print('=' * 70)

    # ③ 特征工程
    print('=' * 70)
    print('③ 特征工程（衍生特征）')
    print('=' * 70)
    print_derived_features()

    # ============================================================
    # ④ 数据集划分（70/30，先划分再清洗/特征，防泄漏）
    # ============================================================
    print('=' * 70)
    print('④ 数据集划分（70/30，固定 seed=%d）' % SEED)
    print('=' * 70)

    train_raw, test_raw = train_test_split(
        df_raw, test_size=TEST_SIZE, stratify=df_raw[TARGET], random_state=SEED
    )

    # 清洗：训练集 fit → 统一 apply
    clean_params = clean_fit(train_raw)
    train_c = clean_apply(train_raw, clean_params)
    test_c = clean_apply(test_raw, clean_params)

    # 特征工程：训练集 fit → 统一 apply
    feat_params = feat_fit(train_c)
    train = feat_apply(train_c, feat_params)
    test = feat_apply(test_c, feat_params)

    print(f'训练集 {len(train)} 行 | 测试集 {len(test)} 行')
    print('训练集坏账率 %.2f%% | 测试集坏账率 %.2f%%'
          % (train[TARGET].mean() * 100, test[TARGET].mean() * 100))
    audit_missing(train, test)
    print('  annual_inc 截尾阈值(训练集): %.0f' % clean_params['inc_cap'])
    print()

    # ============================================================
    # ⑤ 分箱（训练集定边界，测试集套用）
    # ============================================================
    print('=' * 70)
    print('⑤ 分箱')
    print('=' * 70)

    kind_map = get_feature_kind_map()
    # 注意：这里需要手动构造 feature_cols（binner 模块中未暴露）
    cat_cols = ['term', 'grade', 'home_ownership', 'verification_status',
                'purpose', 'addr_state', 'initial_list_status']
    cont_cols = ['loan_amnt', 'installment', 'emp_length', 'annual_inc', 'dti',
                 'open_acc', 'revol_bal', 'revol_util', 'total_acc',
                 'mths_since_earliest_cr_line', 'int_rate'] + DERIVED_FEATURES
    zero_cols = ['delinq_2yrs', 'inq_last_6mths', 'pub_rec', 'acc_now_delinq',
                 'collections_12_mths_ex_med', 'tot_coll_amt']
    special_cols = ['mths_since_last_delinq']
    feature_cols = cat_cols + cont_cols + zero_cols + special_cols
    print('候选特征数:', len(feature_cols))

    NO_MERGE = {'grade'}
    bins_map = {c: build_bins(train[c], kind_map[c], min_freq=(0.0 if c in NO_MERGE else 0.02))
                for c in feature_cols}
    print('分箱规则已在训练集上建立。示例：')
    print('  term:', bins_map['term']['cat_map'])
    print('  delinq_2yrs(0单独成箱) 边界:', bins_map['delinq_2yrs']['edges'])
    print('  mths_since_last_delinq(-1单独成箱) 边界:', bins_map['mths_since_last_delinq']['edges'])
    print()

    # ============================================================
    # ⑥ WOE 与 IV
    # ============================================================
    print('=' * 70)
    print('⑥ WOE 与 IV')
    print('=' * 70)

    woe_map, iv_map, woe_detail = compute_all_woe_iv(train, feature_cols, bins_map)
    print_iv_ranking(iv_map)

    # IV 筛选
    selected, dropped_iv = iv_filter(feature_cols, iv_map)
    print('IV < 0.02 剔除:', dropped_iv)
    print('IV 保留特征数:', len(selected))
    print()

    # WOE 编码矩阵（训练集）
    X_train_woe = pd.DataFrame({c: woe_encode(train, c, bins_map, woe_map) for c in selected})

    # 共线性剔除
    selected, drop_coll = collinearity_filter(selected, iv_map, X_train_woe)
    if drop_coll:
        print('共线性剔除（|WOE相关|>0.9）:', sorted(drop_coll))
    print('最终入模特征数:', len(selected))
    print('最终入模特征:', selected)
    print()

    # ============================================================
    # ⑦ 逻辑回归建模与评估
    # ============================================================
    print('=' * 70)
    print('⑦ 逻辑回归建模与评估')
    print('=' * 70)

    X_train = X_train_woe[selected]
    X_test = pd.DataFrame({c: woe_encode(test, c, bins_map, woe_map) for c in selected})
    y_train = train[TARGET].values
    y_test = test[TARGET].values

    model = train_lr(X_train, y_train)
    auc, ks, fpr, tpr, prob = evaluate_model(model, X_test, y_test)
    print('测试集 AUC = %.4f' % auc)
    print('测试集 KS  = %.4f' % ks)
    print()

    plot_roc_ks(fpr, tpr, auc, ks, save_path=f'{FIGURES_DIR}/fig3_roc_ks.png')
    print_coefficients(model, selected)

    # ============================================================
    # ⑧ 评分映射与评分卡输出
    # ============================================================
    print('=' * 70)
    print('⑧ 评分映射与评分卡')
    print('=' * 70)

    # 手动补回特征名到模型（sklearn 1.0+ 的 feature_names_in_ 在旧版本可能不存在）
    if not hasattr(model, 'feature_names_in_'):
        model.feature_names_in_ = np.array(selected)

    factor, offset, base_score, coef = calc_scoring_params(model)
    # 修正：calc_scoring_params 依赖 feature_names_in_，这里直接用手动映射
    coef = dict(zip(selected, model.coef_[0]))

    print('评分参数：Factor = %.4f | Offset = %.4f | 基准分(base_score) = %.1f'
          % (factor, offset, base_score))
    print('（含义：好客户 odds=50 对应 600 分，odds 翻倍 +50 分）')
    print()

    scorecard_table = build_scorecard(selected, woe_detail, iv_map, coef, factor, base_score)
    print('表 1：标准评分卡（得分>0 加分、得分<0 减分）')
    print(scorecard_table.to_string(index=False))
    print('打分：总分 = 基础分 + 各变量命中档位得分之和，分越高风险越低。')
    scorecard_table.to_csv(f'{OUTPUT_DATA_DIR}/scorecard_table1.csv', index=False, encoding='utf-8-sig')
    print()

    # 打分函数与一致性校验
    test = test.copy()
    test['score'] = score_customer(test, selected, bins_map, woe_map, coef, factor, base_score)
    verify_score_consistency(test['score'].values, prob, offset, factor)
    print('测试集分数分布：')
    print(test['score'].describe().round(1).to_string())
    print()

    demo_scoring(test, selected, bins_map, woe_map, coef, factor, base_score)

    # ============================================================
    # ⑨ 风险分层与 cut-off 校准
    # ============================================================
    print('=' * 70)
    print('⑨ 风险分层与 cut-off 校准')
    print('=' * 70)

    decile = decile_analysis(test)
    q_rej, q_app = calibrate_cutoff(train, selected, bins_map, woe_map, coef, factor, base_score)

    test = apply_decision(test, q_rej, q_app)
    table2, seg_br = generate_decision_table(test, q_rej, q_app)
    table2.to_csv(f'{OUTPUT_DATA_DIR}/scorecard_table2_decision.csv', index=False, encoding='utf-8-sig')

    app_rate = (test['decision'] == '通过').mean() * 100
    rej_rate = (test['decision'] == '拒绝').mean() * 100
    print_decision_advice(q_rej, q_app, seg_br, app_rate, rej_rate)
    plot_risk_stratification(decile, seg_br, save_path=f'{FIGURES_DIR}/fig4_risk_stratify.png')

    # ============================================================
    # 附录 A：模型对比
    # ============================================================
    print('=' * 70)
    print('附录 A：模型对比（RandomForest / XGBoost）')
    print('=' * 70)
    model_comparison(train, test, selected, y_train, y_test, auc)

    # ============================================================
    # 附录 B：时间切分验证
    # ============================================================
    print('=' * 70)
    print('附录 B：时间切分验证（按 issue_month）')
    print('=' * 70)
    oot_validation(df_raw, feature_cols, kind_map)

    # ============================================================
    # 附录 C：衍生特征消融实验
    # ============================================================
    print('=' * 70)
    print('附录 C：衍生特征对模型表现的提升')
    print('=' * 70)
    ablation_test(X_train, X_test, y_train, y_test, selected, DERIVED_FEATURES)

    print('=' * 70)
    print('全部完成。')
    print('=' * 70)


if __name__ == '__main__':
    main()
