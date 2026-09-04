"""
模型评估与加分项模块
职责：多模型对比、时间OOT验证、衍生特征消融实验
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve

from config import TARGET, SEED
from data_cleaner import clean_fit, clean_apply
from feature_engineer import feat_fit, feat_apply
from binner import build_bins, apply_bins, get_feature_kind_map
from woe_iv import calc_woe_iv, collinearity_filter, iv_filter


def model_comparison(train_df, test_df, selected_features, y_train, y_test, auc_lr):
    """附录A：多模型对比（RandomForest / XGBoost / LightGBM）。"""
    # One-Hot编码准备
    cat_cols = ['term', 'grade', 'home_ownership', 'verification_status',
                'purpose', 'addr_state', 'initial_list_status']
    cats = [c for c in cat_cols if c in selected_features]
    X_tr = train_df[selected_features].copy()
    X_te = test_df[selected_features].copy()
    for c in cats:
        X_tr[c] = X_tr[c].astype(str)
        X_te[c] = X_te[c].astype(str)
    X_tr = pd.get_dummies(X_tr, columns=cats)
    X_te = pd.get_dummies(X_te, columns=cats)
    X_tr, X_te = X_tr.align(X_te, join='left', axis=1)
    X_te = X_te.fillna(0)

    # 随机森林
    rf = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=50, n_jobs=-1, random_state=SEED)
    rf.fit(X_tr, y_train)
    auc_rf = roc_auc_score(y_test, rf.predict_proba(X_te)[:, 1])

    # 梯度提升（优先XGBoost，退而求其次LightGBM，最后sklearn）
    try:
        import xgboost as xgb
        gbm = xgb.XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=5,
                                min_child_weight=50, subsample=0.8, colsample_bytree=0.8,
                                eval_metric='auc', random_state=SEED)
        gbm.fit(X_tr, y_train)
        auc_gbm = roc_auc_score(y_test, gbm.predict_proba(X_te)[:, 1])
        gbm_name = 'XGBoost'
    except Exception:
        try:
            import lightgbm as lgb
            gbm = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                                     min_child_samples=50, random_state=SEED, verbose=-1)
            gbm.fit(X_tr, y_train)
            auc_gbm = roc_auc_score(y_test, gbm.predict_proba(X_te)[:, 1])
            gbm_name = 'LightGBM'
        except Exception:
            from sklearn.ensemble import HistGradientBoostingClassifier
            gbm = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                                 max_leaf_nodes=31, min_samples_leaf=50, random_state=SEED)
            gbm.fit(X_tr, y_train)
            auc_gbm = roc_auc_score(y_test, gbm.predict_proba(X_te)[:, 1])
            gbm_name = 'HistGradientBoosting'

    print('模型对比（同一清洗/特征，70/30 划分）：')
    print('  Logistic Regression(WOE)  AUC = %.4f  ← 主模型' % auc_lr)
    print('  RandomForest              AUC = %.4f' % auc_rf)
    print('  %-26s AUC = %.4f' % (gbm_name, auc_gbm))
    print('说明：树模型 AUC 与 LR 基本相当，但评分卡要求可解释，故仍以 LR 为主模型。')
    print()
    return {'LR': auc_lr, 'RF': auc_rf, gbm_name: auc_gbm}


def oot_validation(df_raw, feature_cols, kind_map):
    """附录B：时间切分验证（按issue_month前80%训练/后20%测试）。"""
    df_raw2 = df_raw.copy()
    df_raw2['ym'] = df_raw2['issue_month'].str.replace('-', '').astype(int)
    df_raw2 = df_raw2.sort_values('ym').reset_index(drop=True)
    n_ts = len(df_raw2)
    ts_cut = int(n_ts * 0.8)
    tr_raw_ts, te_raw_ts = df_raw2.iloc[:ts_cut], df_raw2.iloc[ts_cut:]
    print('时间切分：训练 %d 行(%.2f) | 测试 %d 行(%.2f)'
          % (len(tr_raw_ts), len(tr_raw_ts) / n_ts, len(te_raw_ts), len(te_raw_ts) / n_ts))

    # 在时间训练集上重新执行全部拟合流程
    cp_ts = clean_fit(tr_raw_ts)
    tr_c_ts = clean_apply(tr_raw_ts, cp_ts)
    te_c_ts = clean_apply(te_raw_ts, cp_ts)
    fp_ts = feat_fit(tr_c_ts)
    tr_ts = feat_apply(tr_c_ts, fp_ts)
    te_ts = feat_apply(te_c_ts, fp_ts)

    bins_ts = {c: build_bins(tr_ts[c], kind_map[c]) for c in feature_cols}
    woe_ts = {}
    iv_ts = {}
    for c in feature_cols:
        grp, iv = calc_woe_iv(tr_ts.assign(**{'__bin__': apply_bins(tr_ts[c], bins_ts[c])}), '__bin__')
        woe_ts[c] = grp['WOE'].to_dict()
        iv_ts[c] = iv

    sel_iv_ts = [c for c in feature_cols if iv_ts[c] >= 0.02]
    Xtr_ts_all = pd.DataFrame({c: apply_bins(tr_ts[c], bins_ts[c]).map(woe_ts[c]).astype(float).fillna(0.0) for c in sel_iv_ts})
    corr_ts = Xtr_ts_all.corr()
    drop_coll_ts = set()
    for i in range(len(sel_iv_ts)):
        for j in range(i + 1, len(sel_iv_ts)):
            a, b = sel_iv_ts[i], sel_iv_ts[j]
            if abs(corr_ts.loc[a, b]) > 0.9 and a not in drop_coll_ts and b not in drop_coll_ts:
                drop_coll_ts.add(a if iv_ts[a] < iv_ts[b] else b)
    sel_ts = [c for c in sel_iv_ts if c not in drop_coll_ts]

    Xtr_ts = pd.DataFrame({c: apply_bins(tr_ts[c], bins_ts[c]).map(woe_ts[c]).astype(float).fillna(0.0) for c in sel_ts})
    Xte_ts = pd.DataFrame({c: apply_bins(te_ts[c], bins_ts[c]).map(woe_ts[c]).astype(float).fillna(0.0) for c in sel_ts})
    m_ts = LogisticRegression(max_iter=2000).fit(Xtr_ts, tr_ts[TARGET].values)
    p_ts = m_ts.predict_proba(Xte_ts)[:, 1]
    auc_ts = roc_auc_score(te_ts[TARGET].values, p_ts)
    fpr_ts, tpr_ts, _ = roc_curve(te_ts[TARGET].values, p_ts)
    ks_ts = (tpr_ts - fpr_ts).max()
    print('时间切分（OOT 样本外）测试 AUC = %.4f | KS = %.4f' % (auc_ts, ks_ts))
    print()
    return auc_ts, ks_ts


def ablation_test(X_train, X_test, y_train, y_test, selected_features, derived_features):
    """附录C：衍生特征消融实验。"""
    base_features = [c for c in selected_features if c not in derived_features]
    m_base = LogisticRegression(max_iter=2000).fit(X_train[base_features], y_train)
    auc_base = roc_auc_score(y_test, m_base.predict_proba(X_test[base_features])[:, 1])
    auc_full = roc_auc_score(y_test, LogisticRegression(max_iter=2000).fit(X_train, y_train).predict_proba(X_test)[:, 1])
    print('不含衍生特征 LR AUC = %.4f' % auc_base)
    print('含衍生特征 LR AUC   = %.4f（提升 %.4f）' % (auc_full, auc_full - auc_base))
    print()
    return auc_base, auc_full
