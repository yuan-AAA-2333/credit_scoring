"""
模型训练与评估模块
职责：逻辑回归建模、AUC/KS计算、ROC与KS曲线绘制
核心原则：仅使用训练集拟合模型，测试集仅用于评估
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve


def train_lr(X_train, y_train, max_iter=2000):
    """训练逻辑回归模型。"""
    model = LogisticRegression(max_iter=max_iter)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """在测试集上评估模型，返回AUC和KS。"""
    prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, prob)
    fpr, tpr, _ = roc_curve(y_test, prob)
    ks = (tpr - fpr).max()
    return auc, ks, fpr, tpr, prob


def plot_roc_ks(fpr, tpr, auc, ks, save_path=None):
    """绘制ROC曲线和KS曲线。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].plot(fpr, tpr, color='crimson', lw=2, label='AUC = %.3f' % auc)
    axes[0].plot([0, 1], [0, 1], 'k--', lw=1)
    axes[0].set_xlabel('假正率 FPR')
    axes[0].set_ylabel('真正率 TPR')
    axes[0].set_title('ROC 曲线')
    axes[0].legend()

    ks_idx = np.argmax(tpr - fpr)
    axes[1].plot(tpr, label='TPR（累计坏账率）', color='crimson')
    axes[1].plot(fpr, label='FPR（累计好账误判率）', color='steelblue')
    axes[1].axvline(ks_idx, color='gray', ls='--', lw=1, label='KS = %.3f' % ks)
    axes[1].set_xlabel('样本（按风险从高到低排序）')
    axes[1].legend()
    axes[1].set_title('KS 曲线')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
    plt.close()


def print_coefficients(model, feature_names):
    """输出模型系数（按绝对值排序）。"""
    import pandas as pd
    coef_df = pd.DataFrame({'特征': feature_names, '系数': model.coef_[0]})
    coef_df = coef_df.sort_values('系数', key=abs, ascending=False).reset_index(drop=True)
    print('逻辑回归系数（按 |系数| 排序）：')
    print(coef_df.round(4).to_string(index=False))
    print()
    return coef_df
