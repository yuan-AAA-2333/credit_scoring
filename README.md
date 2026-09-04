# 贷前信用评分卡建模 · Credit Scoring Scorecard

> **Application Scorecard (A-Card) for Loan Pre-approval** — A complete, production-style credit scoring system built on real Lending Club loan data: EDA → feature engineering → WOE/IV binning → logistic regression → scorecard conversion → risk-based cutoff decisions.

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-F7931E?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue)
![AUC](https://img.shields.io/badge/Test%20AUC-0.7024-brightgreen)

---

## 📌 项目简介 / About

基于 **Lending Club 2007–2014 年真实放贷数据**（50,000 条样本、31 个原始特征），构建一整套可落地的**申请评分卡（A卡）**：对贷款申请人的违约风险进行量化打分，并输出**通过 / 人工复核 / 拒绝**三档放贷决策。

本项目不只是一个模型，而是一条**从原始数据到业务决策的完整流水线**：数据清洗与特征工程严格遵循 `fit/apply` 分离防止数据泄漏，分箱与 WOE/IV 编码在训练集上定边界，评分卡按业界 600/50 惯例换算，最终用风险分层与 Cut-off 校准直接生成可执行的审批策略。

**英文一句话简介（可用于简历/作品集）**：*End-to-end application credit scorecard built on real Lending Club data — anti-leakage pipeline, WOE/IV feature selection, logistic regression, scorecard mapping (600/50 convention) and three-tier cutoff decisions, with OOT validation and ablation experiments.*

---

## ✨ 核心亮点 / Highlights

1. **全链路防泄漏**：清洗 / 分箱 / WOE / IV / Cut-off 全部在训练集 `fit`，测试集仅 `apply`，杜绝信息泄露
2. **IV 阈值敏感性分析**：测试 0.005~0.050 五个阈值并择优（CV AUC 提升 +0.0036）
3. **业务衍生特征**：构造 5 个偿债压力维度指标，消融实验验证有效（AUC +0.0022）
4. **可落地的审批决策**：三档 Cut-off 自动处理约 60% 申请，通过人群坏账率下降 60%
5. **时间稳定性验证**：按 `issue_month` 做 OOT 外推验证（AUC = 0.7129），证明模型跨期稳定
6. **多模型对比**：附 RandomForest / XGBoost 对比，验证逻辑回归在评分卡场景的工程优势

---

## 📊 关键结果 / Key Results

| 指标 | 数值 | 说明 |
|------|------|------|
| 测试集 AUC | **0.7024** | 区分好坏客户的整体能力（基准 ≈0.70） |
| 测试集 KS | **0.3021** | 好坏客户累计分布最大差异（基准 ≈0.30） |
| 入模特征数 | 21 | 经 IV 筛选 + 共线性剔除后 |
| 基础分 / Factor | 421.3 / 72.13 | 600/50 评分惯例（odds=50 → 600 分，odds 翻倍 +50） |
| 通过阈值 | ≥464 分（坏账率 7.66%） | 自动审批，约覆盖 60% 申请 |
| 拒绝阈值 | <403 分（坏账率 33.08%） | 直接拒绝高风险申请 |
| OOT AUC | **0.7129** | 时间外推验证，跨期稳定性优于随机划分 |

---

## 📁 项目结构 / Structure

```
credit_scoring/
├── data/
│   ├── raw/                    # 原始数据（Lending Club 2007-2014）
│   └── output/                 # 输出：评分卡表1（明细）、表2（决策）
├── src/                        # 核心源码（模块化、可复用）
│   ├── config.py               # 全局配置（随机种子、路径、评分卡参数）
│   ├── data_loader.py          # 数据加载与 EDA 报告
│   ├── data_cleaner.py         # 数据清洗（fit/apply 分离）
│   ├── feature_engineer.py     # 特征工程（5 个偿债压力衍生特征）
│   ├── binner.py               # 等频分箱 / 0 单独成箱 / 稀有类合并
│   ├── woe_iv.py               # WOE 编码、IV 计算与筛选、共线性诊断
│   ├── model_trainer.py        # 逻辑回归训练、AUC/KS 评估、ROC/KS 曲线
│   ├── scorecard_builder.py    # 600/50 评分卡转换、打分、一致性校验
│   ├── risk_calibrator.py      # 十分位分析、Cut-off 校准、决策表生成
│   └── evaluator.py            # 加分项：多模型对比 / OOT / 消融实验
├── outputs/figures/            # 生成的图表（EDA、ROC/KS、风险分层）
├── notebooks/                  # 原始分析笔记本（保留）
├── docs/                       # 工程化待办清单（质量/功能）
├── main.py                     # 主入口：一键运行全流程
├── requirements.txt            # Python 依赖
└── .gitignore
```

---

## 🚀 快速开始 / Quick Start

```bash
# 1. 克隆仓库
git clone https://github.com/yuan-AAA-2333/credit_scoring.git
cd credit_scoring

# 2. 安装依赖
pip install -r requirements.txt

# 3. 一键运行完整建模流程
python main.py
```

运行后控制台输出完整建模日志，并在以下目录生成结果：
- `data/output/` — 评分卡 CSV（表1 明细、表2 决策）
- `outputs/figures/` — EDA 图表、ROC/KS 曲线、风险分层图

---

## 🔄 建模流程 / Pipeline

```
原始数据
  → EDA（缺失值 / 异常值 / 关键变量关系）
  → 数据划分（70/30 分层随机，seed=42）
  → 数据清洗 fit → 特征工程 fit（训练集）──► 测试集统一 apply
  → 特征分箱（训练集定边界，测试集套用）
  → WOE 编码 + IV 筛选 + 共线性剔除
  → 逻辑回归（L2 正则，max_iter=2000）
  → 评分卡转换（600/50 惯例）
  → 风险分层 + Cut-off 校准（30%/70% 分位）
  → 三档决策输出（通过 / 人工复核 / 拒绝）
```

---

## 🛠 技术栈 / Tech Stack

- **Python 3.8+**
- **pandas / numpy** — 数据处理
- **scikit-learn** — 逻辑回归、随机森林、模型评估
- **matplotlib / seaborn** — 可视化
- *可选*: xgboost / lightgbm — 梯度提升模型对比

---

## 👥 团队成员 / Authors

- 作者1 — 建模 / 代码实现
- 作者2 — （按需补充：数据分析 / 模型评估）
- 作者3 — （按需补充：文档 / 汇报）

---

## 📄 License

MIT License — 仅用于学习与作品集展示。数据来自 Lending Club 公开数据集（2007-2014）。
