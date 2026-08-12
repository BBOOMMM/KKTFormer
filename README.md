# KKTFormer

KKTFormer 是一个面向资产配置的端到端决策模型。模型先用 Transformer 提取时间和资产交互特征，再通过可微投资组合优化器得到初始仓位及 KKT 状态；这些 primal-dual 信息会被编码为多头注意力偏置，用于修正资产注意力和最终仓位。

本仓库同时保留了上游 Signature-Informed Transformer（SIT）的部分代码，用于公平对照。KKTFormer 的实验协议默认向 SIT 对齐，但模型输入、KKT 模块、约束和可微优化策略属于 KKTFormer。

> 当前代码是研究实现。默认主入口是 `run_kkt.py`，不是上游 SIT 使用的 `run.py`。

## 方法概览

KKTFormer 的完整前向过程为：

```text
log_return_path (B,H,N,60)
date_feats      (B,H,3)
asset embedding
        ↓
Temporal Attention + Asset Attention
        ↓
μ⁰              (B,H,N)
        ↓
B×H batched differentiable probe optimizer
        ↓
[w, marginal risk, stationarity, dual, active set, factor exposure]
        ↓
multi-head low-rank KKT attention bias
        ↓
KKT-conditioned Asset Attention
        ↓
μ¹              (B,H,N)
        ↓
warm-start batched differentiable optimizer
        ↓
weights         (B,H,N)
        ↓
realized returns (B,H,N)
        ↓
end-to-end CVaR loss
```

模型不使用收益预测 MSE 等 prediction loss。训练梯度来自真实资产收益、组合仓位和 CVaR 目标，并穿过最终优化器、KKT-conditioned attention，以及第一阶段 probe optimizer。

### 三路输入

每个 token 由三类特征融合：

```text
log-return path → Linear(60, d_model)
date features   → Linear(3, d_model)
asset identity  → Embedding(N, d_model)

concat → Linear(3 × d_model, d_model)
```

- `log_return_path`：`(B,H,N,W)`，默认 `W=60`。
- `date_feats`：`(B,H,3)`，包含工作日频率下的星期、月内日期和年内日期特征。
- `asset identity`：模型内部根据资产序号生成。

为匹配 SIT 的 60 个价格点路径，每条 60 维 log-return path 由一个起始零和 59 个实际 log return 构成：

```text
[0, log(P₂/P₁), ..., log(P₆₀/P₅₉)]
```

第 `h` 个 horizon token 使用截至其执行日前一日的滚动历史路径。Temporal Attention 使用因果 mask，因此较早 token 无法读取较晚 token。

### KKT attention bias

第一阶段优化器产生的每资产 KKT token 包含：

- 初始仓位 `w`；
- 边际风险 `Qw`；
- stationarity residual；
- 上下界对偶变量；
- 上下界 active-set 指示；
- 总约束压力；
- 风格因子暴露。

KKT token 经过每个 attention head 独立的低秩 query/key 投影，得到：

```text
kkt_bias: (B,H,num_heads,N,N)
```

它被加入 KKT-conditioned Asset Attention：

```text
attention_scores = QKᵀ / √d + γ_head × kkt_bias
```

默认 `--feedback_mode dual` 使用高效的 primal-dual KKT bias。`--feedback_mode jacobian` 会额外加入 `∂wᵢ/∂μⱼ`，主要用于高成本消融；`--feedback_mode none` 是不使用 KKT feedback 的基线。

## SIT 对齐协议

使用 `--protocol sit` 时，以下设置与发布版 SIT 对齐：

| 项目 | 设置 |
| --- | --- |
| 训练集 | 2000–2016 |
| 验证集 | 2017–2019 |
| 测试集 | 2020–2024 |
| Lookback | 60 个价格点 |
| Horizon | 20 个交易日 |
| Train/Val context | 日频滑动样本 |
| Test context | 覆盖 SIT 的有效日度预测日期网格 |
| 测试调仓日 | SIT 发布代码中的固定调仓日期 |
| 日收益日期约定 | 日期 `t` 对应 `P[t+1]/P[t]-1` |

默认 `pool_30` 缓存规模为：

```text
train: 4197 samples
val:    674 samples
test:  1256 samples
```

测试集中有 1237 个完整 20 日 horizon context；最后 19 个不完整 context 用于补齐 SIT 的有效日度预测和回测日期范围，不进入完整 horizon 测试损失均值。

当前缓存为每个样本保存一个决策时点的 `Sigma`、factor exposure 和初始 `w_prev`。训练时它们广播到 H 个 token；每个 token 的 `μ` 和优化仓位仍然独立，并通过 `B×H` 批量计算。测试时，每个日度 context 的第一个 token 对应该 context 的即时执行日期，实际仓位仅在 SIT 固定调仓日更新。

## 环境安装

本地开发使用 conda 的 `alden` 环境：

```bash
cd /home/alden/SIT/src/KKTFormer
conda activate alden
pip install -r requirements.txt
```

主要依赖包括 PyTorch、NumPy、pandas、scikit-learn、matplotlib 和 joblib。GPU 不是必需的，但可微优化和 `B×H` 批处理建议使用 CUDA。

## 构建 portfolio context

先从价格 CSV 构建 KKTFormer 的 train/val/test 缓存：

```bash
cd /home/alden/SIT/src/KKTFormer

python 2_build_portfolio_context.py \
  --data_path ./asset_data/full_dataset.csv \
  --output_dir ./portfolio_context_cache/pool_30 \
  --data_pool 30 \
  --lookback_window 60 \
  --horizon 20 \
  --protocol sit
```

每个 `.npz` split 的主要字段为：

| 字段 | 单样本形状 | 用途 |
| --- | --- | --- |
| `log_return_path` | `(H,N,60)` | Transformer 路径输入 |
| `date_feats` | `(H,3)` | 时间特征 |
| `future_returns` | `(H,N)` | 端到端 CVaR 的真实收益 |
| `Sigma` | `(N,N)` | 决策时点协方差 |
| `factor_exposure` | `(N,3)` | 市场 beta、动量、波动率暴露 |
| `w_prev` | `(N,)` | 初始/上一期仓位 |
| `future_dates` | `(H,)` | 每个 horizon token 的执行日期 |

修改资产池、lookback、horizon、约束或交易成本后，应重新构建对应缓存。旧版含 `market_window` 的缓存与当前模型不兼容，loader 会拒绝加载。

## 训练和评估

构建缓存后运行：

```bash
python run_kkt.py \
  --model_id kkt_dp30 \
  --protocol sit \
  --root_path ./asset_data/ \
  --data_path full_dataset.csv \
  --context_root ./portfolio_context_cache \
  --data_pool 30 \
  --window_size 60 \
  --horizon 20 \
  --d_model 32 \
  --n_heads 4 \
  --num_layers 1 \
  --ff_dim 64 \
  --feedback_mode dual \
  --kkt_bias_rank 4 \
  --probe_optimizer_iterations 5 \
  --optimizer_iterations 10 \
  --loss_mode cvar \
  --cvar_alpha 0.95 \
  --batch_size 64 \
  --train_epochs 10 \
  --gpu 0
```

`run_kkt.py` 会依次训练、加载最佳 checkpoint，并在 test split 上评估。结果默认写入：

```text
checkpoints_kkt/
results_kkt/
```

### 重要参数

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `--feedback_mode` | `dual` | `none`、`dual` 或 `jacobian` |
| `--kkt_bias_rank` | `4` | 每个 head 的 KKT bias 低秩维度 |
| `--probe_optimizer_iterations` | `5` | 第一阶段 probe optimizer 迭代数 |
| `--optimizer_iterations` | `10` | warm-start 最终优化器迭代数 |
| `--loss_mode` | `cvar` | 当前 sequence KKTFormer 只允许 CVaR |
| `--cvar_alpha` | `0.95` | CVaR 置信水平 |
| `--cvar_temperature` | `1e-3` | CVaR tail excess 的平滑温度 |
| `--eta` | `1e-3` | 优化问题二次正则项 |
| `--upper_bound` | `1.0` | 单资产仓位上界 |
| `--lower_bound` | `0.0` | 单资产仓位下界 |
| `--turnover_penalty` | `0.0` | 二次换手惩罚 |
| `--trade_cost_bps` | `0.0` | SIT 兼容的测试交易成本 |
| `--max_turnover` | disabled | 最大 L1 换手约束 |
| `--gross_exposure_limit` | disabled | 总杠杆约束 |
| `--factor_lower/upper` | disabled | 风格暴露上下界 |
| `--industry_lower/upper` | disabled | 行业暴露上下界 |

最终优化器默认使用 10 次迭代，并从 probe 仓位 warm start。更高迭代数可用于收敛敏感性实验，例如：

```bash
--optimizer_iterations 100
```

但这会显著降低训练速度。默认配置下所有 `B×H` 优化问题均以一次张量批处理运行，不会在 Python 中逐 token 调用优化器。

## 约束示例

限制单资产仓位、总换手和风格暴露：

```bash
python run_kkt.py \
  --context_root ./portfolio_context_cache \
  --data_pool 30 \
  --upper_bound 0.10 \
  --max_turnover 0.50 \
  --factor_lower=-0.2,-0.2,-0.2 \
  --factor_upper=0.2,0.2,0.2 \
  --loss_mode cvar
```

构建缓存和训练时的资产数、上下界及 factor/industry 配置应保持一致。

## 仓库结构

```text
KKTFormer/
├── 2_build_portfolio_context.py   # 构建 point-in-time context 缓存
├── run_kkt.py                     # KKTFormer 训练与评估入口
├── model/KKTFormer.py             # Seq2seq backbone 与 KKT attention
├── portfolio/context_builder.py   # 路径、协方差、因子和目标构造
├── portfolio/optimizer_layer.py   # 批量可微投资组合优化器
├── portfolio/kkt_feedback.py      # primal-dual/Jacobian 状态提取
├── portfolio/losses.py            # 端到端 CVaR 与其他决策损失工具
├── exp/exp_main_kkt.py            # 双阶段训练和 SIT 对齐评估
├── data_provider/                 # Dataset、cache loader 和 DataLoader
├── utils/sit_protocol.py          # SIT split 与固定调仓日
├── run.py                         # 保留的上游 SIT 入口
└── model/SIT.py                   # 保留的上游 SIT 模型
```

## 测试

仓库的测试目录在当前项目配置中可能被 Git 忽略，但本地可以运行完整测试：

```bash
conda run -n alden python -m unittest discover -s tests
```

也可以进行基础语法检查：

```bash
python -m py_compile \
  model/KKTFormer.py \
  portfolio/kkt_feedback.py \
  portfolio/losses.py \
  exp/exp_main_kkt.py
```

## 与 SIT 的关系

SIT 使用 signature path interaction 作为 attention bias，并通过 softmax 仓位和真实收益优化 CVaR。KKTFormer 使用可微优化器产生的 primal-dual 最优性状态作为 cross-asset attention bias，并通过受约束仓位和真实收益优化 CVaR。

两者对比的核心约束是：数据切分、lookback、horizon、测试调仓日、交易成本和评估指标保持一致；输入表征、优化模块、约束设计和 KKT attention 属于 KKTFormer 的方法差异。

## License

本项目沿用仓库中的 MIT License，详见 `LICENSE`。
