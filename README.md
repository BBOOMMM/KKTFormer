# KKTFormer

KKTFormer 是一个面向资产配置的端到端决策模型。模型先用 Transformer 提取时间和资产交互特征，再通过可微投资组合优化器得到初始仓位及近似 KKT 状态；这些 primal-dual 信息会被编码为多头注意力偏置，最终由独立的 softmax allocation head 直接生成组合仓位。

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
[w, marginal risk, box duals, active set, constraint pressure]
        ↓
multi-head low-rank KKT attention bias
        ↓
KKT-conditioned Asset Attention
        ↓
refined hidden ──→ μ¹ (diagnostic)
        ↓
allocation logits
        ↓
temperature softmax
        ↓
weights         (B,H,N)
        ↓
realized returns (B,H,N)
        ↓
end-to-end CVaR loss
```

默认 `--cvar_variant sit` 与发布版 SIT 使用完全相同的训练目标：VaR 由
`torch.quantile` 计算且不 detach，tail excess 使用 `ReLU(L - VaR)`。原先的
detach-VaR + softplus 实现保留为 `--cvar_variant smooth` 消融；SIT 协议下交易成本
和换手惩罚不混入该训练 loss。

模型不使用收益预测 MSE 等 prediction loss。为防止无界 return head 通过任意放大
`mu` 把 probe QP 退化为线性目标和单资产 corner solution，probe optimizer 前默认执行
横截面 signal normalization：

`z = (s - mean(s)) / (std(s) + eps)`，
`mu = c * mean(diag(Sigma + eta I)) * z`。

因此 alpha 的尺度与实际二次风险项一致，而不是由神经网络输出幅度决定。默认固定较小的
`c=0.05`。最终 allocation logits 使用独立 head，不对风险缩放后的 `mu` 直接做 softmax，
避免仓位退化为近似等权。训练梯度来自真实资产收益、组合仓位和 CVaR 目标，并穿过
softmax policy、KKT-conditioned attention 和第一阶段 probe optimizer。

### 三路输入

每个 token 由三类特征融合：

```text
log-return path → Linear(60, log_return_embed_dim)
date features   → Linear(3, date_embed_dim)
asset identity  → Embedding(N, asset_embed_dim)

concat → Linear(log_return_embed_dim + date_embed_dim + asset_embed_dim, d_model)
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

当前主方法的优化问题包含 budget + box + quadratic risk，并可选加入二次换手和熵正则：

```text
min_w  0.5 wᵀQw - μᵀw + 0.5 ρ||w-w_prev||²
       + τ Σᵢ [(wᵢ+ε)log(wᵢ+ε) - εlog ε]
s.t.   1ᵀw = budget,  lower ≤ w ≤ upper
```

box dual 和 active-set Jacobian 来自同一个凸优化问题。熵梯度
`τ(log(w+ε)+1)` 进入 stationarity、dual 和 pressure；熵曲率
`τ/(w+ε)` 进入 active-set Jacobian 的局部 Hessian。factor/industry bounds、
L1 turnover cap 和 gross exposure 仍只属于 `--feedback_mode none` 的扩展能力，
在 `dual`/`jacobian` 下会被明确拒绝。交易成本只进入 realized decision loss 和
评估，不进入主方法的 probe/final optimizer。

第一阶段优化器产生的每资产 KKT token 包含：

- 初始仓位 `w`；
- 边际风险 `Qw`；
- 上下界对偶变量；
- 上下界 active-set 指示；
- 总约束压力。

即严格使用 7 维 token `z_i = [w_i, (Qw)_i, alpha_i, beta_i, I_i^L, I_i^U, p_i]`。raw factor exposure（market beta、momentum、volatility）不进入 KKT token，避免 `dual` 相对 `none` 获得额外市场特征。未来若主问题加入 factor constraints，应编码优化器给出的逐资产约束压力 `(A^T lambda)_i`，而不是原始暴露 `A_i`。

诊断量严格区分
`reduced_gradient = Qw - mu + rho*(w-w_prev) + tau*(log(w+eps)+1) + nu*1`
与完整的 `kkt_stationarity_residual = reduced_gradient - alpha + beta`。前者尚未计入
box dual，不能称为 stationarity residual。`test_diagnostics.csv` 会对每个
probe 解先在资产维计算无穷范数，再报告
`MeanProbeKKTStationarityResidualInf` 和
`MaxProbeKKTStationarityResidualInf`，用于检查默认 5 次 probe 迭代是否足以
支撑“近似 KKT state”的表述。

budget dual `nu` 在存在 free asset 时由 `gradient_i + nu = 0` 估计。当所有资产都
位于 box bound 时，该乘子不再唯一：lower-active 坐标要求
`nu >= -gradient_i`，upper-active 坐标要求 `nu <= -gradient_i`。实现取 admissible
interval 的中点作为规范代表；若区间只有单侧，则取有限端点，不再错误回退为 `nu=0`。

KKT token 经过每个 attention head 独立的低秩 query/key 投影，得到：

```text
kkt_bias: (B,H,num_heads,N,N)
```

它被加入 KKT-conditioned Asset Attention：

```text
attention_scores = QKᵀ / √d + γ_head × kkt_bias
```

默认 `--feedback_mode dual` 同时使用 primal-dual context 和 attention bias。
为排除额外 refinement attention 及参数量带来的混淆，提供以下结构匹配消融：

| 论文名称 | `--feedback_mode` | refinement attention | KKT context | KKT attention bias |
| --- | --- | :---: | :---: | :---: |
| OnePass Softmax | `none` | 否 | 否 | 否 |
| TwoPass-NoKKT | `two_pass` | 是 | 置零 | 置零 |
| KKT-Context | `context` | 是 | 是 | 置零 |
| KKT-Bias | `bias` | 是 | 置零 | 是 |
| KKT-Full | `dual` | 是 | 是 | 是 |

除 `none` 外的五种 two-pass 模式均实例化相同模块，参数量完全一致。
`two_pass` 仍执行 Probe→KKT→refinement，但在进入 refinement attention 前将
`decision_context` 和 `decision_bias` 都严格置零。`jacobian` 在 KKT-Full 上额外
加入 `∂wᵢ/∂μⱼ`，保留为高成本消融。

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
| Test context | SIT 原版完整重叠窗口；逐 token record/dedup |
| 测试调仓日 | SIT 发布代码中的固定调仓日期 |
| 日收益日期约定 | 日期 `t` 对应 `P[t+1]/P[t]-1` |

默认 `pool_30` 缓存规模为：

```text
train: 4197 samples
val:    674 samples
test:  1237 samples
```

测试集中有 1237 个完整 20 日 horizon context。评估按 SIT 原版顺序遍历每个窗口的全部 horizon token，日期已存在时跳过；因此首个窗口之后，每个新日期通常取自 `h=19`，保留了完整 causal temporal context。

当前缓存为每个样本的 H 个 token 分别保存与 rolling path 同时点的 `Sigma` 和 factor exposure；即 token `h` 使用价格窗口 `X_{t+h}` 估计 `Sigma_{t+h}` 与 `F_{t+h}`。只有初始 `w_prev` 等样本级静态量会广播到 H 个 token。每个 token 的 `μ` 和优化仓位独立，并通过 `B×H` 批量计算。测试时使用上述 record/dedup 后的 token 仓位，实际仓位仅在 SIT 固定调仓日更新。

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
| `Sigma` | `(H,N,N)` | 每个 token 同时点的滚动协方差 |
| `factor_exposure` | `(H,N,3)` | 每个 token 同时点的市场 beta、动量、波动率暴露 |
| `w_prev` | `(N,)` | 初始/上一期仓位 |
| `future_dates` | `(H,)` | 每个 horizon token 的执行日期 |

修改资产池、lookback、horizon、约束或交易成本后，应重新构建对应缓存。旧版含 `market_window`，或仅含单个 `(N,N)` `Sigma` / `(N,3)` factor exposure 的缓存与当前模型不兼容，loader 会拒绝加载。

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
  --log_return_embed_dim 32 \
  --date_embed_dim 32 \
  --asset_embed_dim 32 \
  --d_model 32 \
  --n_heads 4 \
  --num_layers 1 \
  --ff_dim 64 \
  --feedback_mode dual \
  --kkt_bias_rank 4 \
  --probe_optimizer_iterations 5 \
  --optimizer_iterations 100 \
  --loss_mode hybrid \
  --regret_weight 0.1 \
  --cvar_alpha 0.95 \
  --cvar_variant sit \
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
| `--log_return_embed_dim` | `32` | log-return path 经线性层后的维度 |
| `--date_embed_dim` | `32` | date feature 经线性层后的维度 |
| `--asset_embed_dim` | `32` | asset identity embedding 的维度 |
| `--d_model` | `32` | 三路特征融合后投影到的 Transformer 隐状态维度 |
| `--feedback_mode` | `dual` | `none`、`two_pass`、`context`、`bias`、`dual` 或 `jacobian`；依次对应 OnePass、TwoPass-NoKKT、KKT-Context、KKT-Bias、KKT-Full 和 Jacobian 消融 |
| `--kkt_bias_rank` | `4` | 每个 head 的 KKT bias 低秩维度 |
| `--probe_optimizer_iterations` | `5` | 第一阶段 probe optimizer 迭代数 |
| `--decision_layer` | `softmax` | 最终组合层；`optimizer` 保留为旧版消融 |
| `--temperature` | `1.0` | 最终 allocation softmax 的固定温度 |
| `--optimizer_iterations` | `10` | optimizer 消融或 hybrid oracle 的迭代数 |
| `--loss_mode` | `cvar` | `cvar` 或 `hybrid`（CVaR + decision regret） |
| `--regret_weight` | `0.1` | hybrid 中的 `lambda_regret`；应只用验证集选择 |
| `--cvar_alpha` | `0.95` | CVaR 置信水平 |
| `--cvar_variant` | `sit` | `sit` 完全复刻原版 quantile + ReLU；`smooth` 为消融 |
| `--cvar_temperature` | `1e-3` | 仅 `cvar_variant=smooth` 使用的平滑温度 |
| `--eta` | `1e-3` | 优化问题二次正则项 |
| `--signal_normalization` | `risk` | 对 return head 做横截面标准化并匹配风险尺度；`none` 仅用于消融 |
| `--signal_scale` | `0.05` | 风险尺度固定系数 `c`；建议做敏感性分析 |
| `--signal_normalization_epsilon` | `1e-6` | signal 标准化数值稳定项 |
| `--upper_bound` | `1.0` | 单资产仓位上界 |
| `--lower_bound` | `0.0` | 单资产仓位下界 |
| `--probe_upper_bound` | `0.1` | 仅 structural probe 使用的单资产上界，不约束最终 Softmax 仓位 |
| `--probe_lower_bound` | `0.0` | 仅 structural probe 使用的单资产下界 |
| `--turnover_penalty` | `0.0` | 二次换手惩罚 |
| `--entropy_regularization` | `0.0` | 熵正则强度 `tau`；支持所有 feedback modes |
| `--entropy_epsilon` | `1e-4` | 熵梯度和 Hessian 在零权重附近的平滑参数 |
| `--trade_cost_bps` | `0.0` | 测试交易成本；SIT 协议下不进入训练 loss 或 KKT QP |
| `--max_turnover` | disabled | 最大 L1 换手约束，仅 `feedback_mode=none` |
| `--gross_exposure_limit` | disabled | 总杠杆约束，仅 `feedback_mode=none` |
| `--factor_lower/upper` | disabled | 风格暴露上下界，仅 `feedback_mode=none` |
| `--industry_lower/upper` | disabled | 行业暴露上下界，仅 `feedback_mode=none` |

默认最终仓位由独立 allocation head 和 softmax 直接产生，只严格保证非负及权重和为 1。
probe optimizer 使用独立的 box bounds，因此可以通过较紧的
`--probe_upper_bound` 产生有信息量的 active set 和 upper-bound shadow price，
而不会触发 softmax 最终组合的硬约束校验。`test_diagnostics.csv` 额外报告
`ProbeActiveLowerRatio`、`ProbeActiveUpperRatio`、`MeanAbsProbeAlpha`、
`MeanAbsProbeBeta`、
`MeanProbePressure` 和 `ProbePressureAbove1e-6Ratio`。
单资产上下界、总换手、风格/行业暴露等硬约束实验需要切回优化器消融：

```bash
--decision_layer optimizer --optimizer_iterations 100
```

optimizer 模式会从 probe 仓位 warm start，并显著降低训练速度。默认 softmax 模式下，
probe 的所有 `B×H` 优化问题仍以一次张量批处理运行，不会在 Python 中逐 token 调用优化器。

### CVaR + 决策遗憾

设置 `--loss_mode hybrid` 后，训练目标为：

```text
L = L_CVaR + regret_weight * L_regret
```

每个 causal horizon token 对应一笔组合决策。训练 oracle 使用该 token 的真实
下一期资产收益，从预测权重的 detached 副本出发，在相同的风险矩阵、预算、
box/factor/industry/换手约束下再次求解。预测组合与 oracle 组合随后由相同的
realized portfolio objective 评价；oracle objective 被 detach，因此梯度只通过
预测 softmax 组合流回预测网络。oracle 只在训练、验证和诊断中使用，不进入
测试时的决策输入。

例如：

```bash
--loss_mode hybrid --regret_weight 0.1
```

`L_regret` 与 CVaR 的数值尺度可能不同，建议只根据验证集搜索
`regret_weight`，例如 `0.01/0.05/0.1/0.5/1.0`，不要按测试 Sharpe 选择。

## 扩展约束示例

扩展优化器可限制单资产仓位、总换手和风格暴露；这些实验必须关闭 KKT feedback：

```bash
python run_kkt.py \
  --context_root ./portfolio_context_cache \
  --data_pool 30 \
  --upper_bound 0.10 \
  --feedback_mode none \
  --decision_layer optimizer \
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
├── exp/exp_main_kkt.py            # KKT-feedback + policy 训练和 SIT 对齐评估
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

SIT 使用 signature path interaction 作为 attention bias，并通过 softmax 仓位和真实收益优化 CVaR。KKTFormer 使用可微优化器产生的 primal-dual 最优性状态作为 cross-asset attention bias，再由独立的 softmax policy 根据该状态修正后的表示生成仓位并优化 CVaR。

两者对比的核心约束是：数据切分、lookback、horizon、测试调仓日、交易成本和评估指标保持一致；输入表征、优化模块、约束设计和 KKT attention 属于 KKTFormer 的方法差异。

## License

本项目沿用仓库中的 MIT License，详见 `LICENSE`。
