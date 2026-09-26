# GLM-5.2 / DeepSeek-V3.2 (DSA) 模型结构学习笔记

> 结论先行：vLLM 中 GLM-5.2（`model_type = "glm_moe_dsa"`）没有独立模型实现，
> 完全复用 DeepSeek-V3.2 的 DSA（DeepSeek Sparse Attention）架构代码，
> 仅叠加了少量 GLM-5.2 专属的内核优化。
> 学会 `vllm/models/deepseek_v32/` 就等于学会了 GLM-5.2 的推理实现。

**本文结构**：
第一部分·预备知识（零基础，没接触过 AI infra 也能读懂）→
第二部分·模型结构（代码地图 + 结构图 + 前向流程）→
第三部分·深入篇（逐块拆解到 kernel 级）→
第四部分·学习路径与自测。

---

# 第一部分：预备知识（零基础必读）

## 0.1 一次 LLM 推理到底发生了什么

LLM 生成文字是**一个 token 接一个 token** 的循环：

```
"你好吗" ──► 模型 ──► P(下一token分布) ──► 采样"很" ──►
"你好吗很" ──► 模型 ──► ...重复
```

每次循环分两种模式（后文反复出现）：

- **prefill（预填充）**：一次吃进整段 prompt（T 个 token），并行算完，
  T 大、计算密集；
- **decode（解码）**：每次只新增 1 个 token（T=1），但要"回看"全部历史，
  访存密集。

模型内部 = N 层相同的 DecoderLayer 堆叠，每层 = 注意力 + MLP。
token 先变成向量（embedding），逐层加工，最后一层的向量过 `lm_head`
（一个大矩阵乘）得到词表上每个词的概率。

## 0.2 Attention 速成（只需要一个公式）

每个 token 的向量去"查询"其他 token 的向量，加权汇总信息：

```
score(i,j) = q_i · k_j / √d          ← token i 对 token j 的关注度
输出 i     = Σ_j softmax_j(score) · v_j
```

- q（query）、k（key）、v（value）都是每个 token 的向量，
  由本 token 的 hidden 向量乘三个权重矩阵得到；
- 直觉：q 是"我想找什么"，k 是"我能提供什么"，v 是"我的内容"；
- 多头 = 并行做 h 组上面的事，每组（head）有自己的小维度，
  最后拼接。vLLM 代码里 `Nh` = 头数，`d_n`/`d_r` = 每头维度。

## 0.3 KV Cache：AI infra 的核心矛盾

decode 时历史 token 的 k、v **完全不变**，于是把它们缓存起来，
每步只算新 token 的 q——这就是 **KV cache**。

但它非常大。粗算（FP16，每 token）：

```
MHA:  2(K和V) × Nh × 每头维度 × 2字节 × 层数
      例: 64头 × 128维 × 2 × 2B × 60层 ≈ 1.9 MB/token
      128K 上下文 × 并发几十请求 → 数百 GB，显存爆炸
```

**而且 decode 是访存瓶颈**：每生成 1 个 token 都要把整个 cache 从
显存读一遍，带宽利用率决定吞吐。所以"把 cache 压小"是模型结构
设计的头号驱动力——MLA 就是为此而生（§0.4）。

## 0.4 五代注意力演进：为什么长成今天这样

用一个贯穿全文的算例（量级取自 DeepSeek-V3.2 一类模型）：

> 假设：Nh=128 个注意力头，每头 d=128 维，60 层，FP16 存储（2 字节/元素）。
> 序列长度按 128K 上下文算。

### 第 0 代：MHA —— 一切的原点

每个头有自己独立的 K、V，全部缓存：

```
每 token 每层 = 2(K和V) × 128头 × 128维 × 2B = 64 KB
每 token 全模型 = 64 KB × 60层            ≈ 3.84 MB
128K 上下文一条请求                      ≈ 503 GB   ← 根本放不下
```

质量最好，但显存和带宽都爆炸。**问题：cache 太大。**

### 第 1 代：MQA —— 砍头数（所有头共用 1 组 KV）

128 个 query 头，但 K/V 只有 1 组：

```
每 token 每层 = 2 × 1 × 128 × 2B = 512 B   (÷128)
128K 上下文 ≈ 3.9 GB                        ← 显存解决了
```

**问题：质量明显下降**（头与头看东西的角度不同，全共享太粗暴）。

### 第 2 代：GQA —— 折中（分组共享）

Llama 系列的选择：128 个头分成 8 组，每组共享一组 KV：

```
每 token 每层 = 2 × 8 × 128 × 2B = 4 KB     (÷16)
128K 上下文 ≈ 31 GB                          ← 还是太大
```

质量比 MQA 好，但 cache 仍随头数/组数线性增长。
**问题：共享是"硬"共享，表达力受限；显存仍不够省。**

### 第 3 代：MLA —— 压缩而非共享

换个思路：与其让头们**共用**同一组 KV，不如承认每头的 K/V 都来自
一个**低秩的公共潜向量** `c`（512 维），再各自线性变换出来：

```
第 j 个 token 的第 h 头:  k_j^h = W_UK^h · c_j ,  v_j^h = W_UV^h · c_j
                          （每头一个专属小变换,不再是"共用"）
```

关键洞察（结合律）：既然 K/V 都是 `c` 的线性函数，那
**只缓存 `c` 本身，变换可以挪到算的时候再做**——

- attention 打分：`q·k_j^h = q·(W_UK^h c_j) = (W_UK^h 的转置 q)·c_j`
  → 把 W_UK 提前乘进 q（"Q 侧吸收"）；
- 输出汇总：`Σ a_j · v_j^h = W_UV^h · (Σ a_j · c_j)`
  → 先在潜空间加权求和，再变回 V 维度（"输出侧吸收"）。

（细节和代码见 §11。）

cache 从此与头数无关：

```
每 token 每层 = (512潜向量 + 64 RoPE位置) × 2B = 1152 B   (vs MHA ÷56)
128K 上下文 ≈ 8.8 GB        ← 且质量与 MHA 相当(DeepSeek-V2 论文)
```

代价：query 从 128 维膨胀到 512 维参与打分、输出多一次小矩阵乘
——**用 FLOPs 换带宽**，而 decode 恰恰是带宽瓶颈，稳赚。

**问题：还剩最后一个——q 仍要和 cache 里全部历史 token 做点积，
计算量随上下文线性涨，128K 时 decode 每步都是全量扫描。**

### 第 4 代：DSA —— 计算量也砍（本文主角）

在每层加一个**闪电索引器（Lightning Indexer）**：一组便宜的小头
（64 个 × 128 维，FP8 打分，MQA 形式）给每个历史 token 算一个
相关性分，选出 **top-k（2048）** 个：

```
注意力计算量:  O(128K 全量) → O(2048)      (÷64, 与上下文长度解耦)
KV cache:      与 MLA 相同(576 元素)        (不多花)
额外成本:      一个小 MQA + top-k 选择      (可忽略)
```

索引器为什么便宜：① 64 个小头远小于主注意力的 128 个大头；
② 全程 FP8；③ 它只"挑人不下注"——选错了最多让注意力少看几个
token，不直接污染输出。

**五代对比总表**（同上算例，128K 上下文单请求）：

| 代际 | cache/token/层 | 128K 总 cache | decode attention 计算量 | 质量 |
|---|---|---|---|---|
| MHA | 64 KB | ~503 GB | O(128K) | 基线 |
| MQA | 512 B | ~3.9 GB | O(128K) | 明显掉点 |
| GQA(8组) | 4 KB | ~31 GB | O(128K) | 小幅掉点 |
| **MLA** | 1.15 KB | **~8.8 GB** | O(128K) | ≈ MHA |
| **DSA** | 1.15 KB | ~8.8 GB | **O(2048)** | ≈ MHA |

> 数字是量级示意（忽略 FP8 量化、页对齐、PCP/DCP 切分等工程因素），
> 但相对关系是真实的：MLA 的 cache 优势 + DSA 的计算解耦，
> 就是 DeepSeek-V3.2/GLM-5.2 能又快又省地伺候长上下文的全部秘密。

DeepSeek-V3.2 / GLM-5.2 = **MLA + DSA + MoE + MTP**。

## 0.5 MoE 速成

MLP 换成"专家团"：一个大 FFN 拆成 E 个小 FFN（专家），
每个 token 由路由器（gate）挑 top-k 个专家处理，其余专家不激活：

- 总参数量大（知识多），每 token 实际算的 FLOPs 小（top-k）；
- `gate` 是一个小矩阵 `[H → E]`，输出打分；
- `shared_experts`：每个 token 都必经的公共专家；
- 工程难点：专家分布在多卡上（EP），路由结果要跨卡搬运。

## 0.6 残差流与 RMSNorm（读懂 vLLM 代码的钥匙）

**残差流**：每层的输出 = 输入 + 本层加工结果（`x + f(x)`），
梯度/信息可以"抄近道"直达输出。vLLM 代码里这个"抄近道的通道"
被显式拆成两个变量：

```python
hidden_states, residual = norm(hidden_states, residual)
#                                            └── 上上层的输出,等着被加
# 返回值: (norm(x + residual), x + residual)
#         └── norm 后的新输入        └── 新残差,传给下一层
```

**为什么拆开**：残差加法被推迟、并与 all-reduce 通信融合
（§17），最后一层才真正相加。第一次读 vLLM 模型代码最懵的就是
这个 `(hidden_states, residual)` 元组——记住它即可。

**RMSNorm**：LayerNorm 的简化版（不减均值，只除均方根），
现代大模型标配，`eps` 是防除零的小数。

## 0.7 vLLM 代码阅读须知（抽象翻译表）

第一次读 vLLM 模型代码，卡住的不是模型数学而是工程抽象。
对照下表即可：

| 代码里看到 | 意思 |
|---|---|
| `ColumnParallelLinear` | 权重按**输出列**切到各 TP 卡，每卡算一部分输出 |
| `RowParallelLinear` | 权重按**输入行**切，每卡输出是部分和；`reduce_results=False` 表示**不**做卡间求和（通信延后，§17） |
| `ReplicatedLinear` | 不切分，每卡放全量（indexer 用它：top-k 下标必须全卡一致） |
| `MergedColumnParallelLinear` | 几个列切权重**拼成一次 GEMM**（如 fused_qkv_a_proj 同时投影 Q 的低秩和 KV） |
| `CustomOp` | 平台分派算子：`forward_cuda` / `forward_hip` / `forward_native` |
| `forward_context` / `slot_mapping` | 引擎传给每层的"本步上下文"：每个 token 该写进 cache 的哪个槽位 |
| `get_kv_cache_spec` / `bind_kv_cache` | 层声明"我的 cache 长什么样"，引擎统一分配显存后绑回来 |
| `PPMissingLayer` / `make_layers` | 流水线并行(PP)时本卡不含的层用占位符；`make_layers` 只建本卡区间 |
| `weight_loader` / `stacked_params_mapping` | checkpoint 参数名 → vLLM 参数名的搬运映射 |
| `eager_break_during_capture` | breakable CUDA graph 的"断点"：捕获时在此回到 eager（§15） |
| `IntermediateTensors` | PP 相邻卡之间传递的 hidden/residual 包裹 |
| `topk_indices_buffer` | 一块所有层共享的 int32 buffer，indexer 写下标、稀疏注意力读 |

**分布式缩写**：TP=张量并行（一层权重切开）｜EP=专家并行（专家分卡）｜
PP=流水线（按层切）｜DP=数据并行（整模型复制）｜SP=序列并行（token 分片）｜
PCP/DCP=prefill/decode 上下文并行（超长序列再切）。

**名词表**：GEMM=通用矩阵乘（GPU 的主粮）｜RoPE=旋转位置编码
（把 token 位置旋进 q/k）｜FP8=8bit 浮点（显存/带宽减半，需配 scale）｜
ue8m0=只存 2 的幂的 scale 格式｜CUDA graph=预录制 kernel 序列回放，
消除启动开销｜MTP=多 token 预测（用小草稿头一次猜多个 token）。

---

# 第二部分：模型结构

## 1. 代码地图（先建立空间感）

```
vllm/models/deepseek_v32/            ← DSA 专属目录（本笔记主角）
├── __init__.py                      ← 平台分发入口
├── attention.py                     ← DeepseekV32Attention + Indexer（DSA 核心）
├── common/kernels.py                ← Triton 融合算子: fused_norm_rope / fused_q / fused_eh_norm
├── nvidia/model.py                  ← CUDA 路径: DecoderLayer / Model / ForCausalLM
├── nvidia/mtp.py                    ← CUDA 路径: MTP 投机草稿模型
├── nvidia/glm52_low_latency_gemm.py ← GLM-5.2 专属: 低延迟 GEMM 选择
├── nvidia/ops/fused_q_cutedsl.py    ← CuTe DSL 的 indexer Q 融合算子
└── amd/…                            ← ROCm 路径（结构同构，内核不同）

vllm/model_executor/models/deepseek_v2.py   ← 祖先基类（~2000 行）
├── DeepseekV2ForCausalLM / DeepseekV2Model  ← 编排层（logits/加载/MoE 元数据）
├── DeepseekV2MoE / DeepseekV2MLP            ← MoE 与稠密 MLP
├── DeepseekV2MLAAttention                   ← MLA 通用实现（v32 走另一条路径）
├── Indexer / DeepseekV32IndexerCache        ← 通用 Indexer（供非 CUDA 平台用）
└── GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM)  ← 非 CUDA 平台上的 GLM-5.2 别名
```

注册表（`vllm/model_executor/models/registry.py`）：

| 架构名 | 实现位置 |
|---|---|
| `DeepseekV32ForCausalLM` | `vllm.models.deepseek_v32`（registry.py:94） |
| `GlmMoeDsaForCausalLM` | `vllm.models.deepseek_v32`（registry.py:122） |
| `DeepseekV32MTPModel` | `vllm.models.deepseek_v32` 的 `DeepseekV32MTP`（registry.py:658） |

> 读代码的入口规律：vLLM 里任何模型都从 `ForCausalLM` 类进入
> （HuggingFace `config.json` 的 `architectures` 字段按 registry 找到它），
> `ForCausalLM.model` 是层堆叠，`forward` 只管把 hidden 传过各层，
> `compute_logits` 出概率，`load_weights` 搬权重。

平台分发（`deepseek_v32/__init__.py:17-29`）：
- CUDA → `nvidia/model.py` 的 DSA 实现，同时把 `DeepseekV32ForCausalLM`
  和 `GlmMoeDsaForCausalLM` 都指到它；
- 其他平台（ROCm/XPU/CPU）→ 退回 `deepseek_v2.py` 的通用 `DeepseekV3ForCausalLM`。

## 2. 三个前置概念（v32 专属）

### 2.1 MLA 复习

标准 MHA 的 KV cache 随头数膨胀。MLA 把 KV 压缩进一个低秩**潜向量**：

- 输入 hidden → `fused_qkv_a_proj`（一次 GEMM 同时产出 q 的低秩压缩 `q_c`
  和共享潜向量 `kv_c` + 位置分量 `k_pe`）
- 缓存时只存 `[kv_c | k_pe]`（`kv_lora_rank + qk_rope_head_dim` = 576 维），
  远小于全量 K/V
- 用时通过两个预变换矩阵"吸收"权重（§11 详解）：
  - `W_UK_T`：把 nope 部分的 query 投到潜空间（`q_nope @ W_UK_T` → `ql_nope`）
  - `W_UV`：把注意力输出从潜空间投回 V 空间（`attn_out @ W_UV`）

这样注意力计算变成对潜空间的 MQA（所有头共享一份 KV），cache 极小。

### 2.2 DSA 与 Lightning Indexer

MLA 解决 cache 大小，DSA 解决**计算量**：长序列时每个 query 仍要
attend 所有历史 token。DSA 在每层加一个 **Lightning Indexer**：

1. 用一组小 head（64 个 × 128 维，独立于主 MLA 的头）为每个历史
   token 打相关性分数；
2. 取分数最高的 `index_topk` 个 token 的下标写入
   `topk_indices_buffer`（全模型共享一块，`nvidia/model.py:185`）；
3. 稀疏 MLA 只 attend 这 top-k 个 token。

即：indexer 便宜（MQA、FP8、只挑不下注），MLA 昂贵但只算 top-k。

### 2.3 权重复用关系（GLM-5.2 为什么"免费"）

`DeepseekV32ForCausalLM` 的 docstring（`nvidia/model.py:416`）：
"Serves DeepSeek V3.2 and any architecture reusing DSA (e.g. GLM-5.2)."

GLM-5.2 checkpoint 的 `model_type` 是 `glm_moe_dsa`，前向逻辑与
DeepSeek-V3.2 完全一致，vLLM 只在构造后按 model_type 开启
GLM-5.2 专属内核（§8）。

## 3. 模块层级结构图（以 GLM-5.2 为例，H=6144）

符号：H=hidden(6144)，qkr=q_lora_rank(2048)，kvr=kv_lora_rank(512)，
d_r=qk_rope(64)，d_n=qk_nope，Nh=注意力头数，
E=路由专家数，k=每 token top-k 专家数，topk=index_topk。
括号内为 GLM-5.2 形状（由 `glm52_low_latency_gemm.py` 的 GEMM 维度反推）。

```
DeepseekV32ForCausalLM  (= GlmMoeDsaForCausalLM)
│
├── model : DeepseekV32Model
│   ├── embed_tokens : VocabEmbedding            [vocab, H]  (与首层 norm 融合)
│   ├── topk_indices_buffer                        [T, topk] int32  ← 全模型共享
│   │
│   ├── layers : DeepseekV32DecoderLayer × N
│   │   ├── input_layernorm : RMSNorm(H)          ◄── 融合上一层未约简的 all-reduce
│   │   │
│   │   ├── self_attn : DeepseekV32Attention      ┐
│   │   │   ├── fused_qkv_a_proj  [H → qkr+kvr+d_r]   (6144→2624)  一次 GEMM
│   │   │   ├── q_a_layernorm     RMSNorm(qkr)
│   │   │   ├── kv_a_layernorm    RMSNorm(kvr)
│   │   │   ├── q_b_proj          [qkr → Nh·(d_n+d_r)]
│   │   │   ├── kv_b_proj         [kvr → Nh·(d_n+v_dim)] ──重排──► W_UK_T / W_UV
│   │   │   ├── o_proj            [Nh·v_dim → H]  (reduce_results=False)
│   │   │   ├── rotary_emb        (RoPE, d_r)
│   │   │   └── indexer : DeepseekV32Indexer      ┐ DSA 专属
│   │   │       ├── wq_b          [qkr → 64·128]  │ (Replicated, 不切TP)
│   │   │       ├── wk_weights_proj [H → 128+64]  │ (融合 GEMM)
│   │   │       ├── k_norm        LayerNorm(128)  │
│   │   │       ├── indexer_rope_emb (interleave) │
│   │   │       └── k_cache       FP8+scale 缓存  ┘ (uint8, 独立于 MLA cache)
│   │   │
│   │   ├── post_attention_layernorm : RMSNorm(H)
│   │   │
│   │   └── mlp
│   │       ├── [浅层 < first_k_dense_replace]  DeepseekV2MLP
│   │       └── [深层] DeepseekV2MoE
│   │           ├── gate          [H → E]  (+e_score_correction_bias, noaux_tc)
│   │           ├── experts       FusedMoE (E 个专家, top-k 路由, 组内选择)
│   │           └── shared_experts DeepseekV2MLP (n_shared_experts>0 时)
│   │                                            输出不做 all-reduce ──┐
│   └── norm : RMSNorm(H)  ◄── 融合最后一层的 all-reduce              │
│                                                                    │
├── lm_head : ParallelLMHead       [H → vocab]                      │
└── logits_processor                                                │
        残差流: 每层返回 (mlp输出未约简, residual) ────────────────────┘

配套 MTP 草稿模型 (DeepseekV32MTP, 每层独立实例)
DeepseekV32MultiTokenPredictorLayer
├── enorm / hnorm : RMSNorm(H)
├── eh_proj       [2H → H]  (GLM-5.2 走低延迟 GEMM 计划)
├── mtp_block     : DeepseekV32DecoderLayer ×1  (含完整 DSA 注意力+MoE)
└── shared_head   : SharedHead(RMSNorm + lm_head)
```

三个 cache 主体（decode 时显存大头）：

| Cache | 形状/元素 | 用途 |
|---|---|---|
| MLA 潜向量缓存 | `[kvr + d_r]` per token (512+64=576) | 稀疏 MQA 的 K/V 来源 |
| Indexer K 缓存 | `[128 + 128/128×4]` uint8 per token | Lightning Indexer 打分用 |
| topk_indices_buffer | `[T, topk]` int32，非缓存、全层共享 | indexer→MLA 的下标接口 |

### 3.0.1 前向数据流（逐步带形状）

以 T 个 token 为例（decode 时 T=batch 内每请求 1 个）；
d_n=128、v_dim=128 按 DeepSeek-V3.2 惯例，其余为 GLM-5.2 形状。

```
input_ids [T]  （或 inputs_embeds [T,H])
   │ embed_tokens 查表(+首层 RMSNorm 融合)
   ▼
hidden [T, 6144] ─────────────────────────── residual = hidden(未 norm)
   │
   │ ① fused_qkv_a_proj  GEMM [6144→2624]
   ▼
qkv_lora [T, 2624]
   ├─ split → q_c  [T, 2048]     Q 的低秩压缩
   │         kv_c [T,  512]      共享潜向量(全序列每 token 一份)
   │         k_pe [T,   64]      RoPE 位置分量
   │
   │ ② fused_norm_rope (1 个 Triton kernel)
   │    q_a RMSNorm + kv_a RMSNorm + k_pe RoPE
   │    旁路: index_k = LayerNorm+RoPE(hidden 前128维)
   │    旁路: kv_c/k_pe 写入 MLA cache, index_k 写入 indexer FP8 cache
   ▼
q_c' [T,2048]   cache 新增 [T,576] 潜向量 + [T,144] uint8 indexer K
   │
   │ ③ q_b_proj  GEMM [2048 → Nh_local·192]
   ▼
q [T, Nh_local, 192] ─split─ q_nope [T,Nh,128] + q_pe [T,Nh,64]
   │
   │ ④ 权重吸收  ql_nope = bmm(q_nope, W_UK_T[Nh,128,512])
   ▼
ql_nope [T, Nh, 512]                    (进入潜空间,与 kv_c 同维度)
   │
   │ ⑤ indexer.wq_b [2048→8192] → index_q [T,64,128]
   │ ⑥ fused_q (Triton): q_pe RoPE + index_q RoPE + FP8 量化
   │    → mqa_q [T,Nh,64](FP8 可选) + index_q_fp8 [T,64,128] + index_weights [T,64]
   │
   │ ⑦a sparse_attn_indexer: index_q_fp8×indexer_cache_K(128维,FP8)
   │      → 每个位置 64 头打分平均 → topk 下标
   ▼
topk_indices_buffer [T, topk]           (int32, 写给本层稀疏 MLA 用)
   │
   │ ⑦b 稀疏 MQA: query=[T,Nh,576](ql_nope‖mqa_q)
   │      key/value = MLA cache 按 topk 下标收集 [topk,576](全头共享)
   │      (短 prefill 退化为稠密 MHA; DCP/PCP 在此跨卡合并)
   ▼
attn_out [T, Nh, 512]                   (潜空间表示)
   │
   │ ⑧ 输出吸收  bmm(attn_out, W_UV[Nh,512,128])
   ▼
out [T, Nh·128]
   │
   │ ⑨ o_proj  GEMM [Nh·128 → 6144]  (reduce_results=False)
   ▼
attn [T, 6144] ──(+residual 延迟到下个 norm 融合)──┐
   │                                              │
   │ post_attention_layernorm(+残差)              │
   ▼                                              │
mlp_in [T, 6144]                                  │
   ├─ 稠密: DeepseekV2MLP [6144→i→6144]
   └─ MoE:  gate [T→E logits] ─top-k─► 权重[E_k]
            experts(FusedMoE) [T,6144→…→6144] + shared_experts
            × routed_scaling_factor,  不做 all-reduce
   ▼
mlp_out [T, 6144] ── 传给下一层: fused_allreduce_rms_norm(┘+residual)
```

MTP 草稿步（T=草稿 token 数，通常为上步 accept+1）：

```
input_ids [T] ─embed─► [T,6144] ─enorm─►┐
prev_hidden [T,6144] ──────hnorm───────►├─ concat [T,12288]
                                        ┘
   │ eh_proj [12288→6144] (GLM-5.2: build_glm52_plan)
   ▼
[T,6144] → mtp_block(=一个完整上面的 decoder layer) → shared_head.norm
   → hidden [T,6144] (recycled 作下一步 prev_hidden)
   → lm_head → logits [T, vocab] → 草稿 token
```

## 4. 四个类，自顶向下

### 4.1 DeepseekV32ForCausalLM（`nvidia/model.py:415`）

最薄的一层，继承 `DeepseekV2ForCausalLM` 只改两处：

- `model_cls = DeepseekV32Model`（把主体模型换成 DSA 版）
- `__init__` 末尾：`if model_type == "glm_moe_dsa": enable_glm52_low_latency_gemm(...)`

继承来的能力：`lm_head` + `LogitsProcessor`、MoE 元数据（EPLB）、
PP/LoRA/Eagle 支持。非 CUDA 平台的 `GlmMoeDsaForCausalLM`
（deepseek_v2.py:1982）连 low-latency GEMM 都没有，纯继承。

### 4.2 DeepseekV32Model（`nvidia/model.py:163`）

标准 transformers 式骨架，vLLM 化的三个特点：

1. **共享 top-k buffer**：`torch.empty(max_num_batched_tokens, index_topk,
   dtype=int32)`（:185）——所有层的 indexer 写、稀疏 MLA 读，零拷贝传递。
2. **PP（流水线并行）感知**：首 rank 才有 `embed_tokens`，尾 rank 才有
   `norm`，中间层用 `IntermediateTensors` 传 `hidden_states/residual`。
3. **残差流的"不约简"约定**：每层 MLP/MoE 输出不做 all-reduce，
   把 reduce **融合进下一层的 input_layernorm**
   （`fused_allreduce_rms_norm`，:307），省一次通信+一次读显存。
   最后一层把融合逻辑换成与最终 norm 融合。

首层还有个彩蛋：TP 有全词表时，embedding 查表与第一层
`input_layernorm` 融合成一次 `fused_embed_norm`（:245）。

### 4.3 DeepseekV32DecoderLayer（`nvidia/model.py:53`）

```
x ──input_layernorm(+fused allreduce)──► self_attn ──(+fused allreduce)──►
   post_attention_layernorm ──► mlp ──► (不约简，留给下一层)
```

- `layer_idx >= first_k_dense_replace` 且满足 `moe_layer_freq` → `DeepseekV2MoE`，
  否则稠密 `DeepseekV2MLP`；
- `use_sequence_parallel`（SP-MoE）时改走
  `sp_all_gather / sp_reduce_scatter` 路径（token 按 rank 分片）；
- 与 `deepseek_v2.py` 里同名父类的差别：DSA 版是普通 nn.Module，
  MoE 遍历逻辑因此重写（`set_moe_parameters`，:428）。

### 4.4 DeepseekV32Attention（`attention.py:118`）

继承 `MLAAttention`（`vllm/model_executor/layers/attention/mla_attention.py:398`），
后者持有 `W_UK_T / W_UV / kv_cache / topk_indices_buffer` 和
后端选择逻辑；v32 子类只负责**定义 DSA 的前向数据流**（§5 逐行讲）。

## 5. 一次前向：DeepseekV32Attention.forward 逐行（attention.py:278）

对照 §3.0.1 的图读代码，9 步：

1. `fused_qkv_a_proj(hidden_states)`（:284）→ split 出
   `q_c / kv_c / k_pe`（:285-287）；
2. `fused_norm_rope(...)`（:340，§12）——双 RMSNorm + 双 RoPE +
   双 cache 写入 + 清 topk buffer，一个 Triton kernel；
3. `q_b_proj(q_c)`（:368）→ `q_nope / q_pe`；
4. 权重吸收 `torch.bmm(q_nope.T, W_UK_T)`（:370）→ `ql_nope`；
5. `indexer.wq_b(q_c)`（:373）→ indexer 的 q；
6. `fused_q(...)`（:378，§13）→ `mqa_q / index_q_fp8 / index_weights`；
7. `_sparse_indexer_and_attn`（:394，§14/§15）——打分 top-k + 稀疏注意力；
8. 输出吸收 `torch.bmm(x, W_UV, out=...)`（:575）；
9. `o_proj`（:408）。

读这段代码的建议锚点：buffer `topk_indices_buffer` 是 indexer 与
sparse MLA 之间唯一的"接口"——它由 Model 层创建、Indexer 写入、
attention 后端消费，贯穿三层代码。

### 5.1 值得体会的工程细节

- **`@eager_break_during_capture`（:410）**：CUDA graph 捕获时在此"断开"，
  让 indexer 动态 shape 部分留在 eager，前后的 GEMM 仍被捕获。
- **skip_topk 交替模式（:162-172）**：`index_topk_freq` 控制每 N 层才真正
  跑 indexer，中间层直接读 buffer 里上一层的结果（近似传递）；
  MTP 层永不 skip（草稿步 0 必须算，之后可共享）。
- **短 prefill 优化（:221-230）**：序列短于 top-k 时稀疏无意义，
  直接走稠密 MHA 路径。
- **PCP/DCP**：prefill/decode context parallel 时 kv_c/k_pe 的跨卡
  gather（:473）、query 的 all-gather（:523）与 LSE 合并（:555）都在这层完成。

## 6. Lightning Indexer 的账本

两层实现，接口一致：

| | CUDA（`attention.py:47` DeepseekV32Indexer） | 通用（`deepseek_v2.py:667` Indexer） |
|---|---|---|
| Q 投影 | `wq_b: qkr → 64×128`（Replicated，不切 TP） | 同 |
| K + 打分权重 | `wk_weights_proj: H → 128+64`（融合 GEMM） | 同 |
| K 归一化 | `LayerNorm(128)` | 同 |
| 打分 | `sparse_attn_indexer`（SparseAttnIndexer 算子） | 同 |
| K cache | `DeepseekV32IndexerCache`，FP8+每128元一个FP32 scale（uint8 存储，head_dim=128+16） | 同 |

要点：

- indexer 的 Q/K 是**独立于主 MLA 的另一组头**，只做 top-k 筛选，
  不参与注意力输出本身。
- Q 侧 FP8 逐 token-组量化，scale 格式 `ue8m0`（纯 2 的幂，硬件友好）；
  K 侧量化与写 cache 融合（kernels.py `_fp8_quant_and_cache_write`）。
- `wk` 若 checkpoint 里是 FP8（带 weight_scale_inv），加载时升 BF16
  以维持与 `weights_proj` 的融合（`_try_load_fp8_indexer_wk`，
  deepseek_v2.py:845；v32 model.py:344 也调用）。
- indexer 计算里的 softmax_scale = `head_dim**-0.5`，再乘 `n_head**-0.5`
  对 64 个打分头求平均（attention.py:317、kernels.py 融合进权重）。

## 7. MoE 与 MTP

### 7.1 MoE 部分（继承自 DeepseekV2）

`DeepseekV2MoE`（deepseek_v2.py:287）：

- `gate`：线性路由，`topk_method == "noaux_tc"` 时带
  `e_score_correction_bias`（偏置修正的辅助无损负载均衡，§16）；
- `FusedMoEFactory`：组内 top-k（`n_group`/`topk_group`）、
  `routed_scaling_factor`、EPLB 冗余专家；
- `shared_experts`：稠密共享专家 MLP，可与路由专家融合成单个 MoE 调用；
- forward 里 SP 模式先 `sequence_parallel_chunk` 再算专家（:416）。

v32 层面对 MoE 的唯一定制：`reduce_results=False` + 融合进下一层 norm
（§4.2），以及 `set_moe_parameters` 按块类型收集 MoE 层（model.py:428）。

### 7.2 MTP（多 token 预测草稿模型，`nvidia/mtp.py`）

结构（`DeepseekV32MultiTokenPredictorLayer` :58）：

```
enorm(embed) ─┐
              ├─ concat ──► eh_proj [2H→H] ──► DeepseekV32DecoderLayer ×1 ──► shared_head(head)
hnorm(prev) ──┘
```

- 一个完整 DSA decoder layer 当草稿体，indexer/MoE 全套都有；
- checkpoint 里第 `num_hidden_layers` 层起的权重（nextn）映射进来，
  `_rewrite_spec_layer_name`（:320）处理 `mtp_block.` 前缀与共享 embed；
- `set_skip_topk`（:188）：草稿步 0 正常算 top-k，步 1+ 复用
  （`index_share_for_mtp_iteration`）；`compact_topk_indices`（:195）
  在 accept 后把 slot 对应的 top-k 行搬到 buffer 头部供下步用；
- 输出 recycled：最终 norm 后的 hidden 既用于算 logits 又作为下一草稿步
  的 `previous_hidden_states`（:138-147，与 deepseek_mtp.py PR #45895 对齐）；
- GLM-5.2 时 `eh_proj` 走 `build_glm52_plan` 低延迟 GEMM（:69-73, :132）。

## 8. GLM-5.2 专属：glm52_low_latency_gemm.py

只做一件事——**解码时给三个特定形状的 BF16 GEMM 挑最优后端**（SM103）：

| 投影 | [n, k] | 后端策略 |
|---|---|---|
| fused_qkv_a_proj | 2624×6144 | M=1 → CuTe skinny GEMM；M=2 → 另一配置；M=3..16 → dsv3_fused_a |
| q_b_proj | 2048×2048 | 同上模式 |
| MTP eh_proj | 6144×12288 | M=1..3 → CuTe，无 dsv3 fallback |

2624 = 2048(q_lora) + 512(kv_lora) + 64(rope)，可反推 GLM-5.2 的
hidden=6144。实现方式是**换掉 UnquantizedLinearMethod**
（`GLM52LowLatencyLinearMethod`，:158），运行时按 M 查表分发，
`VLLM_BATCH_INVARIANT` 下自动禁用（:98）。

## 9. 权重加载（`nvidia/model.py:314`）

checkpoint 命名 → vLLM 参数名的映射表：

```
gate_proj / up_proj          → gate_up_proj（合并列）
q_a_proj / kv_a_proj_with_mqa → fused_qkv_a_proj（合并列）
wk / weights_proj            → wk_weights_proj（indexer 融合）
mlp.experts.{i}.gate|up|down → FusedMoE 参数（expert_params_mapping）
```

其余走 `default_weight_loader`；MTP(nextn) 权重跳过（归 MTP 模型管）；
FP8 wk 特殊处理（§6）。

---

# 第三部分：深入篇（逐块拆解）

## 11. MLA 权重吸收：W_UK_T / W_UV 从哪来（mla_attention.py:1134-1252）

**数学**。MLA 每个 head 的注意力本应是
`softmax(q·[W_UK·kv_c ; k_pe])`，输出本应是 `Σ a_i · W_UV·kv_c_i`。
两处矩阵乘都可以换边（结合律）：

- Q 侧吸收：`q_nope · W_UK` → 先算 `ql_nope = q_nope @ W_UK_T`，
  之后 attention 直接在 512 维潜空间做（k 就是缓存的 kv_c）；
- 输出侧吸收：`a · (W_UV·kv_c)` → 先算潜空间输出 `x = Σ a_i·kv_c_i`，
  再 `x @ W_UV`。

**代码**（加载后一次性变换，`process_weights_after_loading`）：

```
kv_b_proj.weight [kvr, Nh·(d_n+v)] --T--> [Nh·(d_n+v), kvr]
  → view [kvr, Nh, d_n+v]
  → split → W_UK [kvr,Nh,d_n] / W_UV [kvr,Nh,v]
  → W_UK_T = W_UK.permute(1,2,0)   [Nh, d_n, kvr]   ← :1248
  → W_UV   = W_UV.transpose(0,1)   [Nh, kvr, v]      ← :1246
```

前向中 `bmm(q_nope.transpose(0,1), W_UK_T)`（attention.py:370）和
`bmm(x, W_UV, out=...)`（:575）。

**代价账**：cache 从 MHA 的每 token `Nh·(d_n+v)`（如 16×256=4096）
降到 576；代价是 query 从 128 维膨胀到 512 维参与 attention、
输出多一次 bmm——**用 FLOPs 换带宽**，decode（带宽瓶颈）稳赚。

kv_b_proj 本体可以是 FP8 量化的，但 bmm 无量化实现，
所以吸收矩阵是加载时反量化出的 **BF16 副本**
（:1147-1152，显存开销小）；ROCm 另有 FP8/MXFP4 bmm 路径。

## 12. fused_norm_rope：一个 kernel 干 6 件事（kernels.py:91-415）

grid = `(num_tokens, 4)`，第二维的 4 个 program 分工：

| pid | 职责 | 行号 |
|---|---|---|
| 0 | index_k: LayerNorm → RoPE → FP8 量化 → 写 indexer cache | 315-415 |
| 1 | kv_c RMSNorm + k_pe RoPE → 直接写 MLA cache（concat_and_cache 融合） | 210-314 |
| 2 | q_c RMSNorm（DCP 下任何 rank 都要算，不能被 slot<0 跳过） | 184-196 |
| 3 | topk_indices_buffer 该行清 -1（哨兵）；skip 层直接 return | 168-182 |

要点：

- **RoPE 双格式**：`INDEX_ROPE_INTERLEAVE` 真（GLM-5.2）→ 相邻配对
  `partner = block ^ 1`；假（DeepSeek-V3.2 NeoX）→ 半分
  `partner = block ^ HALF`（:350-361）。LayerNorm 的旋转搭档用
  同一份 `(mean, rstd)` 在寄存器里重归一化，避免写 scratch（:336-347）。
- **fp8_ds_mla 缓存布局**（:263-291）：`[512B FP8 NoPE | 4×f32 scale |
  128B bf16 RoPE]`——NoPE 每 128 元一个 tile 动态量化
  （scale=amax/448），RoPE 尾部保 BF16 不量化（位置敏感）。
- **padding/DCP 语义**：slot<0 只跳过 cache 写，pid1 仍要产出
  normed/RoPE 后的 K 行——稠密 prefill 每个 rank 都要读它们（:198-208）。
- **PDL**（:165-167）：`gdc_wait/gdc_launch_dependents` 让本 kernel
  与前后 kernel 重叠启动（programmatic dependent launch）。

## 13. fused_q：query 侧的 RoPE+量化打包（kernels.py:610-983）

grid = `(num_tokens, 3, grid_heads)`：

- **pid 0**：主 MLA 的 q_pe（64 维）RoPE，按 `QUANTIZE_MQA`
  写进 `mqa_q` 尾部（FP8 路径）或单独 `q_pe_out`（BF16 路径）；
- **pid 2**：ql_nope（512 维）量化打包进 `mqa_q` 头部——最终
  `mqa_q[T, Nh, 576]` 就是稀疏 MQA 后端要的完整 query；
- **pid 1**：indexer 的 64 个 head 各自 RoPE + **ue8m0 量化**
  （scale 只取 2 的幂），同时把 `index_weights` 就地乘上
  `q_scale × softmax_scale × n_head_scale`（:816-826）——
  scale 折进权重后，打分 kernel 就不用再碰 per-token scale。

`index_q` 的 RoPE 后 BF16 结果从不落盘（只消费 FP8），
省一次 store-reload（:762-764）。CUDA 上若形状支持则整体换成
CuTe DSL 实现 `fused_q_cutedsl`（:878-892）。`num_warps=1`
是实测 swept 的最优（每 program 就是 128 元的活，纯占用率瓶颈）。

## 14. Lightning Indexer 打分全流程（sparse_attn_indexer.py:296-898）

打分定义：`score(t, j) = Σ_h w_h(t) · (q_h(t) · k(j))`，
64 个 head 的 MQA 点积加权和。w 已在 fused_q 里折好全部 scale。

**prefill 路径**（:433-538）：按 chunk 遍历，把整段 indexer K cache
gather 到 workspace（`k_quant_full` + scale），算 qk logits → 加权 → top-k，
下标 scatter 进 buffer。

**decode 路径**（:540-692）：调 `fp8_fp4_paged_mqa_logits`
（DeepGEMM 系 paged MQA logits kernel）对整段 cache 出 logits，再选 top-k：

- `cooperative_topk`：SM90+、topk∈{512,1024,2048}、batch≤64、
  16B 对齐时的协作式 top-k（:617-642）；
- 否则 `persistent_topk`（同 topk 集合）/ `top_k_per_row_decode` 兜底。

**关键约定**：buffer 先填 **-1 哨兵**（不足 topk 时 padding），
kernel 只 scatter 有效下标；-1 在稀疏 MLA 侧被 mask 掉。
v32 融合路径里清零已由 `_fused_norm_rope_kernel` 的 pid3 做过，
这里 `skip_topk_buffer_clear=True` 避免重复（:426-432）。

**短 prefill 旁路**（:408-424）：主 MLA 判定走稠密 MHA 时
（`dense_mha_metadata_layer_name`），直接 return 不动 buffer——
反正没人消费。

## 15. 稀疏 MLA 后端与三条降级路径（attention.py:491-577）

```
_use_sparse_mha? ──是(短prefill/掩码场景)──► 稠密 MHA: forward_impl(
│                 q_nope‖q_pe[T,Nh,192], kv_c[T,512], k_pe[T,64])   (:500)
│                 ——此时 KV 来自当前 batch 的明文,不是 cache
└─否→ forward_mqa(mqa_q, kv_cache, metadata) (:530)
        │ 输出 attn_out[T,Nh,512] (+lse)
        ├─ DCP>1: 各 rank 部分 softmax 结果用 lse 合并 (:555 dcp_manager.combine)
        ├─ PCP: finalize_mla_pcp_decode (:562)
        ▼
      bmm(attn_out, W_UV) → [T,Nh,128] (:575)
```

`@eager_break_during_capture`（:410）+ 索引器（§14 同装饰器）：
CUDA graph 捕获在动态 shape 段前后"断开"，GEMM 段仍被捕获——
这是 vLLM **breakable graph** 模式的两处典型断点。

## 16. MoE 深读：noaux_tc 无损负载均衡（grouped_topk_router.py:112-161）

DeepSeek 系路由，`n_group`/`topk_group` 细粒度限制 + bias 修正：

```
logits [T,E] --softmax/sigmoid--> scores
scores_biased = scores + e_score_correction_bias     ← 只用于"选谁"
group_scores  = 每 n_group 组内 top-2 求和(:126)     ← 组间比较
选 topk_group 个组 → 组外 -inf → 组内 top-k 得 topk_ids
topk_weights = 原始 scores.gather(topk_ids)          ← 不带 bias(:150)
renormalize → × routed_scaling_factor
```

精髓：**bias 参与选择、不参与权重**——训练时靠 bias 引导均衡，
推理时权重仍是纯 sigmoid/softmax 分数（辅助无损，
DeepSeek-V3 论文的 aux-loss-free balancing）。
CUDA 上有 `fused_grouped_topk` 快路径（组≤32、k≤32 且有 bias，:91-108）。

分发侧：`FusedMoEFactory` 产出的 FusedMoE 把 topk_ids 展开成
`expert_params_mapping`，EP 时每 rank 持 `n_physical_experts/ep_size`
个专家（含 EPLB 冗余）；MoE 输出乘权重后仍是未约简的部分和（§17）。

## 17. 残差流约定与通信融合（fused_allreduce_rms_norm.py）

**约定**：TP>1 时 `o_proj`/`experts` 都 `reduce_results=False`，
每 rank 持部分和；通信延迟到"下一个 RMSNorm"处一次做完：

```
fused_allreduce_rms_norm(partial, residual, norm):
  TP=1  → 直接 norm(partial, residual)                    (:37)
  flashinfer trtllm kernel 可用 → 单 kernel 完成
    allreduce + add residual + RMSNorm,
    并把新 residual 写回 partial 缓冲                      (:40-60)
  否则 → 显式 all_reduce + norm 兜底                       (:62)
```

存在原因：这个模型跑 eager/breakable graph，torch.compile 的
allreduce+norm 融合不生效，只能手写（文件 docstring）。
收益：省 2 次全量显存往返，通信与 norm 计算重叠。
每层两个 norm（input/post_attention）+ 最终 norm 都走这条路；
SP-MoE 模式换成 `sp_reduce_scatter/all_gather`（token 分片，
model.py:141-159）。

## 18. MTP 运行时协作（mtp.py:188-231 + spec 框架）

- **skip_topk 共享**：`set_skip_topk(True)` 让草稿步 1+ 的 mtp_block
  直接复用步 0 的 `topk_indices_buffer`（indexer 不跑，
  fused_norm_rope pid0/3、fused_q pid1 全部短路）；
- **compact_topk_indices**：accept 后把被选中 slot 的 top-k 行
  gather 到 buffer 前部，下一步 query 行号才能对齐（:195-202）；
- **hidden recycling**：草稿层返回 `(post_norm_hidden,
  post_norm_hidden)`——同一个张量既给 `compute_logits`（只过 lm_head，
  不再 norm，:233-243）又作下一步的 `previous_hidden_states`；
  用 pre-norm hidden 会错配 hnorm、降低接受率（:138-147 注释）；
- **get_top_tokens**：TP 分片下每 rank 本地 argmax 拼 top-token，
  免掉全词表 logits all-gather（:245-260）；
- 权重完整性检查：checkpoint 缺 nextn 层直接报错（:451-464）。

## 19. GLM-5.2 低延迟 GEMM 的挂载机制（glm52_low_latency_gemm.py:176-199）

`enable_glm52_low_latency_gemm` 递归遍历子模块，凡是
`UnquantizedLinearMethod` 且 `(n,k)` 命中 `GLM52_PROJECTIONS`
表的，把 `quant_method` 换成 `GLM52LowLatencyLinearMethod`：
其 `apply()` 按 M（token 数）查 plan——M=1/2 用 CuTe skinny GEMM
（静态 K 编译），M=3..16 用 dsv3_fused_a（DeepSeek 系融合 A 矩阵
GEMM），其余回退 cuBLAS。仅 SM103 + BF16 + 非批不变模式生效
（:88-107）。MTP 的 `eh_proj` 是裸 `nn.Linear` 没有 quant_method，
所以在构造时直接 `build_glm52_plan` 预编译、forward 里
`run_glm52_plan` 分发（mtp.py:69-73,132-134）。

---

# 第四部分：学习路径与自测

## 20. 学习路径（按顺序读）

**如果你没接触过 AI infra**：先只读第一部分，直到能不看笔记回答
"MLA 为什么省显存、DSA 为什么省计算"再往下。

1. **热身**：`deepseek_v32/__init__.py`（35 行）→ 弄清平台分发和别名。
2. **主干**：`nvidia/model.py` 自上而下 442 行，重点 §4 的四个类 +
   `fused_allreduce_rms_norm` 的残差约定。
3. **核心攻坚**：`attention.py` 的 `forward`（对照 §3.0.1 的 9 步图），
   再回头看 `MLAAttention` 基类里 `W_UK_T/W_UV` 的构造（§11）。
4. **算子**：`common/kernels.py` 三个 Triton kernel 各自融合了哪些
   小算子（不必逐行读 PTX 级细节）。
5. **Indexer**：对比 `attention.py:47` 与 `deepseek_v2.py:667` 两版实现，
   差异只在内核调度；再看 `sparse_attn_indexer.py` 的 prefill/decode 两路。
6. **MoE**：`deepseek_v2.py:287-429` + `grouped_topk_router.py:112-161`。
7. **MTP**：`nvidia/mtp.py`，理解 skip_topk 共享与 hidden recycling。
8. **GLM-5.2 差异**：`glm52_low_latency_gemm.py`（200 行，很薄）。

## 21. 自测问题

入门（第一部分就该会）：
1. prefill 和 decode 的区别？为什么说 decode 是访存瓶颈？
2. MLA 只缓存 `kv_lora_rank + qk_rope_head_dim` 维向量，
   为什么数学上等价于缓存全量 K/V？
3. DSA 的 indexer 和主注意力是什么关系？为什么 indexer "便宜"？

进阶（第二、三部分）：
4. `topk_indices_buffer` 在哪创建、谁写、谁读？为什么它可以全模型共享？
5. `index_topk_freq=2` 时，偶数层和奇数层分别执行什么路径？
6. 为什么每层 MLP 输出 `reduce_results=False`？通信去哪了？
7. MoE 的 `e_score_correction_bias` 为什么只影响选择不影响权重？
8. MTP 草稿步 1+ 为什么可以 skip top-k？`compact_topk_indices` 解决什么？
9. GLM-5.2 与 DeepSeek-V3.2 在 vLLM 代码里有哪两处不同？（提示：§7.2/§8）

## 22. 关键文件行号速查

| 内容 | 位置 |
|---|---|
| 平台分发 | `vllm/models/deepseek_v32/__init__.py:17` |
| DecoderLayer | `vllm/models/deepseek_v32/nvidia/model.py:53` |
| Model(topk buffer/嵌入融合) | `nvidia/model.py:163,185,245` |
| ForCausalLM(GLM 门控) | `nvidia/model.py:415,425` |
| Attention 前向 | `vllm/models/deepseek_v32/attention.py:278` |
| Indexer(CUDA) | `attention.py:47` |
| 权重吸收注释 | `vllm/model_executor/layers/attention/mla_attention.py:62` |
| W_UK_T/W_UV 构造 | `vllm/model_executor/layers/attention/mla_attention.py:1134-1252` |
| MoE | `vllm/model_executor/models/deepseek_v2.py:287` |
| 通用 Indexer | `deepseek_v2.py:667` |
| fp8 wk 升精度 | `deepseek_v2.py:845` |
| Indexer 打分/top-k | `vllm/model_executor/layers/sparse_attn_indexer.py:296-898` |
| grouped top-k 路由 | `vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py:112-161` |
| allreduce+norm 融合 | `vllm/models/common/ops/fused_allreduce_rms_norm.py:23` |
| fp8_ds_mla 缓存布局 | `vllm/models/deepseek_v32/common/kernels.py:263-291` |
| MTP 层/共享/回收 | `vllm/models/deepseek_v32/nvidia/mtp.py:58,188,138` |
| GLM 低延迟 GEMM | `vllm/models/deepseek_v32/nvidia/glm52_low_latency_gemm.py:29` |
| Triton 融合算子 | `vllm/models/deepseek_v32/common/kernels.py:418,829,1021` |
