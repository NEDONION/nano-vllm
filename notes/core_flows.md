# Nano-vLLM 核心运行流程与组件协同

## 目录
1. [整体执行流程](#整体执行流程)
2. [初始化流程](#初始化流程)
3. [Prefill 阶段流程](#prefill-阶段流程)
4. [Decode 阶段流程](#decode-阶段流程)
5. [调度器工作流程](#调度器工作流程)
6. [KV Cache 管理流程](#kv-cache-管理流程)
7. [张量并行通信流程](#张量并行通信流程)
8. [CUDA 图优化流程](#cuda-图优化流程)
9. [完整时序图](#完整时序图)

---

## 整体执行流程

### 1. 高层视角

```
用户调用 llm.generate(prompts, sampling_params)
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ 阶段 1: 初始化                                           │
│ - 加载模型和权重                                         │
│ - 分配 KV Cache 显存                                     │
│ - 启动多进程张量并行                                     │
│ - 捕获 CUDA 图（如果启用）                               │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│ 阶段 2: 请求添加                                         │
│ - 分词（Tokenization）                                  │
│ - 创建 Sequence 对象                                     │
│ - 添加到 Scheduler 的 waiting 队列                       │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│ 阶段 3: Prefill 阶段（首次推理）                         │
│ - Scheduler 调度等待中的序列                             │
│ - BlockManager 分配 KV Cache 块                          │
│ - ModelRunner 执行前向传播                               │
│ - 生成第一个 token                                       │
│ - 序列进入 running 队列                                  │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│ 阶段 4: Decode 阶段（逐 token 生成）                     │
│ - 循环调用 step() 直到所有序列完成                       │
│ - 每次生成一个新 token                                   │
│ - 使用 CUDA 图加速（如果可用）                           │
│ - 检查停止条件（EOS 或 max_tokens）                      │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│ 阶段 5: 后处理                                           │
│ - 释放 KV Cache                                          │
│ - 解码（Detokenization）                                │
│ - 返回结果                                               │
└─────────────────────────────────────────────────────────┘
```

### 2. 核心循环

```python
# LLMEngine.generate() 的核心逻辑

# 1. 添加所有请求
for prompt, sp in zip(prompts, sampling_params):
    self.add_request(prompt, sp)

# 2. 循环执行直到所有序列完成
while not self.is_finished():
    # 2.1 执行一步推理
    outputs, num_tokens = self.step()

    # 2.2 收集完成的序列
    for seq_id, token_ids in outputs:
        results[seq_id] = token_ids

# 3. 解码并返回结果
return [tokenizer.decode(token_ids) for token_ids in results]
```

---

## 初始化流程

### 1. LLMEngine 初始化

```
LLMEngine.__init__(model_path, **config)
    │
    ├─► 解析配置参数 → Config 对象
    │
    ├─► 启动多进程张量并行
    │   └─► for rank in range(1, tensor_parallel_size):
    │           启动子进程 → ModelRunner(config, rank)
    │
    ├─► 创建主进程的 ModelRunner(config, rank=0)
    │
    ├─► 加载 Tokenizer
    │   └─► AutoTokenizer.from_pretrained(model_path)
    │
    └─► 创建 Scheduler
        └─► Scheduler.__init__(config)
            └─► BlockManager.__init__(num_blocks, block_size)
```

### 2. ModelRunner 初始化

```
ModelRunner.__init__(config, rank, events)
    │
    ├─► 初始化分布式环境
    │   └─► dist.init_process_group("nccl", ...)
    │
    ├─► 加载模型
    │   ├─► model = Qwen3ForCausalLM(hf_config)
    │   └─► load_model(model, model_path)
    │       └─► 从 SafeTensors 加载权重
    │           └─► 张量并行切片（通过 weight_loader）
    │
    ├─► 创建 Sampler
    │
    ├─► 预热模型
    │   └─► warmup_model()
    │       └─► 运行一次 dummy 推理以触发 lazy initialization
    │
    ├─► 分配 KV Cache
    │   └─► allocate_kv_cache()
    │       ├─► 计算可用显存
    │       ├─► 计算 KV Cache 块数量
    │       └─► 分配 tensor: [2, num_layers, num_blocks, block_size, num_heads, head_dim]
    │
    └─► 捕获 CUDA 图（如果启用）
        └─► capture_cudagraph()
            └─► for batch_size in [1, 2, 4, ..., 512]:
                    捕获该 batch_size 的 CUDA 图
```

### 3. 多进程架构

```
┌─────────────────────────────────────────────────────────┐
│ 主进程 (rank=0)                                          │
│                                                          │
│  ┌──────────────┐    ┌────────────┐                     │
│  │ LLMEngine    │───►│ Scheduler  │                     │
│  └──────┬───────┘    └──────┬─────┘                     │
│         │                   │                            │
│         │                   ▼                            │
│         │         ┌──────────────────┐                   │
│         │         │ BlockManager     │                   │
│         │         └──────────────────┘                   │
│         │                                                 │
│         ▼                                                 │
│  ┌──────────────────────────────────┐                    │
│  │ ModelRunner (rank=0)             │                    │
│  │  - 模型推理                       │                    │
│  │  - 共享内存写入                   │                    │
│  └──────────────┬───────────────────┘                    │
└─────────────────┼───────────────────────────────────────┘
                  │ 共享内存 (pickled method calls)
                  │ multiprocessing.Event (同步信号)
┌─────────────────┼───────────────────────────────────────┐
│ 子进程 (rank=1,2,...,N-1)                                │
│                 │                                         │
│  ┌──────────────▼───────────────────┐                    │
│  │ ModelRunner (rank=1)             │                    │
│  │  - 从共享内存读取                 │                    │
│  │  - 执行相同的推理                 │                    │
│  │  - NCCL 通信同步                  │                    │
│  └──────────────────────────────────┘                    │
└─────────────────────────────────────────────────────────┘
```

**通信机制**:
```python
# 主进程写入共享内存
def write_shm(self, method_name, *args):
    data = pickle.dumps([method_name, *args])
    n = len(data)
    self.shm.buf[0:4] = n.to_bytes(4, "little")  # 数据长度
    self.shm.buf[4:n+4] = data                    # 序列化数据
    for event in self.events:
        event.set()  # 通知子进程

# 子进程读取共享内存
def read_shm(self):
    self.event.wait()  # 等待信号
    n = int.from_bytes(self.shm.buf[0:4], "little")
    method_name, *args = pickle.loads(self.shm.buf[4:n+4])
    self.event.clear()
    return method_name, args
```

---

## Prefill 阶段流程

### 1. 完整流程图

```
用户请求到达
    │
    ▼
LLMEngine.add_request(prompt, sampling_params)
    │
    ├─► Tokenizer.encode(prompt) → token_ids
    │
    ├─► 创建 Sequence(token_ids, sampling_params)
    │   └─► status = WAITING
    │
    └─► Scheduler.add(seq)
        └─► waiting.append(seq)

────────────────────────────────────────────

LLMEngine.step() [Prefill 轮次]
    │
    ▼
Scheduler.schedule()
    │
    ├─► 从 waiting 队列取出序列
    │
    ├─► 检查资源限制
    │   ├─► num_batched_tokens + len(seq) ≤ max_num_batched_tokens?
    │   └─► BlockManager.can_allocate(seq)?
    │
    ├─► BlockManager.allocate(seq)
    │   │
    │   ├─► 计算需要的块数量: num_blocks = ceil(len(seq) / block_size)
    │   │
    │   └─► for each block:
    │       ├─► 计算哈希: h = xxhash(token_ids, prefix_hash)
    │       │
    │       ├─► 查找哈希表: block_id = hash_to_block_id.get(h)
    │       │
    │       ├─► if 缓存命中 and token_ids 匹配:
    │       │   ├─► 复用块（增加 ref_count）
    │       │   └─► num_cached_tokens += block_size
    │       │
    │       └─► else (缓存未命中):
    │           ├─► 分配新块
    │           └─► 更新哈希表
    │
    ├─► 序列状态转换: WAITING → RUNNING
    │
    └─► return (scheduled_seqs, is_prefill=True)

────────────────────────────────────────────

ModelRunner.run(seqs, is_prefill=True)
    │
    ├─► prepare_prefill(seqs) → 准备输入数据
    │   │
    │   ├─► input_ids: 展平的 token IDs（跳过缓存的部分）
    │   │   例如: seq1=[1,2,3], seq2=[4,5]
    │   │       → input_ids=[1,2,3,4,5]
    │   │
    │   ├─► positions: 每个 token 的位置索引
    │   │   例如: seq1 缓存了 0, seq2 缓存了 0,1
    │   │       → positions=[0,1,2,2,3]
    │   │
    │   ├─► slot_mapping: token → KV Cache slot 映射
    │   │   例如: slot_mapping=[10,11,12,15,16]
    │   │       （10,11,12 是 seq1 的 slots，15,16 是 seq2 的）
    │   │
    │   ├─► cu_seqlens_q/k: 累积序列长度（用于 Flash Attention）
    │   │   例如: cu_seqlens_q=[0, 3, 5]
    │   │       （seq1 长度 3，seq2 长度 2）
    │   │
    │   └─► block_tables: 每个序列的块表（用于 Paged Attention）
    │       例如: [[5,12,-1], [7,8,9]]
    │
    ├─► set_context(context) → 设置全局上下文
    │   └─► is_prefill=True, 以及上面准备的所有元数据
    │
    ├─► 模型前向传播
    │   └─► model.forward(input_ids, positions)
    │       │
    │       ├─► Embedding Layer
    │       │   └─► hidden_states = embed_tokens(input_ids)
    │       │       shape: [total_tokens, hidden_size]
    │       │
    │       ├─► Decoder Layers (循环)
    │       │   └─► for layer in layers:
    │       │       │
    │       │       ├─► RMSNorm + Attention
    │       │       │   ├─► QKV 投影
    │       │       │   ├─► RoPE 位置编码
    │       │       │   ├─► Flash Attention (Prefill 模式)
    │       │       │   │   └─► flash_attn_varlen_func(
    │       │       │   │           q, k, v,
    │       │       │   │           cu_seqlens_q, cu_seqlens_k,
    │       │       │   │           block_table,  # 支持 Prefix Cache
    │       │       │   │           causal=True
    │       │       │   │       )
    │       │       │   │   ├─► 同时存储 KV 到 Cache
    │       │       │   │   │   └─► store_kvcache(k, v, slot_mapping)
    │       │       │   │   │
    │       │       │   │   └─► 返回 attention_output
    │       │       │   │
    │       │       │   └─► O 投影
    │       │       │
    │       │       └─► RMSNorm + MLP
    │       │           ├─► Gate/Up 投影
    │       │           ├─► SiluAndMul 激活
    │       │           └─► Down 投影
    │       │
    │       ├─► 最终 RMSNorm
    │       │
    │       ├─► 仅取最后一个 token 的 hidden_states
    │       │   └─► hidden_states = hidden_states[context.last_token_indices]
    │       │       shape: [num_seqs, hidden_size]
    │       │
    │       └─► LM Head 投影
    │           └─► logits = lm_head(hidden_states)
    │               shape: [num_seqs, vocab_size]
    │
    ├─► 采样
    │   └─► token_ids = sampler(logits, temperatures)
    │       └─► sample(logits / temperature) → 多项式采样
    │
    └─► return token_ids

────────────────────────────────────────────

Scheduler.postprocess(seqs, token_ids)
    │
    └─► for seq, token_id in zip(seqs, token_ids):
        ├─► seq.append_token(token_id)
        └─► 序列仍在运行，进入下一轮 Decode
```

### 2. Prefill 阶段关键点

**输入数据准备**:
```python
# 示例：两个序列
seq1 = [1, 2, 3, 4, 5]  # 长度 5, 缓存了前 3 个 tokens
seq2 = [6, 7, 8]        # 长度 3, 无缓存

# 输入数据
input_ids = [4, 5, 6, 7, 8]  # 仅未缓存的部分
positions = [3, 4, 0, 1, 2]  # 位置索引

# Flash Attention 元数据
cu_seqlens_q = [0, 2, 5]     # seq1 贡献 2 个 tokens, seq2 贡献 3 个
cu_seqlens_k = [0, 5, 8]     # seq1 上下文长度 5, seq2 上下文长度 3
max_seqlen_q = 3
max_seqlen_k = 5

# Slot Mapping (KV Cache 位置)
seq1 的块表: [0, 1]  # 块 0 和块 1
seq2 的块表: [2]     # 块 2

# 块大小 = 16 的情况下
slot_mapping = [
    3, 4,        # seq1 的 token 4,5 → 块 0 的 slot 3,4
    32, 33, 34   # seq2 的 token 6,7,8 → 块 2 的 slot 0,1,2 (offset=32)
]
```

**Prefix Caching 示例**:
```
请求 1: "你是一个AI助手。请介绍自己。"
请求 2: "你是一个AI助手。请讲个笑话。"

公共前缀: "你是一个AI助手。" (假设 10 个 tokens)

块大小 = 16, 公共前缀占用 1 个块

请求 1:
  - 块 0: tokens[0:10] (公共前缀) → 哈希 = h1
  - 块 1: tokens[10:26] → 新分配

请求 2:
  - 块 0: tokens[0:10] (公共前缀) → 哈希 = h1 (命中！)
    → 复用请求 1 的块 0, ref_count = 2
  - 块 1: tokens[10:26] → 新分配

结果: 请求 2 跳过公共前缀的计算，直接从缓存读取 KV！
```

---

## Decode 阶段流程

### 1. 完整流程图

```
LLMEngine.step() [Decode 轮次]
    │
    ▼
Scheduler.schedule()
    │
    ├─► 从 running 队列取出所有序列
    │
    └─► for each seq in running:
        │
        ├─► 检查是否需要新的 KV Cache 块
        │   └─► if len(seq) % block_size == 1:  # 当前块已满
        │       └─► BlockManager.can_append(seq)?
        │
        ├─► if 资源不足:
        │   └─► 抢占低优先级序列（从队尾开始）
        │       └─► Scheduler.preempt(seq)
        │           ├─► BlockManager.deallocate(seq)
        │           └─► seq.status = WAITING
        │
        ├─► else:
        │   └─► BlockManager.may_append(seq)
        │       └─► 预分配下一个 token 的 slot
        │
        └─► return (scheduled_seqs, is_prefill=False)

────────────────────────────────────────────

ModelRunner.run(seqs, is_prefill=False)
    │
    ├─► prepare_decode(seqs) → 准备输入数据
    │   │
    │   ├─► input_ids: 每个序列的最后一个 token
    │   │   例如: seq1 刚生成 token 99, seq2 刚生成 token 88
    │   │       → input_ids=[99, 88]
    │   │
    │   ├─► positions: 当前位置索引
    │   │   例如: seq1 长度 10, seq2 长度 5
    │   │       → positions=[9, 4]
    │   │
    │   ├─► context_lens: 每个序列的上下文长度
    │   │   例如: context_lens=[10, 5]
    │   │
    │   ├─► slot_mapping: 新 token 的 KV Cache slot
    │   │   例如: slot_mapping=[160, 80]
    │   │
    │   └─► block_tables: 每个序列的完整块表
    │       例如: [[0,1,2], [3,4,-1]]
    │
    ├─► set_context(context) → 设置全局上下文
    │   └─► is_prefill=False, 以及上面准备的元数据
    │
    ├─► 判断是否使用 CUDA 图
    │   └─► if !enforce_eager and batch_size in graphs:
    │       └─► 使用 CUDA 图重放
    │   └─► else:
    │       └─► 直接前向传播
    │
    ├─► 【路径 A: CUDA 图重放】
    │   ├─► graph_pool = graphs[batch_size]
    │   ├─► 拷贝输入到固定缓冲区
    │   │   └─► graph_pool.input_ids_buf.copy_(input_ids)
    │   │       graph_pool.positions_buf.copy_(positions)
    │   │       ...
    │   ├─► 重放 CUDA 图
    │   │   └─► graph_pool.graph.replay()
    │   └─► 从固定缓冲区读取输出
    │       └─► logits = graph_pool.logits_buf.clone()
    │
    └─► 【路径 B: 直接前向传播】
        └─► logits = model.forward(input_ids, positions)
            │
            ├─► Embedding Layer
            │   └─► hidden_states = embed_tokens(input_ids)
            │       shape: [num_seqs, hidden_size]
            │
            ├─► Decoder Layers
            │   └─► for layer in layers:
            │       │
            │       ├─► RMSNorm + Attention
            │       │   ├─► QKV 投影
            │       │   ├─► RoPE 位置编码
            │       │   ├─► Flash Attention (Decode 模式)
            │       │   │   └─► flash_attn_with_kvcache(
            │       │   │           q.unsqueeze(1),  # [batch, 1, num_heads, head_dim]
            │       │   │           k_cache, v_cache,
            │       │   │           cache_seqlens,
            │       │   │           block_table,
            │       │   │           causal=True
            │       │   │       )
            │       │   │   ├─► 同时存储新 KV 到 Cache
            │       │   │   │   └─► store_kvcache(k, v, slot_mapping)
            │       │   │   │
            │       │   │   └─► 返回 attention_output
            │       │   │
            │       │   └─► O 投影
            │       │
            │       └─► RMSNorm + MLP
            │
            ├─► 最终 RMSNorm
            │
            └─► LM Head 投影
                └─► logits = lm_head(hidden_states)

────────────────────────────────────────────

采样和后处理
    │
    ├─► token_ids = sampler(logits, temperatures)
    │
    └─► Scheduler.postprocess(seqs, token_ids)
        │
        └─► for seq, token_id in zip(seqs, token_ids):
            │
            ├─► seq.append_token(token_id)
            │
            ├─► 检查停止条件
            │   ├─► if token_id == EOS and not ignore_eos:
            │   │   └─► seq.status = FINISHED
            │   └─► if num_completion_tokens == max_tokens:
            │       └─► seq.status = FINISHED
            │
            └─► if seq.is_finished:
                ├─► BlockManager.deallocate(seq) → 释放 KV Cache
                └─► running.remove(seq)
```

### 2. Decode 阶段关键点

**CUDA 图 vs 直接前向传播**:

```python
# 条件判断
if !enforce_eager and num_seqs in self.graphs:
    # 使用 CUDA 图（更快）
    use_cudagraph = True
else:
    # 直接前向传播（更灵活）
    use_cudagraph = False
```

**CUDA 图优势**:
- 减少 kernel 启动开销（~30% 延迟降低）
- CPU-GPU 同步次数减少
- kernel 调度优化

**CUDA 图限制**:
- 输入形状必须固定（因此仅支持特定 batch sizes）
- 不支持动态控制流
- 首次捕获耗时长

**Decode 阶段输入准备**:
```python
# 示例：3 个序列正在 Decode
seq1 = [1,2,3,4,5,6,7]    # 长度 7, 刚生成了 token 7
seq2 = [8,9,10]           # 长度 3, 刚生成了 token 10
seq3 = [11,12,13,14,15]   # 长度 5, 刚生成了 token 15

# 输入数据
input_ids = [7, 10, 15]       # 每个序列的最后一个 token
positions = [6, 2, 4]         # 当前位置（长度 - 1）
context_lens = [7, 3, 5]      # 每个序列的上下文长度

# Slot Mapping (新 token 的 KV Cache 位置)
# 假设块大小 = 16
seq1 的块表: [0, 1]  # token 7 在块 0 的 slot 7
seq2 的块表: [2]     # token 10 在块 2 的 slot 2
seq3 的块表: [3, 4]  # token 15 在块 3 的 slot 4

slot_mapping = [7, 34, 52]  # (0*16+7, 2*16+2, 3*16+4)
```

---

## 调度器工作流程

### 1. 调度策略流程图

```
Scheduler.schedule()
    │
    ├─► 阶段 1: 尝试调度 Prefill 请求
    │   │
    │   └─► while waiting 队列非空 and num_seqs < max_num_seqs:
    │       │
    │       ├─► seq = waiting[0]
    │       │
    │       ├─► 检查 token 限制
    │       │   └─► if num_batched_tokens + len(seq) > max_num_batched_tokens:
    │       │       └─► break  # 无法继续添加
    │       │
    │       ├─► 检查 KV Cache 可用性
    │       │   └─► if !BlockManager.can_allocate(seq):
    │       │       └─► break  # KV Cache 不足
    │       │
    │       ├─► 分配资源
    │       │   ├─► BlockManager.allocate(seq)
    │       │   ├─► seq.status = RUNNING
    │       │   ├─► waiting.popleft()
    │       │   └─► running.append(seq)
    │       │
    │       └─► scheduled_seqs.append(seq)
    │
    ├─► if scheduled_seqs 非空:
    │   └─► return (scheduled_seqs, is_prefill=True)
    │
    └─► 阶段 2: 调度 Decode 请求
        │
        └─► while running 队列非空 and num_seqs < max_num_seqs:
            │
            ├─► seq = running.popleft()
            │
            ├─► 检查 KV Cache 是否需要扩展
            │   └─► while !BlockManager.can_append(seq):
            │       │
            │       ├─► if running 队列非空:
            │       │   └─► 抢占队尾序列
            │       │       └─► preempt(running.pop())
            │       │
            │       └─► else:
            │           └─► 抢占当前序列
            │               └─► preempt(seq)
            │               └─► break
            │
            ├─► 预分配 slot
            │   └─► BlockManager.may_append(seq)
            │
            └─► scheduled_seqs.append(seq)
```

### 2. 调度优先级

```
优先级从高到低:
1. Prefill 请求（减少用户等待时间）
2. Decode 请求（保持序列继续生成）

抢占顺序:
1. 优先抢占 running 队列尾部的序列（后进先出）
2. 如果 running 为空，抢占当前序列自己
```

### 3. 资源抢占机制

```
BlockManager.can_append(seq) 返回 False
    │
    ▼
触发抢占
    │
    ├─► 选择受害序列: victim = running.pop()  # 队尾
    │
    ├─► BlockManager.deallocate(victim)
    │   └─► for block_id in victim.block_table:
    │       ├─► blocks[block_id].ref_count -= 1
    │       └─► if ref_count == 0:
    │           └─► 释放块到 free_block_ids
    │
    ├─► victim.status = WAITING
    │
    └─► waiting.appendleft(victim)  # 插入队首，优先重新调度
```

**抢占示例**:
```
场景: KV Cache 不足，需要为新 token 分配块

running = [seq1, seq2, seq3, seq4]

# seq4 需要新块，但 free_block_ids 为空
抢占 seq4 自己（队尾）
    → seq4.status = WAITING
    → 释放 seq4 的所有块
    → waiting.appendleft(seq4)

running = [seq1, seq2, seq3]

下一轮:
    seq1, seq2, seq3 继续 Decode
    seq4 在 waiting 队列等待资源
```

---

## KV Cache 管理流程

### 1. 分配流程

```
BlockManager.allocate(seq)
    │
    ├─► 计算需要的块数量
    │   └─► num_blocks = ceil(len(seq) / block_size)
    │
    └─► for i in range(num_blocks):
        │
        ├─► 获取当前块的 tokens
        │   └─► token_ids = seq.block(i)
        │
        ├─► 计算哈希（仅完整块）
        │   └─► if len(token_ids) == block_size:
        │       └─► h = xxhash(token_ids, prefix=previous_hash)
        │   └─► else:
        │       └─► h = -1  # 不完整的块不缓存
        │
        ├─► 查找哈希表
        │   └─► block_id = hash_to_block_id.get(h, -1)
        │
        ├─► 验证哈希（防止碰撞）
        │   └─► if block_id != -1 and blocks[block_id].token_ids == token_ids:
        │       └─► cache_hit = True
        │   └─► else:
        │       └─► cache_hit = False
        │
        ├─► 【路径 A: 缓存命中】
        │   │
        │   ├─► if block_id in used_block_ids:
        │   │   ├─► 块正在使用，增加引用计数
        │   │   └─► blocks[block_id].ref_count += 1
        │   │
        │   └─► else:
        │       ├─► 块已释放但哈希仍在表中
        │       └─► 重新分配该块
        │           └─► _allocate_block(block_id)
        │
        └─► 【路径 B: 缓存未命中】
            │
            ├─► 从 free_block_ids 获取新块
            │   └─► block_id = free_block_ids[0]
            │
            ├─► 分配块
            │   └─► _allocate_block(block_id)
            │       ├─► blocks[block_id].reset()
            │       ├─► free_block_ids.remove(block_id)
            │       └─► used_block_ids.add(block_id)
            │
            └─► 更新哈希表
                └─► blocks[block_id].update(h, token_ids)
                └─► hash_to_block_id[h] = block_id
```

### 2. 追加流程

```
BlockManager.may_append(seq)
    │
    ├─► last_block = blocks[seq.block_table[-1]]
    │
    ├─► 情况 1: 当前块刚满（长度 % block_size == 1）
    │   │
    │   ├─► 上一个块已填满，需要新块
    │   │   └─► block_id = free_block_ids[0]
    │   │       _allocate_block(block_id)
    │   │       seq.block_table.append(block_id)
    │   │
    │   └─► 将上一个块的哈希记录到表中
    │       └─► h = compute_hash(seq.block(num_blocks-1), prefix)
    │           last_block.update(h, token_ids)
    │           hash_to_block_id[h] = block_id
    │
    ├─► 情况 2: 当前块刚填满（长度 % block_size == 0）
    │   │
    │   └─► 更新当前块的哈希
    │       └─► h = compute_hash(seq.block(num_blocks-1), prefix)
    │           last_block.update(h, token_ids)
    │           hash_to_block_id[h] = last_block.block_id
    │
    └─► 情况 3: 当前块未满
        └─► 无操作（哈希保持 -1，表示不完整）
```

### 3. 释放流程

```
BlockManager.deallocate(seq)
    │
    └─► for block_id in reversed(seq.block_table):
        │
        ├─► blocks[block_id].ref_count -= 1
        │
        └─► if ref_count == 0:
            └─► _deallocate_block(block_id)
                ├─► used_block_ids.remove(block_id)
                └─► free_block_ids.append(block_id)
```

### 4. KV Cache 内存布局

```
KV Cache Tensor:
shape = [2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
         │   │           │           │            │              │
         │   │           │           │            │              └─► 每个头的维度 (128)
         │   │           │           │            └────────────────► KV 头数 (2)
         │   │           │           └─────────────────────────────► 块大小 (256)
         │   │           └─────────────────────────────────────────► 块数量 (动态计算)
         │   └─────────────────────────────────────────────────────► 层数 (28)
         └─────────────────────────────────────────────────────────► K/V (0=K, 1=V)

示例:
  Qwen3-0.6B, block_size=256, num_blocks=1000
  内存占用: 2 * 28 * 1000 * 256 * 2 * 128 * 2 bytes (bfloat16)
          = ~3.6 GB
```

**Slot Mapping 示例**:
```
序列: [1,2,3,4,5,6,7,8]
块表: [5, 12]  # 块 5 和块 12
块大小: 16

Slot Mapping:
  token 1 → slot 5*16+0 = 80
  token 2 → slot 5*16+1 = 81
  ...
  token 8 → slot 5*16+7 = 87

KV Cache 访问:
  k_cache[layer_id, block_id, slot_in_block, head_id, :]
  = k_cache[0, 5, 7, 0, :]  # 第 0 层, 块 5, slot 7, 头 0
```

---

## 张量并行通信流程

### 1. 权重加载阶段

```
load_model(model, model_path)
    │
    └─► for name, param in model.named_parameters():
        │
        └─► if hasattr(param, 'weight_loader'):
            │
            └─► param.weight_loader(param, loaded_weight)
                │
                ├─► 计算切片范围
                │   └─► shard_size = param.size(tp_dim)
                │       start_idx = tp_rank * shard_size
                │
                ├─► 切片权重
                │   └─► loaded_weight = loaded_weight.narrow(tp_dim, start_idx, shard_size)
                │
                └─► 复制到参数
                    └─► param.data.copy_(loaded_weight)

示例:
  完整权重: [4096, 14336]  # output_size, input_size
  TP size = 2

  GPU 0: [4096, 7168]  # 切分 input_size 维度
  GPU 1: [4096, 7168]
```

### 2. 前向传播阶段

#### 词嵌入层（VocabParallelEmbedding）

```
forward(input_ids)
    │
    ├─► 计算本 GPU 负责的词表范围
    │   └─► vocab_start = tp_rank * vocab_size_per_gpu
    │       vocab_end = (tp_rank + 1) * vocab_size_per_gpu
    │
    ├─► 创建掩码
    │   └─► mask = (input_ids >= vocab_start) & (input_ids < vocab_end)
    │
    ├─► 本地查找
    │   └─► local_ids = (input_ids - vocab_start) * mask
    │       embeddings = F.embedding(local_ids, self.weight)
    │
    └─► All-Reduce（跨 GPU 求和）
        └─► dist.all_reduce(embeddings, op=dist.ReduceOp.SUM)

示例:
  vocab_size = 10000, TP size = 2

  GPU 0 负责: vocab[0:5000]
  GPU 1 负责: vocab[5000:10000]

  input_ids = [123, 6789]

  GPU 0:
    - token 123 → embedding[123]
    - token 6789 → zero (超出范围)

  GPU 1:
    - token 123 → zero (超出范围)
    - token 6789 → embedding[6789-5000]

  All-Reduce 后:
    - token 123 → GPU 0 的 embedding[123]
    - token 6789 → GPU 1 的 embedding[1789]
```

#### 列并行线性层（ColumnParallelLinear）

```
forward(x)
    │
    └─► output = F.linear(x, self.weight, self.bias)
        # 无需通信！每个 GPU 产生部分输出

示例:
  input: [batch, seq, 4096]
  完整权重: [14336, 4096]
  TP size = 2

  GPU 0:
    weight: [7168, 4096]
    output: [batch, seq, 7168]

  GPU 1:
    weight: [7168, 4096]
    output: [batch, seq, 7168]

  下一层会自动拼接（或使用 RowParallelLinear）
```

#### 行并行线性层（RowParallelLinear）

```
forward(x)
    │
    ├─► local_output = F.linear(x, self.weight)
    │   # 每个 GPU 产生完整输出，但基于部分输入
    │
    └─► output = dist.all_reduce(local_output, op=dist.ReduceOp.SUM)
        # 跨 GPU 求和

示例:
  input: [batch, seq, 14336]（已被列切分）
  完整权重: [4096, 14336]
  TP size = 2

  GPU 0:
    weight: [4096, 7168]
    input: [batch, seq, 7168]（前半部分）
    local_output: [batch, seq, 4096]

  GPU 1:
    weight: [4096, 7168]
    input: [batch, seq, 7168]（后半部分）
    local_output: [batch, seq, 4096]

  All-Reduce:
    output = GPU0_output + GPU1_output
```

### 3. 通信模式总结

```
VocabParallelEmbedding:
  input → local_embed → all_reduce → output

ColumnParallelLinear:
  input → local_linear → output (无通信)

RowParallelLinear:
  input → local_linear → all_reduce → output

QKV 投影:
  input → ColumnParallelLinear (QKV 并行) → output (无通信)

Attention:
  Q,K,V → Flash Attention (本地) → output (无通信)

O 投影:
  input → RowParallelLinear → all_reduce → output

MLP Gate/Up:
  input → MergedColumnParallelLinear → output (无通信)

MLP Down:
  input → RowParallelLinear → all_reduce → output
```

**每层的通信次数**: 2 次 All-Reduce（O 投影 + MLP Down 投影）

---

## CUDA 图优化流程

### 1. 捕获流程

```
ModelRunner.capture_cudagraph()
    │
    └─► for num_seqs in [1, 2, 4, 8, 16, ..., 512]:
        │
        ├─► 准备固定大小的输入 tensors
        │   ├─► input_ids_buf = torch.zeros(num_seqs, dtype=torch.int64)
        │   ├─► positions_buf = torch.zeros(num_seqs, dtype=torch.int64)
        │   ├─► slot_mapping_buf = torch.zeros(num_seqs, dtype=torch.int64)
        │   └─► ...
        │
        ├─► 准备固定大小的输出 tensors
        │   └─► logits_buf = torch.zeros(num_seqs, vocab_size, dtype=...)
        │
        ├─► 创建 CUDA 图
        │   └─► graph = torch.cuda.CUDAGraph()
        │
        ├─► 捕获前向传播
        │   └─► with torch.cuda.graph(graph):
        │       └─► logits_buf = model.forward(
        │               input_ids_buf,
        │               positions_buf,
        │               ...
        │           )
        │
        └─► 存储图和缓冲区
            └─► self.graphs[num_seqs] = GraphPool(
                    graph=graph,
                    input_ids_buf=input_ids_buf,
                    ...
                    logits_buf=logits_buf
                )
```

### 2. 重放流程

```
ModelRunner.run_cudagraph(seqs)
    │
    ├─► num_seqs = len(seqs)
    │
    ├─► 获取对应的图
    │   └─► graph_pool = self.graphs[num_seqs]
    │
    ├─► 拷贝输入到固定缓冲区
    │   ├─► graph_pool.input_ids_buf.copy_(input_ids)
    │   ├─► graph_pool.positions_buf.copy_(positions)
    │   └─► ...
    │
    ├─► 重放 CUDA 图
    │   └─► graph_pool.graph.replay()
    │       # 执行预先录制的所有 kernel 调用
    │
    └─► 从固定缓冲区读取输出
        └─► logits = graph_pool.logits_buf.clone()
        └─► return logits
```

### 3. CUDA 图优势

```
传统执行:
  Python → PyTorch API → CUDA kernel 启动 (多次 CPU-GPU 同步)

CUDA 图执行:
  Python → 重放图 (一次 CPU-GPU 同步，内部无同步)

性能提升:
  - Kernel 启动开销: ~30% 降低
  - CPU-GPU 同步: ~90% 减少
  - 端到端延迟: ~20-30% 降低
```

### 4. 支持的 Batch Sizes

```
# 捕获的 batch sizes (2 的幂次)
supported_batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

# 实际使用
if num_seqs in supported_batch_sizes:
    use_cudagraph = True
else:
    use_cudagraph = False  # 降级到直接前向传播
```

---

## 完整时序图

### 单个请求的完整生命周期

```
时间轴
  │
  ├─► T0: 用户调用 llm.generate(["Hello"])
  │       │
  │       └─► LLMEngine.add_request("Hello", sampling_params)
  │           ├─► Tokenizer.encode("Hello") → [1234]
  │           └─► Sequence([1234], sampling_params) → waiting 队列
  │
  ├─► T1: LLMEngine.step() [第 1 轮]
  │       │
  │       ├─► Scheduler.schedule()
  │       │   ├─► 调度 seq 从 waiting → running
  │       │   └─► BlockManager.allocate(seq) → 分配块 [5]
  │       │
  │       ├─► ModelRunner.run(seqs, is_prefill=True)
  │       │   ├─► prepare_prefill([seq])
  │       │   │   └─► input_ids=[1234], positions=[0]
  │       │   │
  │       │   ├─► model.forward(input_ids, positions)
  │       │   │   ├─► Embedding → hidden_states
  │       │   │   ├─► Decoder Layers (Flash Attention Prefill)
  │       │   │   │   └─► 存储 KV 到块 5
  │       │   │   └─► LM Head → logits
  │       │   │
  │       │   └─► sampler(logits) → token_id=5678
  │       │
  │       └─► Scheduler.postprocess([seq], [5678])
  │           └─► seq.append_token(5678)
  │               seq = [1234, 5678]
  │
  ├─► T2: LLMEngine.step() [第 2 轮]
  │       │
  │       ├─► Scheduler.schedule()
  │       │   └─► 调度 seq (Decode 阶段)
  │       │       BlockManager.may_append(seq)
  │       │
  │       ├─► ModelRunner.run(seqs, is_prefill=False)
  │       │   ├─► prepare_decode([seq])
  │       │   │   └─► input_ids=[5678], positions=[1], context_lens=[2]
  │       │   │
  │       │   ├─► 使用 CUDA 图重放 (batch_size=1)
  │       │   │   ├─► 拷贝输入到缓冲区
  │       │   │   ├─► graph.replay()
  │       │   │   │   └─► Flash Attention Decode (读取块 5 的 KV)
  │       │   │   └─► 读取 logits
  │       │   │
  │       │   └─► sampler(logits) → token_id=9012
  │       │
  │       └─► Scheduler.postprocess([seq], [9012])
  │           └─► seq.append_token(9012)
  │               seq = [1234, 5678, 9012]
  │
  ├─► T3-TN: 重复 Decode 阶段...
  │
  └─► TN: Decode 完成
          │
          ├─► token_id = EOS
          │
          ├─► Scheduler.postprocess()
          │   ├─► seq.status = FINISHED
          │   └─► BlockManager.deallocate(seq)
          │       └─► 释放块 5
          │
          └─► LLMEngine.generate() 返回
              └─► Tokenizer.decode([1234, 5678, 9012, ...])
                  → "Hello, how can I help you today?"
```

---

## 总结

### 核心流程特点

1. **两阶段推理**:
   - Prefill: 并行处理所有输入 tokens
   - Decode: 逐个生成输出 tokens

2. **动态调度**:
   - 连续批处理（Continuous Batching）
   - 优先级调度（Prefill > Decode）
   - 资源抢占（Preemption）

3. **高效内存管理**:
   - 分块 KV Cache（Paged Attention）
   - Prefix Caching（哈希共享）
   - 引用计数（Copy-on-Write）

4. **性能优化**:
   - CUDA 图（减少启动开销）
   - Flash Attention（O(N) 内存）
   - 张量并行（多 GPU 加速）

5. **多进程协同**:
   - 共享内存通信
   - NCCL 集合通信
   - 事件同步

这些设计共同构成了一个高性能、高效率的大语言模型推理引擎！
