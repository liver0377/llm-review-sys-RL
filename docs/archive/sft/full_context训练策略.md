# Full Context 训练策略详解

## 概述

`train_full_context.py` 实现了使用 QLoRA 技术对 LLaMA-3.1-8B-Instruct 模型进行完整上下文微调的完整流程，适用于需要在 H100 GPU 上处理超长论文内容（~18,000 tokens）的场景。

---

## 整体架构图

```
配置加载 → 模型初始化 → 数据预处理 → 训练配置 → 训练执行 → 保存模型
```

---

## 详细工作流程

### 阶段 1: 环境初始化

```python
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TRANSFORMERS_CACHE"] = "/workspace/hf_cache"
os.environ["HF_HOME"] = "/workspace/hf_home"
```

**配置说明:**
| 环境变量 | 作用 |
|---------|------|
| `PYTORCH_CUDA_ALLOC_CONF` | 允许 CUDA 内存动态扩展，减少 OOM 风险 |
| `TRANSFORMERS_CACHE` | 设置模型缓存路径，避免重复下载 |
| `HF_HOME` | 设置 HuggingFace 主目录，统一管理缓存 |

---

### 阶段 2: 辅助函数定义

#### 2.1 配置加载 - `load_config()`

```python
def load_config(config_path):
    print(f"📄 Loading config from: {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
```

**功能**: 从 YAML 文件加载训练配置

**示例配置文件 (`qlora_train_config.yaml`):**
```yaml
model_name_or_path: "meta-llama/Llama-3.1-8B-Instruct"
hf_dataset_repo: "guochenmeinian/openreview_dataset"
output_dir: "../model/full_context_qlora"

lora_rank: 8
lora_alpha: 32
lora_dropout: 0.05

per_device_train_batch_size: 1
gradient_accumulation_steps: 4
num_train_epochs: 2
learning_rate: 0.00002

cutoff_len: 18000
load_in_4bit: true
bnb_4bit_compute_dtype: bfloat16
bnb_4bit_use_double_quant: true

bf16: true
```

---

#### 2.2 数据集加载 - `load_dataset_from_hf()`

```python
def load_dataset_from_hf(dataset_repo, file_name, split="train"):
    print(f"📦 Loading HF dataset: {dataset_repo}/{file_name}")
    dataset = load_dataset(dataset_repo, data_files=file_name, split=split)
    print(f"✅ Loaded {len(dataset)} examples.")
    return dataset
```

**功能**: 从 HuggingFace Hub 加载数据集

**使用示例:**
```python
dataset = load_dataset_from_hf(
    "guochenmeinian/openreview_dataset",
    "qlora_train",
    split="train"
)
```

---

#### 2.3 Prompt 构建 - `build_prompt()`

```python
def build_prompt(example):
    return {
        "prompt": f"{example['instruction']}\n\n{example['input']}".strip(),
        "response": example["output"]
    }
```

**功能**: 将原始数据格式转换为 prompt-response 格式

**数据转换示例:**

**输入:**
```json
{
  "instruction": "Review this paper in detail.",
  "input": "论文全文内容...",
  "output": "审稿内容..."
}
```

**输出:**
```json
{
  "prompt": "Review this paper in detail.\n\n论文全文内容...",
  "response": "审稿内容..."
}
```

---

#### 2.4 数据预处理 - `preprocess_full_prompt()` ⭐ **核心函数**

```python
def preprocess_full_prompt(examples, tokenizer, max_length=4096):
    input_ids, attention_masks, labels = [], [], []

    for i in range(len(examples["prompt"])):
        # 步骤 1: 构建 prompt 和 response
        prompt = examples["prompt"][i].strip()
        response = examples["response"][i].strip()
        full_text = f"{prompt}\n\n### Response:\n{response}"

        # 步骤 2: Tokenize 完整文本
        encoding = tokenizer(
            full_text,
            truncation=True,
            max_length=max_length,
            padding="max_length"
        )

        # 步骤 3: 计算 prompt 长度
        prompt_ids = tokenizer(
            prompt,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False
        )["input_ids"]
        prompt_len = min(len(prompt_ids), max_length)

        # 步骤 4: 构造 labels (关键!)
        label_ids = encoding["input_ids"].copy()
        label_ids[:prompt_len] = [-100] * prompt_len  # Mask prompt 部分

        input_ids.append(encoding["input_ids"])
        attention_masks.append(encoding["attention_mask"])
        labels.append(label_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "labels": labels
    }
```

**关键机制 - Label Masking:**

```
原始 tokens:  [instruction_tokens, input_tokens, response_tokens]
                ←───────── prompt 部分 ─────────→  ← response →

Labels:       [-100, -100, ..., -100, response_label_1, response_label_2, ...]
               ↑ prompt 部分 (被 mask) ↑        ↑ response 部分 (参与计算) ↑
```

**为什么使用 -100?**

| 标记值 | 含义 | 作用 |
|-------|------|------|
| `-100` | PyTorch 的特殊忽略标记 | 计算损失时自动跳过这些位置 |
| 正整数 | token 的 label | 参与损失计算 |

**核心思想:**
- **只对 response 部分计算损失**
- 让模型专注于学习"生成审稿内容"的能力
- 避免浪费计算资源在理解 prompt 上

---

#### 2.5 量化配置 - `build_bnb_config()`

```python
def build_bnb_config(config):
    return BitsAndBytesConfig(
        load_in_4bit=config.get("load_in_4bit", True),
        bnb_4bit_compute_dtype=getattr(torch, config.get("bnb_4bit_compute_dtype", "bfloat16")),
        bnb_4bit_use_double_quant=config.get("bnb_4bit_use_double_quant", True)
    )
```

**量化参数详解:**

| 参数 | 默认值 | 作用 |
|------|-------|------|
| `load_in_4bit` | `True` | 启用 4-bit 量化 |
| `bnb_4bit_compute_dtype` | `bfloat16` | 计算时的数据类型 |
| `bnb_4bit_use_double_quant` | `True` | 双重量化，进一步节省显存 |

**量化策略:**

```
FP16 权重: 16 bits per parameter
    ↓ 4-bit 量化
FP4 权重: 4 bits per parameter (4x 压缩)
    ↓ 双重量化
量化参数也量化 (再压缩)
```

**显存节省:**
```
原始模型 (FP16): 80 亿参数 × 2 bytes = 160GB
4-bit 量化: 80 亿参数 × 0.5 bytes = 40GB
双重量化: 额外节省 ~5-10GB
总显存: ~35-45GB (H100 可承载)
```

---

#### 2.6 LoRA 配置 - `build_lora_config()`

```python
def build_lora_config(config):
    return LoraConfig(
        r=config.get("lora_rank", 8),
        lora_alpha=config.get("lora_alpha", 32),
        lora_dropout=config.get("lora_dropout", 0.05),
        bias=config.get("lora_bias", "none"),
        task_type="CAUSAL_LM"
    )
```

**LoRA 参数详解:**

| 参数 | 默认值 | 含义 | 计算 |
|------|-------|------|------|
| `r` | 8 | Low-rank 维度 | LoRA 矩阵的秩 |
| `lora_alpha` | 32 | 缩放因子 | α/r = 32/8 = 4 |
| `lora_dropout` | 0.05 | Dropout 比例 | 防止过拟合 |
| `bias` | `"none"` | Bias 训练策略 | 不训练 bias |
| `task_type` | `"CAUSAL_LM"` | 任务类型 | 因果语言模型 |

**LoRA 原理:**

```
原始线性层: y = Wx

添加 LoRA:   y = Wx + BAx
                    ↓
                 LoRA adapter (低秩矩阵)
                 B: d×r, A: r×d
                 r << d (通常 r=8, d=4096)
```

**参数量对比:**
```
原始权重: 4096 × 4096 = 16,777,216 参数
LoRA 权重: 4096 × 8 + 8 × 4096 = 65,536 参数
节省比例: 0.39% (仅训练原参数量的 0.4%)
```

---

#### 2.7 训练参数配置 - `build_training_args()`

```python
def build_training_args(config, output_dir):
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config.get("per_device_train_batch_size", 1),
        per_device_eval_batch_size=config.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
        num_train_epochs=config.get("num_train_epochs", 3),
        eval_strategy=config.get("eval_strategy", "steps"),
        eval_steps=config.get("eval_steps", 100),
        save_steps=config.get("save_steps", 100),
        learning_rate=float(config.get("learning_rate", 2e-5)),
        bf16=config.get("bf16", True),
        logging_steps=config.get("logging_steps", 10),
        save_total_limit=config.get("save_total_limit", 2),
        logging_dir=os.path.join(output_dir, "logs"),
        report_to=config.get("report_to", "none")
    )
```

**关键参数详解:**

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `per_device_train_batch_size` | 1 | 每个设备的训练 batch size |
| `gradient_accumulation_steps` | 4 | 梯度累积步数 |
| `num_train_epochs` | 3 | 训练轮数 |
| `learning_rate` | 2e-5 | 学习率 |
| `bf16` | True | 启用 bfloat16 混合精度 |
| `eval_steps` | 100 | 每 100 步评估一次 |
| `save_steps` | 100 | 每 100 步保存一次 |
| `logging_steps` | 10 | 每 10 步记录一次日志 |

**有效 Batch Size:**
```
effective_batch_size = per_device_train_batch_size × gradient_accumulation_steps × num_gpus
                     = 1 × 4 × 1 = 4
```

---

### 阶段 3: 主函数 `main()` 执行流程

#### Step 1: 解析命令行参数

```python
parser = argparse.ArgumentParser()
parser.add_argument("--config_file", type=str, default="qlora_train_config.yaml")
args = parser.parse_args()
```

**使用方法:**
```bash
# 使用默认配置文件
python train_full_context.py

# 指定自定义配置文件
python train_full_context.py --config_file my_config.yaml
```

---

#### Step 2: 加载配置

```python
config = load_config(args.config_file)
model_path = config["model_name_or_path"]
output_dir = config["output_dir"]
cutoff_len = config.get("cutoff_len", 4096)
```

**配置加载流程:**
```
YAML 文件 → 解析为字典 → 提取关键参数
```

---

#### Step 3: 加载 Tokenizer

```python
print(f"🚀 Loading tokenizer and model from: {model_path}")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
```

**关键配置:**
- `trust_remote_code=True`: 信任远程代码（LLaMA 需要）
- `use_fast=True`: 使用 fast tokenizer（性能更好）
- `pad_token = eos_token`: 将 pad_token 设为 eos_token

**为什么设置 pad_token = eos_token?**

LLaMA 系列 tokenizer 默认没有 pad_token，需要手动设置：
```python
# LLaMA 的特殊 token
bos_token_id = 128000
eos_token_id = 128001
pad_token_id = None  # 默认没有

# 设置后
tokenizer.pad_token = tokenizer.eos_token
pad_token_id = 128001  # 与 eos_token 相同
```

---

#### Step 4: 加载并量化模型

```python
# 4.1 构建量化配置
bnb_config = build_bnb_config(config)

# 4.2 加载模型 (4-bit 量化)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    low_cpu_mem_usage=True,
    torch_dtype=torch.bfloat16
)

# 4.3 准备 k-bit 训练
model = prepare_model_for_kbit_training(model)
```

**加载参数详解:**

| 参数 | 作用 |
|------|------|
| `quantization_config` | 量化配置（4-bit） |
| `low_cpu_mem_usage=True` | 降低 CPU 内存占用 |
| `torch_dtype=torch.bfloat16` | 模型权重类型 |

**`prepare_model_for_kbit_training()` 的作用:**

```python
# 内部执行以下操作:
1. 冻结量化权重 (requires_grad=False)
2. 为 LoRA adapter 添加可训练层
3. 转换为 k-bit 训练模式
4. 启用梯度检查点支持
```

---

#### Step 5: 应用 LoRA 配置

```python
print("🔧 Applying LoRA configuration...")
lora_config = build_lora_config(config)
model = get_peft_model(model, lora_config)
model.config.use_cache = False
model.gradient_checkpointing_enable()
```

**模型结构变化:**

```
原始模型:
[Layer1: W1] → [Layer2: W2] → [Layer3: W3] → ...
   ↓            ↓            ↓
(全量训练)  (全量训练)  (全量训练)

添加 LoRA 后:
[Layer1: W1 + LoRA1] → [Layer2: W2 + LoRA2] → [Layer3: W3 + LoRA3] → ...
   ↓                    ↓                    ↓
(冻结)             (冻结)             (冻结)
   ↑ 可训练            ↑ 可训练            ↑ 可训练
(仅 0.4% 参数)     (仅 0.4% 参数)     (仅 0.4% 参数)
```

**优化配置:**
- `use_cache = False`: 训练时禁用 KV cache（节省显存）
- `gradient_checkpointing_enable()`: 启用梯度检查点

---

#### Step 6: 加载数据集

```python
print("📊 Processing training and validation datasets...")
train_dataset = load_dataset(config["hf_dataset_repo"], "qlora_train")["train"].map(build_prompt)
val_dataset = load_dataset(config["hf_dataset_repo"], "qlora_validation")["train"].map(build_prompt)
```

**数据集信息:**

| 数据集 | 样本数 | 来源 |
|-------|-------|------|
| `qlora_train` | 6,410 | 训练集 |
| `qlora_validation` | 337 | 验证集 |

**数据流:**
```
HF Dataset (instruction, input, output)
    ↓
build_prompt()
    ↓
{"prompt": "...", "response": "..."}
```

---

#### Step 7: Tokenize 数据

```python
print("✂️ Tokenizing datasets...")
train_dataset = train_dataset.map(
    lambda examples: preprocess_full_prompt(examples, tokenizer, max_length=cutoff_len),
    batched=True,
    remove_columns=train_dataset.column_names
)
val_dataset = val_dataset.map(
    lambda examples: preprocess_full_prompt(examples, tokenizer, max_length=cutoff_len),
    batched=True,
    remove_columns=val_dataset.column_names
)

train_dataset.set_format(type="torch")
val_dataset.set_format(type="torch")
```

**处理流程:**

```
原始格式:
{
  "prompt": "...",
  "response": "..."
}
    ↓
preprocess_full_prompt()
    ↓
Tensor 格式:
{
  "input_ids": [128000, 1234, 5678, ..., 128001],
  "attention_mask": [1, 1, 1, ..., 1],
  "labels": [-100, -100, ..., response_token_ids...]
}
```

**关键点:**
- `batched=True`: 批量处理，提高效率
- `remove_columns`: 移除原始列，只保留 tokenized 结果
- `set_format("torch")`: 转换为 PyTorch Tensor 格式

---

#### Step 8: 构建训练器

```python
# 8.1 构建训练参数
training_args = build_training_args(config, output_dir)

# 8.2 数据整理器
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 8.3 创建 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator
)
```

**DataCollator 详解:**

```python
DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
```

| 参数 | 作用 |
|------|------|
| `tokenizer` | 分词器 |
| `mlm=False` | 不是 Masked Language Model（MLM），而是 Causal LM |

**功能:**
- 动态 padding（每个 batch 内 padding 到最长序列）
- 创建 attention_mask
- 处理特殊 token

---

#### Step 9: 执行训练

```python
print("🏋️ Starting training...")
trainer.train()
```

**训练循环:**

```
for epoch in range(num_train_epochs):
    for step, batch in enumerate(dataloader):
        # 1. Forward pass
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        loss = outputs.loss

        # 2. Backward pass
        loss.backward()

        # 3. Gradient accumulation
        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        # 4. Logging
        if step % logging_steps == 0:
            logger.info(f"Step {step}: loss={loss.item():.4f}")

        # 5. Evaluation
        if step % eval_steps == 0:
            eval_results = trainer.evaluate()
            logger.info(f"Eval: {eval_results}")

        # 6. Save checkpoint
        if step % save_steps == 0:
            trainer.save_checkpoint(f"checkpoint-step-{step}")
```

**损失计算:**
```python
loss = CrossEntropyLoss(
    logits,      # [batch_size, seq_len, vocab_size]
    labels       # [batch_size, seq_len] (含 -100)
)
# -100 位置不参与损失计算
```

---

#### Step 10: 保存模型

```python
print(f"💾 Saving model and tokenizer to {output_dir}")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print("✅ Training complete.")
```

**保存内容:**

```
output_dir/
├── adapter_config.json          # LoRA 配置文件
├── adapter_model.safetensors   # LoRA 权重 (可训练参数)
├── tokenizer_config.json        # Tokenizer 配置
├── tokenizer.json              # Tokenizer 词汇表
├── special_tokens_map.json     # 特殊 token 映射
└── logs/                        # 训练日志
    ├── trainer_log.jsonl
    └── events.out.tfevents...
```

**文件说明:**

| 文件 | 内容 | 大小 |
|------|------|------|
| `adapter_config.json` | LoRA 超参数 (r, alpha, dropout 等) | ~1 KB |
| `adapter_model.safetensors` | LoRA 权重矩阵 | ~160 MB |
| `tokenizer.json` | 词汇表 + 合并规则 | ~10 MB |
| `tokenizer_config.json` | Tokenizer 配置 | ~5 KB |

---

## 数据流转图

```
┌─────────────────────────────────────────────────────────────┐
│                    原始数据 (HF Dataset)                    │
├─────────────────────────────────────────────────────────────┤
│ {                                                           │
│   "instruction": "Review this paper in detail.",            │
│   "input": "[论文全文 ~30,000 字符...]",                    │
│   "output": "[审稿内容 ~2,500 字符...]"                     │
│ }                                                           │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    build_prompt()                           │
├─────────────────────────────────────────────────────────────┤
│ {                                                           │
│   "prompt": "Review this paper in detail.\n\n[论文全文]",    │
│   "response": "[审稿内容]"                                  │
│ }                                                           │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              preprocess_full_prompt()                       │
├─────────────────────────────────────────────────────────────┤
│ {                                                           │
│   "input_ids": [128000, 1234, 5678, ..., 128001],         │
│   "attention_mask": [1, 1, 1, ..., 1],                      │
│   "labels": [-100, -100, ..., 2345, 6789, ...]             │
│ }                                                           │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                     DataLoader (batching)                   │
├─────────────────────────────────────────────────────────────┤
│ Batch:                                                      │
│   input_ids: [batch_size, seq_len]                         │
│   attention_mask: [batch_size, seq_len]                    │
│   labels: [batch_size, seq_len]                             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      Model Forward                           │
├─────────────────────────────────────────────────────────────┤
│ logits = model(input_ids, attention_mask)                  │
│ loss = CrossEntropyLoss(logits, labels)                    │
│ # 注意: labels 中 -100 位置不参与损失计算                  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                 Backward & Optimizer Step                   │
├─────────────────────────────────────────────────────────────┤
│ loss.backward()                                             │
│ optimizer.step()  # 仅更新 LoRA 参数                        │
│ optimizer.zero_grad()                                       │
└─────────────────────────────────────────────────────────────┘
```

---

## 关键技术点详解

### 1. Label Masking 策略

**代码:**
```python
label_ids = encoding["input_ids"].copy()
label_ids[:prompt_len] = [-100] * prompt_len
```

**目的:**
- 只训练生成能力，不训练理解能力
- 让模型专注于学会"根据 prompt 生成 response"

**原理:**
```python
# PyTorch CrossEntropyLoss 内部实现
loss = 0
for i in range(seq_len):
    if labels[i] != -100:  # 跳过 -100
        loss += -log(softmax(logits[i])[labels[i]])
loss = loss / num_valid_positions
```

**效果对比:**

| 训练策略 | Prompt 损失 | Response 损失 | 训练重点 |
|---------|------------|--------------|---------|
| 完整训练 | ✓ | ✓ | 理解 + 生成 |
| Label Masking | ✗ (mask) | ✓ | 仅生成 |

---

### 2. 4-bit QLoRA 量化

**显存占用分析:**

```
原始模型 (FP16):
  参数量: 8,000,000,000
  每参数: 2 bytes
  总显存: 16 GB

4-bit 量化:
  参数量: 8,000,000,000
  每参数: 0.5 bytes
  量化显存: 4 GB

LoRA Adapter (FP16):
  参数量: ~80,000,000 (1%)
  每参数: 2 bytes
  LoRA 显存: 160 MB

激活显存 (训练时):
  Batch=1, Seq=18000
  ~20-30 GB

总显存: 4 GB (量化) + 160 MB (LoRA) + 30 GB (激活) ≈ 35 GB
```

**量化流程:**

```
FP16 权重
    ↓
NormalFloat4 (NF4) 量化
    ↓
计算时反量化为 bfloat16
    ↓
更新时重新量化为 NF4
```

**NF4 量化特点:**
- 基于正态分布的量化
- 保持权重分布的统计特性
- 比均匀量化效果更好

---

### 3. 梯度累积

**代码:**
```python
per_device_train_batch_size=1
gradient_accumulation_steps=4
```

**计算流程:**

```
Step 1:  batch_size=1, 计算梯度，不更新
Step 2:  batch_size=1, 累加梯度
Step 3:  batch_size=1, 累加梯度
Step 4:  batch_size=1, 累加梯度
         → 总梯度 = grad_1 + grad_2 + grad_3 + grad_4
         → optimizer.step()
         → optimizer.zero_grad()

有效 batch size = 1 × 4 = 4
```

**优势:**
- 在显存受限时模拟大 batch size
- 提高训练稳定性
- 不增加显存占用

---

### 4. 梯度检查点

**代码:**
```python
model.gradient_checkpointing_enable()
```

**原理:**

```
正常前向传播 (保存所有激活):
  Layer1 → Layer2 → Layer3 → Layer4 → output
    ↓        ↓        ↓        ↓
  Act1     Act2     Act3     Act4  (保存到显存)

反向传播 (使用保存的激活):
  Grad_out ← Grad3 ← Grad2 ← Grad1
     ↑        ↑        ↑        ↑
   Act4     Act3     Act2     Act1  (从显存读取)

显存占用: 所有激活 = ~30 GB
```

```
梯度检查点 (不保存激活):
  Layer1 → Layer2 → Layer3 → Layer4 → output
            ↓        ↓        ↓
          (只保存 Act2, Act4)

反向传播:
  Grad_out ← Grad3 ← (重新计算) ← Grad1
               ↑           ↑
             Act4        (重新计算)
            ↓
           (重新计算) ← Act2
               ↑
             (重新计算)

显存占用: 部分激活 = ~12 GB
计算时间: 增加 ~30% (需要重新计算)
```

**权衡:**
- **显存节省**: ~60%
- **时间成本**: +30%

---

### 5. 混合精度训练 (BF16)

**代码:**
```python
bf16=True  # 使用 bfloat16
```

**数据类型对比:**

| 类型 | 位宽 | 范围 | 精度 | 显存 |
|------|------|------|------|------|
| FP32 | 32 bits | ±3.4e38 | 高 | 4 bytes |
| FP16 | 16 bits | ±6.5e4 | 中 | 2 bytes |
| BF16 | 16 bits | ±3.4e38 | 中 | 2 bytes |

**BF16 vs FP16:**

```
BF16: 1 bit 符号 + 8 bit 指数 + 7 bit 尾数
      ↑ FP32 指数范围  ↑ FP32 精度降低

FP16: 1 bit 符号 + 5 bit 指数 + 10 bit 尾数
      ↑ 小范围      ↑ 更高精度

优势:
- BF16 不容易溢出 (指数范围大)
- 适合 LLM 训练 (梯度数值范围大)
```

---

## 训练超参数建议

### 基础配置

| 参数 | 建议值 | 说明 |
|------|-------|------|
| `lora_rank` | 8-16 | 更高 rank = 更多参数，但未必更好 |
| `lora_alpha` | 32-64 | alpha/r 通常在 4-8 之间 |
| `lora_dropout` | 0.05-0.1 | 防止过拟合 |
| `learning_rate` | 1e-5 - 5e-5 | LoRA 通常用较低学习率 |

### 训练配置

| 参数 | 建议值 | 说明 |
|------|-------|------|
| `per_device_train_batch_size` | 1-2 | 根据显存调整 |
| `gradient_accumulation_steps` | 4-8 | 有效 batch size = 前者 × 后者 |
| `num_train_epochs` | 2-3 | 防止过拟合 |
| `warmup_ratio` | 0.1 | 前 10% 步数 warmup |

### 优化配置

| 参数 | 建议值 | 说明 |
|------|-------|------|
| `bf16` | True | 启用 bfloat16 |
| `gradient_checkpointing` | True | 启用梯度检查点 |
| `dataloader_num_workers` | 4-8 | 数据加载并行 |

---

## 常见问题与解决方案

### 1. CUDA Out of Memory (OOM)

**原因:** 显存不足

**解决方案:**

```python
# 方案 1: 减小 batch size
per_device_train_batch_size: 1  # 改为 1

# 方案 2: 增加梯度累积
gradient_accumulation_steps: 8  # 增加到 8

# 方案 3: 减小序列长度
cutoff_len: 8192  # 从 18000 减到 8192

# 方案 4: 启用梯度检查点 (已启用)
model.gradient_checkpointing_enable()
```

---

### 2. 损失不下降或 NaN

**原因:** 学习率过高或数值不稳定

**解决方案:**

```python
# 方案 1: 降低学习率
learning_rate: 1e-5  # 从 2e-5 降到 1e-5

# 方案 2: 增加 warmup
warmup_ratio: 0.1  # 前 10% 步数线性 warmup

# 方案 3: 检查数据
# 确保没有异常值或空样本
```

---

### 3. 训练速度慢

**原因:** 数据加载或计算瓶颈

**解决方案:**

```python
# 方案 1: 增加数据加载线程
dataloader_num_workers: 8

# 方案 2: 预处理数据并缓存
preprocessing_num_workers: 16
overwrite_cache: false  # 第二次训练使用缓存

# 方案 3: 禁用梯度检查点 (如果显存充足)
# model.gradient_checkpointing_enable()  # 注释掉
```

---

### 4. 模型保存失败

**原因:** 权限或路径问题

**解决方案:**

```bash
# 检查输出目录权限
ls -la output_dir/

# 如果使用 HuggingFace Hub
# 登录
huggingface-cli login
```

---

## 与滑动窗口策略的对比

| 特性 | Full Context | Sliding Window |
|------|-------------|----------------|
| **上下文长度** | ~18,000 tokens | 8,192 tokens |
| **显存需求** | H100 (80GB) | 4090 (24GB) |
| **训练数据量** | 原始数据 | 2-3x 扩展 |
| **处理方式** | 一次性处理 | 分段处理 |
| **长距离依赖** | 保留完整 | 可能丢失 |
| **显存占用** | 高 | 低 |
| **适用场景** | 全局理解任务 | 局部理解任务 |
| **训练时间** | 较短 | 较长 (数据量更大) |
| **泛化能力** | 较好 | 意外的好 (数据增强效应) |

---

## 使用示例

### 基础使用

```bash
# 1. 准备配置文件
cat > train_config.yaml <<EOF
model_name_or_path: "meta-llama/Llama-3.1-8B-Instruct"
hf_dataset_repo: "guochenmeinian/openreview_dataset"
output_dir: "../model/full_context_qlora"
cutoff_len: 18000
lora_rank: 8
per_device_train_batch_size: 1
gradient_accumulation_steps: 4
num_train_epochs: 2
learning_rate: 0.00002
EOF

# 2. 执行训练
cd train/code
python train_full_context.py --config_file train_config.yaml
```

---

### 监控训练

```bash
# 查看实时日志
tail -f ../model/full_context_qlora/logs/trainer_log.jsonl

# 使用 TensorBoard
tensorboard --logdir ../model/full_context_qlora/logs
```

---

### 恢复训练

```bash
# 训练会自动从最新 checkpoint 恢复
python train_full_context.py --config_file train_config.yaml
```

---

## 总结

`train_full_context.py` 实现了完整的 **QLoRA 微调流程**，核心特点：

### 优势

1. **内存高效**: 4-bit 量化 + LoRA + 梯度检查点
2. **数据智能**: 只对 response 计算损失
3. **配置灵活**: YAML 配置文件集中管理
4. **生产就绪**: 日志、检查点、评估完整

### 技术亮点

- **Label Masking**: 只训练生成能力
- **QLoRA**: 4-bit 量化 + 低秩适配器
- **梯度累积**: 模拟大 batch size
- **混合精度**: BF16 训练
- **梯度检查点**: 节省显存

### 适用场景

- 需要理解完整论文上下文的任务
- 有 H100 GPU 资源
- 对长距离依赖要求高的场景

---

## 参考资源

- [LLaMA 3.1 Paper](https://ai.meta.com/research/publications/the-llama-3-herd-of-models/)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
