# mini-transformer

手写 Mini-Transformer 语言模型，从零实现 BPE 分词、模型架构、训练、推理。

基于 [Datawhale DIY-LLM](https://github.com/datawhalechina/diy-llm) 课程的 Assignment 1 学习笔记。

## 文件结构

```
mini-transformer/
├── model.py              # Transformer 模型定义 + 推理（200+ 行纯手写）
├── train.py              # 训练脚本
└── bpe_tokenizer/
    └── tokenizer.json    # BPE 分词器（50257 词表，50074 合并规则）
```

## 模型架构

- **Token Embedding** — 手写 Embedding 层
- **RMSNorm** — Root Mean Square Layer Normalization
- **RoPE** — 旋转位置编码
- **Multi-Head Attention** — 手写多头注意力（Flash Attention 可选）
- **SwiGLU** — Swish + Gated Linear Unit 前馈网络
- **TransformerBlock** — Pre-Norm 残差结构
- **TransformerLM** — 完整语言模型 + top-k/top-p 采样生成
- **CustomAdamW** — 手写优化器

## 使用

```bash
# 训练
python train.py

# 推理
python model.py
```