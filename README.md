# mini-transformer

涓€涓粠闆跺疄鐜扮殑 Mini GPT / Decoder-only Transformer 瀛︿範椤圭洰銆傞」鐩洰鏍囦笉鏄缁冨彲鐢ㄧ殑澶фā鍨嬶紝鑰屾槸鎶婅瑷€妯″瀷鐨勬牳蹇冨伐绋嬮摼璺畬鏁磋窇閫氾細

```text
鏂囨湰璇枡 -> BPE tokenizer -> token ids -> data.bin -> TransformerLM 璁粌 -> checkpoint -> 鎺ㄧ悊鐢熸垚 -> 鍗曞厓娴嬭瘯 -> 鎬ц兘娴嬭瘯
```

鏈」鐩熀浜?Datawhale DIY-LLM Assignment 1 杩涜鏁寸悊锛屽苟琛ュ厖浜嗘暟鎹噯澶囪剼鏈€佹帹鐞嗚剼鏈€佹祴璇曠敤渚嬨€佹€ц兘娴嬭瘯鑴氭湰鍜屽彲澶嶇幇璁粌鍛戒护锛屼娇鍏朵粠鈥滆绋嬩唬鐮佲€濆彉鎴愪竴涓彲浠ユ湰鍦拌繍琛屽拰灞曠ず鐨勫皬鍨嬪伐绋嬮」鐩€?
## Features

- 鎵嬪啓 `Linear`銆乣Embedding`
- 鎵嬪啓 `RMSNorm`
- 鎵嬪啓 RoPE 鏃嬭浆浣嶇疆缂栫爜
- 鎵嬪啓 Multi-Head Causal Self-Attention
- 鎵嬪啓 `SwiGLU` 鍓嶉缃戠粶
- 鎵嬪啓 `TransformerBlock` 鍜?`TransformerLM`
- 鎵嬪啓 `CustomAdamW` 浼樺寲鍣?- 鏀寔 next-token prediction 璁粌
- 鏀寔 cosine learning rate schedule 鍜?warmup
- 鏀寔 top-k / top-p sampling 鏂囨湰鐢熸垚
- 鎻愪緵 pytest 鍗曞厓娴嬭瘯
- 鎻愪緵 inference benchmark

## Project Structure

```text
mini-transformer/
鈹溾攢鈹€ model.py                         # 妯″瀷缁撴瀯銆丷oPE銆丄ttention銆丼wiGLU銆佺敓鎴愬嚱鏁?鈹溾攢鈹€ train.py                         # 璁粌娴佺▼銆丏ataset銆佷紭鍖栧櫒銆乧heckpoint銆丳PL 鏇茬嚎
鈹溾攢鈹€ prepare_data.py                  # 鏂囨湰璇枡 -> token ids -> data.bin
鈹溾攢鈹€ infer.py                         # 鍔犺浇 checkpoint 杩涜鏂囨湰鐢熸垚
鈹溾攢鈹€ requirements.txt                 # Python 渚濊禆
鈹溾攢鈹€ README.md
鈹溾攢鈹€ bpe_tokenizer/
鈹?  鈹斺攢鈹€ tokenizer.json               # BPE tokenizer 璇嶈〃鍜?merge 瑙勫垯
鈹溾攢鈹€ data/
鈹?  鈹斺攢鈹€ tiny_corpus.txt              # smoke test 灏忚鏂?鈹溾攢鈹€ tests/
鈹?  鈹溾攢鈹€ test_model.py                # 娴嬭瘯妯″瀷 forward 鍜岀敓鎴愬嚱鏁?鈹?  鈹溾攢鈹€ test_dataset.py              # 娴嬭瘯 next-token 鏁版嵁鏋勯€?鈹?  鈹溾攢鈹€ test_train_step.py           # 娴嬭瘯鍗曟璁粌鏄惁鏇存柊鍙傛暟
鈹?  鈹斺攢鈹€ test_checkpoint.py           # 娴嬭瘯 checkpoint 淇濆瓨鍜屽姞杞?鈹斺攢鈹€ benchmarks/
    鈹斺攢鈹€ benchmark_inference.py       # 鎺ㄧ悊鎬ц兘娴嬭瘯
```

## Environment

```bash
pip install -r requirements.txt
```

涓昏渚濊禆锛?
```text
torch
transformers
tokenizers
numpy
pytest
matplotlib
psutil
```

## Quick Start

### 1. Run tests

```bash
python -m pytest tests -q
```

娴嬭瘯瑕嗙洊锛?
- `TransformerLM` 杈撳嚭 logits shape
- 鐢熸垚鍑芥暟鑳藉惁鎵╁睍搴忓垪
- `CausalMemmapDataset` 鏄惁姝ｇ‘鏋勯€?`x/y`
- 鍗曟 `forward -> loss -> backward -> optimizer.step`
- checkpoint 淇濆瓨鍜岄噸鏂板姞杞?
### 2. Prepare demo data

```bash
python prepare_data.py --input data/tiny_corpus.txt --output data/data.bin
```

`prepare_data.py` 浼氳鍙栨枃鏈鏂欙紝浣跨敤 `bpe_tokenizer/tokenizer.json` 缂栫爜鎴?token ids锛屽苟浠?`int32` 浜岃繘鍒舵牸寮忎繚瀛樹负 `data.bin`銆?
### 3. Train a small demo model

```bash
python train.py \
  --data_path data/data.bin \
  --epochs 1 \
  --batch_size 2 \
  --context_length 32 \
  --d_model 64 \
  --num_heads 4 \
  --num_layers 2 \
  --checkpoint_dir ckpt
```

璁粌瀹屾垚鍚庝細鐢熸垚锛?
```text
ckpt/epoch_1.pt
ckpt/train_ppl.png
ckpt/val_ppl.png
```

### 4. Inference

```bash
python infer.py \
  --checkpoint ckpt/epoch_1.pt \
  --prompt "Lily likes" \
  --max_new_tokens 40
```

`infer.py` 浼氫粠 checkpoint 涓鍙栨ā鍨嬬粨鏋勯厤缃紝鍥犳涓嶉渶瑕佹墜鍔ㄩ噸鏂版寚瀹?`d_model`銆乣num_layers`銆乣num_heads` 绛夊弬鏁般€?
### 5. Benchmark

```bash
python benchmarks/benchmark_inference.py
```

璇ヨ剼鏈細杈撳嚭锛?
- 鍙傛暟閲?- forward latency
- generation tokens/s
- CPU/GPU 鍐呭瓨鍗犵敤

## Training With TinyStories

鏈粨搴撲笉鐩存帴淇濆瓨澶у瀷璁粌璇枡銆傝嫢闇€瑕佽缁冩洿澶х殑妯″瀷锛屽彲浠ヤ笅杞?TinyStories锛?
```bash
mkdir -p data/raw
wget -O data/raw/TinyStoriesV2-GPT4-train.txt \
https://hf-mirror.com/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
```

涓轰簡閬垮厤涓€娆℃€у鐞嗗畬鏁?2GB 璇枡瀵艰嚧鍐呭瓨鍘嬪姏杩囧ぇ锛屽缓璁厛鎴彇涓€閮ㄥ垎鏁版嵁锛?
```bash
python - <<'PY'
from pathlib import Path

src = Path("data/raw/TinyStoriesV2-GPT4-train.txt")
dst = Path("data/raw/TinyStories_150MB.txt")
limit = 150 * 1024 * 1024

written = 0
with src.open("r", encoding="utf-8") as f, dst.open("w", encoding="utf-8") as g:
    for line in f:
        b = line.encode("utf-8")
        if written + len(b) > limit:
            break
        g.write(line)
        written += len(b)

print(f"wrote {written / 1024 / 1024:.1f} MB to {dst}")
PY
```

鐒跺悗缂栫爜涓鸿缁冩暟鎹細

```bash
python prepare_data.py \
  --input data/raw/TinyStories_150MB.txt \
  --output data/tinystories_150mb.bin
```

绀轰緥璁粌閰嶇疆锛?
```bash
python train.py \
  --data_path data/tinystories_150mb.bin \
  --epochs 1 \
  --batch_size 1 \
  --context_length 256 \
  --d_model 1024 \
  --num_heads 16 \
  --num_layers 8 \
  --checkpoint_dir ckpt_tinystories_204m_150mb
```

## Model Architecture

妯″瀷鏄竴涓?decoder-only Transformer锛?
```text
input_ids
  -> token embedding
  -> TransformerBlock x N
      -> RMSNorm
      -> Causal Self-Attention + RoPE
      -> residual add
      -> RMSNorm
      -> SwiGLU FFN
      -> residual add
  -> final RMSNorm
  -> LM Head
  -> logits
```

璁粌鐩爣鏄?next-token prediction锛?
```text
x = [t0, t1, t2, ..., t127]
y = [t1, t2, t3, ..., t128]
```

妯″瀷鏍规嵁 `x` 棰勬祴 `y`锛屼娇鐢?`CrossEntropyLoss` 鏇存柊鍙傛暟銆?
## Large Files

GitHub 鏅€氫粨搴撲笉淇濆瓨澶у瀷鏁版嵁鍜屾潈閲嶃€傛湰浠撳簱榛樿蹇界暐锛?
```text
data/raw/
data/*.bin
*.bin
ckpt*/
checkpoints/
*.pt
*.pth
TinyStoriesV2-GPT4-train.txt
TinyStoriesV2-GPT4-valid.txt
```

濡傛灉闇€瑕佸垎浜緝澶х殑 checkpoint锛屽缓璁娇鐢細

- GitHub Releases
- Git LFS
- Hugging Face Hub

## Current Limitations

- `data/tiny_corpus.txt` 鍙敤浜?smoke test锛屼笉鑳借缁冨嚭楂樿川閲忚瑷€妯″瀷銆?- 褰撳墠璁粌鑴氭湰浠ユ暀瀛﹀拰鍙鎬т负涓伙紝娌℃湁鍋氭贩鍚堢簿搴︺€佹搴︾疮绉€佸垎甯冨紡璁粌绛夊伐绋嬩紭鍖栥€?- 澶ц鏂欏拰 checkpoint 涓嶆彁浜ゅ埌 GitHub锛岄渶瑕佹寜 README 鍛戒护鑷涓嬭浇鎴栫敓鎴愩€?- 妯″瀷璐ㄩ噺鍙栧喅浜庤鏂欒妯°€佹ā鍨嬭妯″拰璁粌鏃堕暱锛涙湰椤圭洰閲嶇偣鏄悊瑙ｅ拰澶嶇幇 GPT 璁粌娴佺▼銆?
## References

- Datawhale DIY-LLM Assignment 1
- TinyStories dataset
- GPT / Decoder-only Transformer
- RoPE, RMSNorm, SwiGLU, AdamW