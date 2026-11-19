# NTHU-LLM-Pretrain-Lab06

一個**純 PyTorch、可在 CPU 上跑**的小型 GPT pretraining 示範。從零開始建立 tokenizer、模型、訓練迴圈與生成腳本，不依賴 HuggingFace / CUDA / 網路下載資料。

## 檔案

| 檔案 | 說明 |
| --- | --- |
| `model.py` | Decoder-only Transformer（token/positional embedding、causal self-attention、LayerNorm、MLP、weight tying） |
| `train.py` | 字元級 tokenizer + 訓練迴圈（AdamW、warmup + cosine decay、gradient clipping、週期性 eval / sample、checkpoint） |
| `generate.py` | 載入 checkpoint 做 autoregressive 生成（temperature / top-k） |
| `data/input.txt` | 內建原創英文小語料（約 7.8K 字元，58 個字元）；也可換成任何 UTF-8 `.txt` |
| `requirements.txt` | 只依賴 PyTorch |

## 安裝

建議安裝 CPU 版 wheel（約 200 MB；若直接 `pip install torch` 會拉進 CUDA 依賴，體積大很多）：

```bash
python3 -m pip install --index-url https://download.pytorch.org/whl/cpu -r requirements.txt
```

已在本機以 Python 3.14 + `torch 2.15.0+cpu` 驗證。

## 快速開始

```bash
# 約 1~2 分鐘的 smoke test：loss 從 ~4.09 降到 ~3.5
python train.py --max-iters 200 --eval-interval 50 --sample-interval 100

# 稍完整的示範（預設 1500 iters；約 5~10 分鐘，視 CPU 而定）
python train.py
```

訓練過程會印出 train / val loss，並定期讓模型生成一段文字。結束後 checkpoint 存到 `out/model.pt`。

本機實測（CPU，預設 617K 參數）：

| steps | 耗時 | train loss | val loss |
| --- | --- | --- | --- |
| 0 | — | 4.09 | 4.08 |
| 100 | ~26s | 2.50 | 2.57 |
| 500 | ~2min | 2.19 | 2.33 |
| 1500 | ~5.5min | 1.44 | 2.21 |

1500 步後 train/val 差距變大是**過擬合**（語料只有 7.8K 字元），属正常現象；想改善可換更大的語料或提高 dropout。本機執行完會在 `out/model.pt` 留下訓練好的 checkpoint（2.6 MB，已被 `.gitignore` 忽略，不會進版本控制），可直接執行：

```bash
python generate.py --prompt "The small model" --tokens 300 --temperature 0.7 --top-k 30
```

生成文字：

```bash
python generate.py --prompt "The small model" --tokens 300 --temperature 0.8 --top-k 40
```

## 常用參數

```bash
python train.py \
  --data data/input.txt \
  --out-dir out \
  --device cpu \
  --block-size 128 \
  --n-layer 3 --n-head 4 --n-embd 128 \
  --batch-size 16 --max-iters 1500 \
  --learning-rate 3e-4 --warmup-iters 100 \
  --eval-interval 200 --sample-interval 500
```

- `--threads N`：限制 PyTorch 的 CPU thread 數（預設由 PyTorch 決定）。
- `--sample-interval 0`：關閉訓練中生成。
- 換語料：把自己的 UTF-8 文字放到 `--data` 即可，tokenizer 會自動從語料建立。注意 prompt 裡的字元必須都出現在訓練語料中。

## 模型規模

預設約 **61.7 萬參數**（vocab 58、block_size 128、3 層、4 heads、embedding 128）。想更快可下調 `--n-layer 2 --n-embd 96 --n-head 3 --batch-size 8`；想讓生成更通順可增加 `n-layer / n-embd / max-iters` 或換更大的語料。

## 換成中文語料

程式是字元級 tokenizer，可直接用中文；只要把 `data/input.txt` 換成中文文字即可（詞表會變大， emb/head 参数量略增）。生成時的 `--prompt` 也必須完全由語料中出現過的字組成。

## 常見問答

- 啟動時出現 `UserWarning: Failed to initialize NumPy: No module named 'numpy'`：這是 PyTorch 在沒有安裝 NumPy 時的無害警告，不影響訓練與生成。想消除可 `pip install numpy`。
- 生成出來的文字是亂碼：訓練步數還不夠。先跑 500~1500 iters 再看；也可降低 temperature（如 0.6）。

## 注意

這是一個教学用的玩具模型：它沒有世界知識、記憶長度約一個 context window，流利的句子不代表真的理解。真正的 pretraining 需要更大的模型、資料與運算資源。
