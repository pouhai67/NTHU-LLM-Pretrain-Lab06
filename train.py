"""Train a tiny character-level GPT on CPU.

Example (quick smoke test, ~1 minute on a laptop):
    python train.py --max-iters 200 --eval-interval 50 --sample-interval 100

Example (slightly longer run):
    python train.py --max-iters 3000 --batch-size 16 --n-layer 4 --n-head 4 --n-embd 192
"""

from __future__ import annotations

import argparse
import math
import os
import time

import torch

from model import GPT, GPTConfig


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a tiny GPT on CPU")
    p.add_argument("--data", type=str, default="data/input.txt", help="Path to a UTF-8 text corpus")
    p.add_argument("--out-dir", type=str, default="out", help="Where to save checkpoints")
    p.add_argument("--device", type=str, default="cpu", help="'cpu' (default) or 'cuda' if available")
    p.add_argument("--seed", type=int, default=1337)

    # Model architecture
    p.add_argument("--block-size", type=int, default=128, help="Context length")
    p.add_argument("--n-layer", type=int, default=3)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-embd", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)

    # Optimisation
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-iters", type=int, default=1500)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=3e-5)
    p.add_argument("--warmup-iters", type=int, default=100)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)

    # Logging / evaluation
    p.add_argument("--eval-interval", type=int, default=200)
    p.add_argument("--eval-iters", type=int, default=20)
    p.add_argument("--sample-interval", type=int, default=500, help="0 disables sampling")
    p.add_argument("--sample-tokens", type=int, default=300)
    p.add_argument("--threads", type=int, default=0, help="CPU threads (0 = let PyTorch decide)")
    return p.parse_args()


def load_corpus(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if len(text) < 100:
        raise ValueError(f"Corpus {path!r} is too short ({len(text)} chars).")
    return text


def get_batch(
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a batch of (input, target) sequences from a 1-D tensor of token ids."""
    ix = torch.randint(
        len(data) - block_size - 1, (batch_size,), generator=generator
    )
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(
    model: GPT,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    args: argparse.Namespace,
    generator: torch.Generator,
) -> dict[str, float]:
    out: dict[str, float] = {}
    model.eval()
    for split, data in (("train", train_data), ("val", val_data)):
        losses = torch.zeros(args.eval_iters)
        for i in range(args.eval_iters):
            xb, yb = get_batch(data, args.block_size, args.batch_size, args.device, generator)
            _, loss = model(xb, yb)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def learning_rate(step: int, args: argparse.Namespace) -> float:
    """Linear warmup followed by cosine decay down to min_lr."""
    if step < args.warmup_iters:
        return args.learning_rate * (step + 1) / args.warmup_iters
    decay_steps = max(1, args.max_iters - args.warmup_iters)
    ratio = (step - args.warmup_iters) / decay_steps
    ratio = min(1.0, ratio)
    coef = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return args.min_lr + coef * (args.learning_rate - args.min_lr)


def main() -> None:
    args = get_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available; use --device cpu")

    # ---- Data: build a simple character-level tokenizer from the corpus ----
    text = load_corpus(args.data)
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    encode = lambda s: [stoi[c] for c in s]  # noqa: E731
    decode = lambda ids: "".join([itos[i] for i in ids])  # noqa: E731

    data = torch.tensor(encode(text), dtype=torch.long)
    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]
    if min(len(train_data), len(val_data)) <= args.block_size + 1:
        raise ValueError(
            "Train/val split is too small for --block-size; use a longer corpus or a smaller block size."
        )

    print(f"Corpus: {len(text):,} chars | vocab: {len(chars)} | device: {device}")
    print(f"Train tokens: {len(train_data):,} | val tokens: {len(val_data):,}")

    # ---- Model + optimiser ----
    config = GPTConfig(
        block_size=args.block_size,
        vocab_size=len(chars),
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = GPT(config).to(device)
    print(f"Parameters: {model.num_params():,}")

    # Decay only 2-D weights (embeddings and linear layers), as is common practice.
    decay_params = [p for p in model.parameters() if p.dim() >= 2]
    nodecay_params = [p for p in model.parameters() if p.dim() < 2]
    optimiser = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=args.learning_rate,
    )

    batch_generator = torch.Generator().manual_seed(args.seed + 1)
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()
    model.train()

    # ---- Training loop ----
    for step in range(args.max_iters):
        lr = learning_rate(step, args)
        for group in optimiser.param_groups:
            group["lr"] = lr

        xb, yb = get_batch(train_data, args.block_size, args.batch_size, device, batch_generator)
        _, loss = model(xb, yb)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimiser.step()

        if step % 10 == 0 or step == args.max_iters - 1:
            elapsed = time.time() - t0
            print(
                f"step {step:5d}/{args.max_iters} | lr {lr:.2e} "
                f"| loss {loss.item():.4f} | {elapsed:.1f}s",
                flush=True,
            )

        if args.eval_interval > 0 and (step % args.eval_interval == 0 or step == args.max_iters - 1):
            losses = estimate_loss(model, train_data, val_data, args, batch_generator)
            print(
                f"  eval @ {step}: train {losses['train']:.4f} | val {losses['val']:.4f}",
                flush=True,
            )

        if args.sample_interval > 0 and step > 0 and step % args.sample_interval == 0:
            context = torch.zeros((1, 1), dtype=torch.long, device=device)
            sample = model.generate(context, args.sample_tokens, temperature=0.8, top_k=40)
            print("  --- sample ---")
            print("  " + decode(sample[0].tolist()).replace("\n", "\n  "))
            print("  --------------", flush=True)
            model.train()

    # ---- Save checkpoint (model + tokenizer + hyper-parameters) ----
    ckpt_path = os.path.join(args.out_dir, "model.pt")
    torch.save(
        {
            "model": model.state_dict(),
            "config": {
                "block_size": config.block_size,
                "vocab_size": config.vocab_size,
                "n_layer": config.n_layer,
                "n_head": config.n_head,
                "n_embd": config.n_embd,
                "dropout": config.dropout,
            },
            "stoi": stoi,
            "itos": itos,
            "step": args.max_iters,
        },
        ckpt_path,
    )
    print(f"Saved checkpoint to {ckpt_path} ({(time.time() - t0):.1f}s total)")


if __name__ == "__main__":
    main()
