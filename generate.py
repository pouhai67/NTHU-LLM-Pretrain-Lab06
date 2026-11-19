"""Sample text from a checkpoint produced by train.py (CPU by default).

Example:
    python generate.py --prompt "The little llama" --tokens 400
"""

from __future__ import annotations

import argparse

import torch

from model import GPT, GPTConfig


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate text with a trained tiny GPT")
    p.add_argument("--checkpoint", type=str, default="out/model.pt")
    p.add_argument("--prompt", type=str, default="", help="Seed text (empty = start from <BOS>-like token 0)")
    p.add_argument("--tokens", type=int, default=300, help="Number of new characters to generate")
    p.add_argument("--temperature", type=float, default=0.8, help="Lower = more conservative, higher = more random")
    p.add_argument("--top-k", type=int, default=40, help="Keep only the top-k candidates (0 = no filtering)")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = get_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = GPT(GPTConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    stoi = ckpt["stoi"]
    itos = ckpt["itos"]
    encode = lambda s: [stoi[c] for c in s]  # noqa: E731
    decode = lambda ids: "".join([itos[i] for i in ids])  # noqa: E731

    if args.prompt:
        unknown = [c for c in args.prompt if c not in stoi]
        if unknown:
            raise ValueError(f"Characters not seen during training: {sorted(set(unknown))}")
        context = torch.tensor([encode(args.prompt)], dtype=torch.long, device=device)
    else:
        context = torch.zeros((1, 1), dtype=torch.long, device=device)

    with torch.no_grad():
        out = model.generate(
            context,
            max_new_tokens=args.tokens,
            temperature=args.temperature,
            top_k=args.top_k if args.top_k > 0 else None,
        )
    print(decode(out[0].tolist()))


if __name__ == "__main__":
    main()
