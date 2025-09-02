#!/usr/bin/env python3
import argparse
import torch
import numpy as np

from util import CharacterTokenizer, Dataset, MultiStyleDataset
from gpt import GPTLanguageModel
from metrics import Metrics


@torch.no_grad()
def estimate_loss(data, model, style, eval_iters=100):
    device = next(model.parameters()).device
    style_tensor = torch.full((data.batch_size,), style, dtype=torch.long, device=device)

    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = data.get_batch(split, device)
            _, loss = model(X, style_tensor, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def train(multi_style_data, model, tokenizer, steps, report_frequency, lr):
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    metrics = Metrics()

    for step in range(steps):
        xb, yb, style = multi_style_data.get_batch('train', device)

        _, loss = model(xb, style, targets=yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % report_frequency == 0 or step == steps - 1:
            for style_index, dataset in enumerate(multi_style_data.datasets):
                losses_dict = estimate_loss(dataset, model, style_index)
                metrics_dict = metrics(dataset, model, style_index, tokenizer)
                report_str = ", ".join(
                    [f"{k} loss: {v:.4f}" for k, v in losses_dict.items()] +
                    [f"{k} metric: {v:.4f}" for k, v in metrics_dict.items()]
                )
                print(f"Step {step}, style {style_index}: {report_str}")
            print()


# ----------------- CLI -----------------
parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default="input.txt")
parser.add_argument("--finetune-input", type=str, default="finetune_input.txt")
parser.add_argument("--seed", type=int, default=1337)

parser.add_argument("--context-size", type=int, default=256)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--n-embd", type=int, default=384)
parser.add_argument("--n-head", type=int, default=6)
parser.add_argument("--n-layer", type=int, default=6)
parser.add_argument("--dropout", type=float, default=0.2)

subparsers = parser.add_subparsers(dest="command", required=True)

# finetune
finetune_parser = subparsers.add_parser("finetune")
finetune_parser.add_argument("--load", type=str, default="model.pth")
finetune_parser.add_argument("--save", type=str, default="model_finetuned.pth")
finetune_parser.add_argument("--steps", type=int, default=3000)
finetune_parser.add_argument("--lr", type=float, default=5e-5)
finetune_parser.add_argument("--report", type=int, default=150)

# eval
eval_parser = subparsers.add_parser("eval")
eval_parser.add_argument("--load", type=str, default="model_finetuned.pth")
eval_parser.add_argument("--prompt", type=str)
eval_parser.add_argument("--token-count", type=int, default=300)
eval_parser.add_argument("--style", type=int, default=0, help="Style index for style embedding")

args = parser.parse_args()

# ----------------- Setup -----------------
torch.manual_seed(args.seed)
batch_size = args.batch_size
context_size = args.context_size
n_embd = args.n_embd
n_head = args.n_head
n_layer = args.n_layer
dropout = args.dropout

device = 'cuda' if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
if device == "cpu":
    print("WARNING: Running on cpu!")

with open(args.input, "r", encoding="utf-8") as f:
    content = f.read()

tokenizer = CharacterTokenizer(content)
data = torch.tensor(tokenizer.encode(content), dtype=torch.long)
dataset = Dataset(data, context_size, batch_size)

with open(args.finetune_input, "r", encoding="utf-8") as f:
    finetune_content = f.read()

# ensure Earnest only uses base vocab from Canterbury tokenizer
base_vocab = set(tokenizer.vocab)
finetune_content = ''.join(ch for ch in finetune_content if ch in base_vocab)

finetune_data = torch.tensor(tokenizer.encode(finetune_content), dtype=torch.long)
finetune_dataset = Dataset(finetune_data, context_size, batch_size)

# Canterbury-heavy: 0.8 (style 0) vs Earnest 0.2 (style 1)
multi_style_dataset = MultiStyleDataset([dataset, finetune_dataset], [0.8, 0.2])

model = GPTLanguageModel(
    vocab_size=len(tokenizer.vocab),
    n_embd=n_embd,
    context_size=context_size,
    n_head=n_head,
    n_layer=n_layer,
    n_styles=len(multi_style_dataset.datasets)
).to(device)

print(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.3f}M")
print(f"Using device: {device}\n")

# ----------------- Modes -----------------
if args.command == "eval":
    print("=" * 20, "INFERENCE", "=" * 20)
    model.eval()
    # load weights (finetuned by default)
    ckpt = torch.load(args.load, map_location=device)
    model.load_state_dict(ckpt, strict=False)

    # build context and style
    if args.prompt is not None:
        context = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
    else:
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
    style_tensor = torch.tensor([args.style], dtype=torch.long, device=device)

    out = model.generate(start_idx=context, style=style_tensor, number_of_tokens=args.token_count)[0].tolist()
    print(tokenizer.decode(out))

elif args.command == "finetune":
    print("=" * 20, "MULTI-STYLE FINETUNE", "=" * 20)
    model.train()

    # allow loading a base model with n_styles=1 into n_styles=2 by duplicating row
    try:
        ckpt = torch.load(args.load, map_location=device)
        se_key = "style_embedding_table.weight"
        if se_key in ckpt and ckpt[se_key].shape[0] == 1 and model.style_embedding_table.weight.shape[0] == 2:
            w = ckpt[se_key]
            ckpt[se_key] = torch.cat([w, w.clone()], dim=0)
        model.load_state_dict(ckpt, strict=False)
        print(f"[info] Loaded base weights from {args.load}")
    except FileNotFoundError:
        print(f"[warn] Base model '{args.load}' not found; starting from scratch.")

    train(multi_style_dataset, model, tokenizer, args.steps, args.report, args.lr)
    torch.save(model.state_dict(), args.save)
    print(f"[done] Saved finetuned model to {args.save}")
