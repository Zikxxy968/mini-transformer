import argparse
import math
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerFast

from model import TransformerLM


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


class CustomAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad
                state = self.state[param]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param)
                    state["exp_avg_sq"] = torch.zeros_like(param)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                state["step"] += 1
                t = state["step"]

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t

                step_size = lr * (math.sqrt(bias_correction2) / bias_correction1)
                denom = exp_avg_sq.sqrt().add_(eps)

                param.addcdiv_(exp_avg, denom, value=-step_size)

                if weight_decay != 0:
                    param.add_(param, alpha=-lr * weight_decay)

        return loss


def safe_exp(x):
    return math.exp(min(x, 50.0))


def get_lr_cosine_schedule(t, alpha_max, alpha_min, warmup_steps, total_steps):
    if warmup_steps <= 0:
        warmup_steps = 1

    if t < warmup_steps:
        return (t / warmup_steps) * alpha_max

    if total_steps <= warmup_steps:
        return alpha_min

    progress = (t - warmup_steps) / (total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    cosine_out = 0.5 * (1 + math.cos(math.pi * progress))
    return alpha_min + cosine_out * (alpha_max - alpha_min)


def run_gradient_clipping(params, max_norm, eps=1e-6):
    params_with_grad = [p for p in params if p.grad is not None]
    if not params_with_grad:
        return

    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), 2) for p in params_with_grad]),
        2,
    )
    clip_coeff = max_norm / (total_norm + eps)

    if clip_coeff < 1.0:
        for p in params_with_grad:
            p.grad.detach().mul_(clip_coeff)


class CausalMemmapDataset(Dataset):
    def __init__(self, data_path, context_length, start_block=0, end_block=None):
        self.data = np.memmap(data_path, mode="r", dtype=np.int32)
        self.context_length = context_length

        total_blocks = (len(self.data) - context_length - 1) // context_length

        if end_block is None:
            end_block = total_blocks

        self.start_block = start_block
        self.end_block = end_block
        self.num_blocks = end_block - start_block

        if self.num_blocks <= 0:
            print(
                f"Warning: Dataset has 0 blocks. "
                f"start_block={start_block}, end_block={end_block}",
                flush=True,
            )

    def __len__(self):
        return max(0, self.num_blocks)

    def __getitem__(self, idx):
        block_idx = self.start_block + idx
        start_idx = block_idx * self.context_length

        x = torch.from_numpy(
            self.data[start_idx:start_idx + self.context_length].astype(np.int64)
        )
        y = torch.from_numpy(
            self.data[start_idx + 1:start_idx + self.context_length + 1].astype(np.int64)
        )
        return x, y


def save_ppl_curve(train_ppls, val_ppls, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    if train_ppls:
        plt.figure()
        plt.plot(train_ppls)
        plt.yscale("log")
        plt.xlabel("Training Step")
        plt.ylabel("Perplexity")
        plt.title("Training Perplexity")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "train_ppl.png"))
        plt.close()

    if val_ppls:
        plt.figure()
        plt.plot(val_ppls)
        plt.yscale("log")
        plt.xlabel("Validation Step")
        plt.ylabel("Perplexity")
        plt.title("Validation Perplexity")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "val_ppl.png"))
        plt.close()


def save_checkpoint(path, model, optimizer, iteration, epoch, config):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
        "epoch": epoch,
        "config": config,
    }

    tmp_path = f"{path}.tmp"
    torch.save(ckpt, tmp_path)
    os.replace(tmp_path, path)
    print(f"[Checkpoint] Saved: {path}", flush=True)


def get_memory_usage(device):
    if device == "cuda":
        mem = torch.cuda.memory_allocated() / 1024 ** 2
        return f"{mem:.2f} MB (GPU)"
    if device == "mps":
        mem = torch.mps.current_allocated_memory() / 1024 ** 2
        return f"{mem:.2f} MB (MPS)"

    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 ** 2
    return f"{mem:.2f} MB (CPU)"


def build_checkpoint_config(args, vocab_size):
    return {
        "vocab_size": vocab_size,
        "context_length": args.context_length,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "d_model": args.d_model,
    }


@torch.no_grad()
def run_validation(model, val_loader, criterion, vocab_size, device, max_val_batches):
    model.eval()

    val_losses = []
    val_ppls = []

    for batch_idx, (x, y) in enumerate(val_loader):
        if max_val_batches > 0 and batch_idx >= max_val_batches:
            break

        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))

        loss_val = loss.item()
        val_losses.append(loss_val)
        val_ppls.append(safe_exp(loss_val))

    if val_losses:
        avg_loss = sum(val_losses) / len(val_losses)
        avg_ppl = safe_exp(avg_loss)
    else:
        avg_loss = 0.0
        avg_ppl = 0.0

    return avg_loss, avg_ppl, val_ppls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--context_length", type=int, default=128)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--vocab_size", type=int, default=50257)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=3e-5)
    parser.add_argument("--checkpoint_dir", type=str, default="./ckpt")
    parser.add_argument("--data_path", type=str, default="data.bin")
    parser.add_argument("--tokenizer_path", type=str, default="bpe_tokenizer/tokenizer.json")
    parser.add_argument("--save_every_steps", type=int, default=5000)
    parser.add_argument("--max_val_batches", type=int, default=1000)
    parser.add_argument("--skip_validation", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Using Device: {device}", flush=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=args.tokenizer_path)
    vocab_size = tokenizer.vocab_size

    if not os.path.exists(args.data_path):
        print(f"Creating dummy data at {args.data_path}...", flush=True)
        dummy_len = args.context_length * args.batch_size * 20
        dummy_data = np.random.randint(0, vocab_size, (dummy_len,), dtype=np.int32)
        dummy_data.tofile(args.data_path)

    total_ds = CausalMemmapDataset(args.data_path, args.context_length)
    total_blocks = len(total_ds)
    split_block = int(total_blocks * 0.8)

    train_ds = CausalMemmapDataset(
        args.data_path,
        args.context_length,
        start_block=0,
        end_block=split_block,
    )
    val_ds = CausalMemmapDataset(
        args.data_path,
        args.context_length,
        start_block=split_block,
        end_block=total_blocks,
    )

    if len(train_ds) == 0:
        raise ValueError("Training dataset is empty. Increase data size or reduce context_length.")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )

    train_ppls = []
    val_ppls = []

    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        max_seq_len=args.context_length,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count / 1e6:.2f}M", flush=True)

    optimizer = CustomAdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    criterion = nn.CrossEntropyLoss()

    total_steps = len(train_loader) * args.epochs
    warmup_steps = max(1, int(0.1 * total_steps))
    step_t = 0
    current_epoch = 0

    config = build_checkpoint_config(args, vocab_size)

    try:
        for epoch in range(1, args.epochs + 1):
            current_epoch = epoch
            epoch_start_time = time.time()
            model.train()

            current_epoch_losses = []

            print(f"\n--- Epoch {epoch} ---", flush=True)
            print(f"Initial memory: {get_memory_usage(device)}", flush=True)

            step_interval_start = time.time()

            for batch_idx, (x, y) in enumerate(train_loader):
                x = x.to(device)
                y = y.to(device)

                lr = get_lr_cosine_schedule(
                    step_t,
                    args.lr,
                    args.min_lr,
                    warmup_steps,
                    total_steps,
                )

                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

                optimizer.zero_grad(set_to_none=True)

                logits = model(x)
                loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))

                loss_val = loss.item()
                ppl_val = safe_exp(loss_val)

                loss.backward()
                run_gradient_clipping(model.parameters(), max_norm=1.0)
                optimizer.step()

                current_epoch_losses.append(loss_val)
                train_ppls.append(ppl_val)
                step_t += 1

                if args.save_every_steps > 0 and step_t % args.save_every_steps == 0:
                    save_checkpoint(
                        path=os.path.join(args.checkpoint_dir, "latest.pt"),
                        model=model,
                        optimizer=optimizer,
                        iteration=step_t,
                        epoch=epoch,
                        config=config,
                    )

                if batch_idx % 100 == 0:
                    interval_time = time.time() - step_interval_start
                    print(f"Batch {batch_idx} | Memory: {get_memory_usage(device)}", flush=True)
                    print(
                        f"Epoch {epoch} | Step {step_t}/{total_steps} | "
                        f"LR: {lr:.6f} | Train Loss: {loss_val:.4f} | "
                        f"Train PPL: {ppl_val:.4f} | Interval Time: {interval_time:.2f}s",
                        flush=True,
                    )
                    step_interval_start = time.time()

            if current_epoch_losses:
                epoch_train_loss = sum(current_epoch_losses) / len(current_epoch_losses)
                epoch_train_ppl = safe_exp(epoch_train_loss)
            else:
                epoch_train_loss = 0.0
                epoch_train_ppl = 0.0

            train_duration = time.time() - epoch_start_time

            print(
                f"[Epoch {epoch} Train Done] "
                f"Time: {train_duration:.2f}s | "
                f"Avg Loss: {epoch_train_loss:.4f} | "
                f"Train PPL: {epoch_train_ppl:.2f} | "
                f"Memory: {get_memory_usage(device)}",
                flush=True,
            )

            save_checkpoint(
                path=os.path.join(args.checkpoint_dir, f"epoch_{epoch}.pt"),
                model=model,
                optimizer=optimizer,
                iteration=step_t,
                epoch=epoch,
                config=config,
            )

            save_ppl_curve(train_ppls, val_ppls, args.checkpoint_dir)

            if args.skip_validation:
                print(f"[Epoch {epoch}] Validation skipped.", flush=True)
                continue

            val_start_time = time.time()
            val_loss, val_ppl, epoch_val_ppls = run_validation(
                model=model,
                val_loader=val_loader,
                criterion=criterion,
                vocab_size=vocab_size,
                device=device,
                max_val_batches=args.max_val_batches,
            )
            val_ppls.extend(epoch_val_ppls)

            val_duration = time.time() - val_start_time

            print(
                f"[Epoch {epoch} Val Done] "
                f"Time: {val_duration:.2f}s | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val PPL: {val_ppl:.2f} | "
                f"Memory: {get_memory_usage(device)}",
                flush=True,
            )

            save_ppl_curve(train_ppls, val_ppls, args.checkpoint_dir)

    except KeyboardInterrupt:
        print("\n[Interrupted] Saving interrupted checkpoint...", flush=True)
        save_checkpoint(
            path=os.path.join(args.checkpoint_dir, "interrupted.pt"),
            model=model,
            optimizer=optimizer,
            iteration=step_t,
            epoch=current_epoch,
            config=config,
        )
        raise


if __name__ == "__main__":
    main()