"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
"""

import torch
import math
import os
from contextlib import nullcontext

from model import GPT, GPTConfig
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os
from dataloader import DataLoaderLite
import time

# ----------------------------------------------------------------------

# log file
log_dir = "log"
eval_interval = 500

# data
total_batch_size = 524288
B = 64
T = 1024
grad_accum_steps = total_batch_size // (B * T)

# training parameters
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073 

# model (using GPTConfig)
vocab_size=50304

# system
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True if torch.cuda.is_available() else False

# ----------------------------------------------------------------------
# set up DDP (distributed data parallel)
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    assert torch.cuda.is_available()
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK']) # process id of gpu
    ddp_local_rank = int(os.environ['LOCAL_RANK']) # we are single node machine so local = rank
    ddp_world_size = int(os.environ['WORLD_SIZE']) # processes running
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0

    assert grad_accum_steps % ddp_world_size == 0
    grad_accum_steps //= ddp_world_size

else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"

    print(f"Using device: {device}")

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device == 'cpu' or device == 'mps' else torch.autocast(device_type=device, dtype=ptdtype)

if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

# ----------------------------------------------------------------------
# load the data
train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")

#torch.set_float32_matmul_precision("high")

# ----------------------------------------------------------------------
# create model
model = GPT(GPTConfig(vocab_size=vocab_size))
model.to(device)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model

# gcc of pytorch
if compile:
    model = torch.compile(model)

# ----------------------------------------------------------------------
# train the model

# create the optimizer
optimizer = raw_model.configure_optimizers(weight_decay=.1, learning_rate=6e-4, betas=(0.9, 0.95), device_type=device)

# open the log file
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f: # reset the file
    pass

# learning rate decay
def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    if it > max_steps:
        return min_lr
    
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)

    assert 0 <= decay_ratio <= 1

    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

laststep = False
for step in range(max_steps):
    if step == max_steps - 1:
        laststep = True

    t0 = time.time()

    if step % eval_interval == 0 or laststep:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with ctx:
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()

        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
            print(f"validation loss: {val_loss_accum.item():.4f}")

            if step > 0 and (step % (eval_interval * 10) == 0 or laststep):
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'config': raw_model.config,
                    'step': step,
                    'val_loss': val_loss_accum.item(),
                }

                torch.save(checkpoint, checkpoint_path)

    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for microstep in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)

        # mixed precision for faster training (if cuda)
        with ctx:
            logits, loss = model(x, y)

        loss = loss / grad_accum_steps # scale down loss to account for missing mean reduction before
        loss_accum += loss.detach()

        if ddp:
            # using instead of model.no_sync context manager for now
            model.require_backward_grad_sync = (microstep == grad_accum_steps - 1)
        loss.backward()

    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

    # avoid great shocks by clipping norm
    norm = torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)

    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.step()
    if ddp:
        torch.cuda.synchronize()
    t1 = time.time()
    tokens_per_sec = (train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size) / (t1 - t0)
    
    if master_process:
        print(f"step {step}, loss: {loss_accum.item():.4f} | norm: {norm:.4f} | lr: {lr:.4f} | time: {(t1 - t0) * 1000:.2f}ms | tokens per second: {tokens_per_sec:.2f}")

        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")

if ddp:
    destroy_process_group()

