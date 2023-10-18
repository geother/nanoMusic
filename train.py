import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.distributed import init_process_group, destroy_process_group
from torch.nn import functional as F
from model import GPTConfig, GPT, DISCRIMINATOR
from torch.nn import BCELoss, L1Loss

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = "out"
eval_interval = 2000
log_interval = 10
eval_iters = 50
eval_only = False
always_save_checkpoint = True
init_from = "scratch"
resume_model = "pop32k.pt"
use_GAN = True
# wandb logging
wandb_log = True
wandb_project = "nanoMusic"
wandb_run_name = "exp_GAN"
# data
dataset = "qml"
batch_size = 16
block_size = 512
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False
# adamw optimizer
learning_rate = 6e-4
max_iters = 200000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
# learning rate decay settings
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5

# system
device = "cuda"
dtype = (
    "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
)
compile = False
# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith("_") and isinstance(v, (int, float, bool, str))]
exec(open("configurator.py").read())  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}  # will be useful for logging
# -----------------------------------------------------------------------------


master_process = True
seed_offset = 0

tokens_per_iter = batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

data_dir = os.path.join("data", dataset)
train_data = np.load(os.path.join(data_dir, "train.npy")).astype(np.int64)
train_tail = np.load(os.path.join(data_dir, "train_tail.npy")).astype(np.int64)
val_data = np.load(os.path.join(data_dir, "val.npy")).astype(np.int64)
val_tail = np.load(os.path.join(data_dir, "val_tail.npy")).astype(np.int64)
print(f"train_data len: {train_data.shape[0]}, train_roll cnt: {train_tail.shape[0]}")
print(f"val_data   len: {val_data.shape[0]},   val_roll   cnt: {val_tail.shape[0]}")
if device_type == "cuda":
    train_data = torch.from_numpy(train_data).clone().to(device)
    val_data = torch.from_numpy(val_data).clone().to(device).detach()


def get_batch(split):
    data = train_data if split == "train" else val_data
    data_tail = train_tail if split == "train" else val_tail

    tails = torch.randint(1, data_tail.shape[0], (batch_size,))
    starts = tails - 1
    tails = data_tail[tails] - block_size
    starts = data_tail[starts]
    dur = tails - starts
    ix = [starts[i] + np.random.randint(dur[i]) for i in range(batch_size)]

    if device_type == "cuda":
        x = torch.stack([data[i : i + block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    else:
        x = torch.stack([torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64)) for i in ix])
        x, y = x.to(device), y.to(device)
    return x, y


iter_num = 0
best_val_loss = 1e9

meta_path = os.path.join(data_dir, "meta.pkl")
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    meta_vocab_size = meta["vocab_size"]
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(
    n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size, bias=bias, vocab_size=None, dropout=dropout
)  # start with model_args from command line
if init_from == "scratch":
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    model_args["vocab_size"] = 260
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == "resume":
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, resume_model)
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint["model_args"]
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint["iter_num"]
    best_val_loss = checkpoint["best_val_loss"]

# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args["block_size"] = block_size  # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == "resume":
    optimizer.load_state_dict(checkpoint["optimizer"])
checkpoint = None  # free up memory

# cGAN
if use_GAN:
    D_model = DISCRIMINATOR(gptconf).to(device)
    D_model.weight_init(mean=0.0, std=0.02)
    D_optimizer = torch.optim.AdamW(D_model.parameters(), lr=learning_rate, betas=(beta1, beta2))
    BCE_Loss = BCELoss().cuda()

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)  # requires PyTorch 2.0


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


# logging
if wandb_log and master_process:
    import wandb

    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y = get_batch("train")  # fetch the very first batch
t0 = time.time()
local_iter_num = 0  # number of iterations in the lifetime of this process
raw_model = model  # unwrap DDP container if needed
running_mfu = -1.0


while True:
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log(
                data={
                    "val/traindata_loss": losses["train"],
                    "val/valdata_loss": losses["val"],
                    "lr": lr,
                    "mfu": running_mfu * 100,  # convert to percentage
                },
                step=iter_num,
            )
        if losses["val"] < best_val_loss or always_save_checkpoint:
            now_best = losses["val"] < best_val_loss
            best_val_loss = losses["val"]
            if iter_num > 0:
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_args": model_args,
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "config": config,
                }
                print(f"saving checkpoint to {out_dir}")
                if iter_num % 10000 == 0:
                    torch.save(
                        checkpoint,
                        os.path.join(out_dir, f"new_GAN{iter_num // 1000}k{'_curBEST' if now_best else ''}.pt"),
                    )
    if iter_num == 0 and eval_only:
        break

    x_, y_ = get_batch("train")

    if use_GAN:
        D_result = D_model(x_, y_[:, -1]).squeeze()
        # print('\033[91m'+'D_result: ' + '\033[92m', D_result.data)
        D_real_loss = BCE_Loss(D_result, torch.ones(D_result.size(), device=device))

        G_result, _ = model(x_, gan_training=True)
        probs = F.softmax(G_result, dim=-1)
        G_result = torch.multinomial(probs.view(-1, 260), num_samples=1)
        G_result = G_result.view(batch_size, -1)[:, -1]
        D_result = D_model(x_, G_result).squeeze()
        # print('\033[91m'+'D_result: ' + '\033[92m', D_result.data)
        # print()
        D_fake_loss = BCE_Loss(D_result, torch.zeros(D_result.size(), device=device))

        D_train_loss = (D_real_loss + D_fake_loss) * 0.5
        D_train_loss.backward()
        D_optimizer.step()

    if use_GAN:
        # train generator G
        with ctx:
            G_result, G_loss = model(x_, y_, gan_training=True)
        probs = F.softmax(G_result, dim=-1)
        G_result = torch.multinomial(probs.view(-1, 260), num_samples=1)
        G_result = G_result.view(batch_size, -1)[:, -1]
        D_result = D_model(x_, G_result).squeeze()
        G_train_loss = BCE_Loss(
            D_result, torch.ones(D_result.size(), device=device)
        ) + 10.0 * G_loss
        scaler.scale(G_train_loss).backward()
    else:
        scaler.scale(G_loss).backward()

    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = G_loss.item()
        D_lossf = D_train_loss.item()
        if local_iter_num >= 5:  # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        if iter_num % (1 * log_interval) == 0:
            print(f"iter {iter_num}: G_loss {lossf:.4f}, D_loss {D_lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
            print(f"D_real_loss {D_real_loss.item():.4f},  D_fake_loss {D_fake_loss.item():.4f}")
        if wandb_log:
            wandb.log(
                data={
                    "train/G_loss": lossf,
                    "train/D_loss": D_lossf,
                },
                step=iter_num,
            )
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

