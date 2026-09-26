"""QLoRA fine-tuning of the Qwen3-4B judge (master plan SS16.2, SS16.4).

- 4-bit NF4 base (double quantisation, fp16 compute: T4 has no bf16), LoRA
  r16 / alpha 32 / dropout 0.05 on all linear projections, gradient
  checkpointing, SDPA attention, one GPU.
- Prompts use the Qwen3 chat template with `enable_thinking=False`; the
  answer is read at the first assistant position. The loss is the 2-way
  cross-entropy over the "Yes" / "No" logits there (completion-only loss on
  the answer token, restricted to the two answers), which is exactly the
  quantity used at inference: p = softmax(logit_Yes, logit_No)[Yes].
- paged AdamW 8-bit, lr 1e-4, cosine with 3% warmup, effective batch 32
  (4 x 8), grad clip 0.3. Throughput is measured over the first 50 optimizer
  steps and the run is cut to fit the time budget; the adapter is saved
  every `save_every` steps.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def chat_prompt(tok, system: str, user: str) -> str:
    """Chat-template text ending right where the assistant's answer starts (thinking disabled)."""
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:  # template without the thinking switch
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def answer_ids(tok) -> tuple[int, int]:
    """Token ids of "Yes" and "No" (first sub-token of each, used consistently; plan SS16.5)."""
    y = tok.encode("Yes", add_special_tokens=False)
    n = tok.encode("No", add_special_tokens=False)
    return y[0], n[0]


def load_model(model_dir: str | Path, r: int = 16, alpha: int = 32, dropout: float = 0.05, train: bool = True,
               device: int = 0):
    """4-bit Qwen3 + LoRA (train) or plain 4-bit model (for adapter loading at inference).

    Outputs: (tokenizer, model) on cuda:<device>.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(str(model_dir))
    tok.padding_side = "left"
    tok.truncation_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.float16)
    kw = dict(quantization_config=bnb, attn_implementation="sdpa", device_map={"": device})
    try:  # transformers >= 5 renamed torch_dtype -> dtype
        model = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=torch.float16, **kw)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(str(model_dir), torch_dtype=torch.float16, **kw)
    if not train:
        return tok, model
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=TARGETS, bias="none",
                     task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    return tok, model


def encode(tok, systems, users, max_len: int) -> list[list[int]]:
    """Token ids of every prompt (left-truncated to max_len so the answer position survives)."""
    texts = [chat_prompt(tok, s, u) for s, u in zip(systems, users)]
    return tok(texts, add_special_tokens=False, truncation=True, max_length=max_len)["input_ids"]


def _batch(ids: list[list[int]], pad: int, device):
    """Left-padded tensors (input_ids, attention_mask) for a list of token-id lists."""
    import torch

    n = max(len(x) for x in ids)
    inp = torch.full((len(ids), n), pad, dtype=torch.long)
    att = torch.zeros((len(ids), n), dtype=torch.long)
    for i, x in enumerate(ids):
        inp[i, n - len(x):] = torch.tensor(x)
        att[i, n - len(x):] = 1
    return inp.to(device), att.to(device)


def yes_no_logits(model, inp, att, yes: int, no: int):
    """(batch, 2) logits of Yes / No at the last prompt position (only the last position is materialised)."""
    try:
        out = model(input_ids=inp, attention_mask=att, logits_to_keep=1)
    except TypeError:
        out = model(input_ids=inp, attention_mask=att)
    last = out.logits[:, -1, :]
    return last[:, [yes, no]].float()


def predict(model, tok, ids: list[list[int]], batch: int = 16, device: str = "cuda:0") -> np.ndarray:
    """p(Yes) for tokenised prompts (eval mode, no grad), length-sorted batches."""
    import torch

    yes, no = answer_ids(tok)
    order = np.argsort([len(x) for x in ids])
    out = np.empty(len(ids), dtype=np.float32)
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for s in range(0, len(ids), batch):
            idx = order[s : s + batch]
            inp, att = _batch([ids[i] for i in idx], tok.pad_token_id, device)
            lg = yes_no_logits(model, inp, att, yes, no)
            out[idx] = torch.softmax(lg, dim=-1)[:, 0].cpu().numpy()
    return out


def train(model, tok, ids: list[list[int]], labels: np.ndarray, out_dir: Path, lr: float = 1e-4, micro: int = 4,
          accum: int = 8, epochs: int = 1, budget_s: float = 3 * 3600, measure_steps: int = 50, save_every: int = 200,
          warmup: float = 0.03, clip: float = 0.3, seed: int = 42, log=print) -> dict:
    """1-epoch QLoRA training with a wall-clock budget; returns a training log dict.

    Inputs: model (peft), tok; ids - token ids; labels - 0/1; out_dir -
            adapter checkpoints; budget_s - training time cap (the run is cut
            to the steps that fit after measuring `measure_steps`).
    """
    import bitsandbytes as bnb
    import torch

    torch.manual_seed(seed)
    yes, no = answer_ids(tok)
    rng = np.random.default_rng(seed)
    order = np.concatenate([rng.permutation(len(ids)) for _ in range(epochs)])
    n_steps_full = len(order) // (micro * accum)
    total = n_steps_full
    params = [p for p in model.parameters() if p.requires_grad]
    opt = bnb.optim.PagedAdamW8bit(params, lr=lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")
    target = torch.tensor(labels, dtype=torch.long)
    model.train()
    t0 = time.time()
    hist, losses, saved = [], [], []
    step = 0
    for step in range(n_steps_full):
        if step >= total:
            break
        warm = max(1, int(warmup * total))
        f = (step + 1) / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
        for g in opt.param_groups:
            g["lr"] = lr * f
        for a in range(accum):
            idx = order[(step * accum + a) * micro : (step * accum + a + 1) * micro]
            inp, att = _batch([ids[i] for i in idx], tok.pad_token_id, "cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                lg = yes_no_logits(model, inp, att, yes, no)
            loss = torch.nn.functional.cross_entropy(lg, 1 - target[idx].to("cuda")) / accum  # class 0 = Yes
            scaler.scale(loss).backward()
            losses.append(float(loss) * accum)
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, clip)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        if step + 1 == measure_steps or (step + 1 < measure_steps and time.time() - t0 > budget_s / 4):
            sec = (time.time() - t0) / (step + 1)
            fit = int((budget_s - (time.time() - t0)) / sec) + step + 1
            total = min(n_steps_full, fit)
            log(f"    throughput: {sec:.2f} s/step ({micro * accum / sec:.1f} ex/s); steps {total}/{n_steps_full} fit the budget")
            hist.append({"measured_sec_per_step": round(sec, 3), "planned_steps": total, "full_steps": n_steps_full})
        if (step + 1) % 10 == 0:
            hist.append({"step": step + 1, "loss": round(float(np.mean(losses[-10 * accum:])), 5),
                         "lr": round(lr * f, 7), "elapsed_s": round(time.time() - t0, 1)})
            log(f"    step {step + 1}/{total} loss {hist[-1]['loss']} ({hist[-1]['elapsed_s']}s)")
        if (step + 1) % save_every == 0:
            path = out_dir / f"ckpt-{step + 1}"
            model.save_pretrained(str(path))
            saved.append(path)
            if len(saved) > 2:  # keep the last two checkpoints (disk)
                import shutil

                shutil.rmtree(saved.pop(0), ignore_errors=True)
    model.save_pretrained(str(out_dir / "adapter_final"))
    return {"steps_done": min(step + 1, total), "steps_full": n_steps_full, "examples_seen": min(step + 1, total) * micro * accum,
            "train_s": round(time.time() - t0, 1), "history": hist}


def main() -> None:
    """Print the LoRA target modules (smoke test; training needs a GPU)."""
    print(TARGETS)


if __name__ == "__main__":
    main()
