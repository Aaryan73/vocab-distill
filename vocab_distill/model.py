"""Qwen3 with extra token rows, LoRA, and a switch back to the untouched model.

Methods (what is trainable):
  native       nothing; the original model
  native-lora  LoRA, native tokenizer
  naive        new rows (tied) plus a small adapter on them; the body stays frozen
  distill      new rows (untied) plus LoRA; trained with scripts/train.py
  ce-only      same model as distill, trained without the alignment losses
"""
from contextlib import contextmanager
import random

import torch
from torch import nn
from torch.nn import functional as F

from .common import MODEL_ID, MODEL_REVISION, SEED

METHODS = ['native', 'native-lora', 'naive', 'distill', 'ce-only']
LORA_TARGETS = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']


def uses_new_tokens(method):
    return method in ('naive', 'distill', 'ce-only')


class RowAdapter(nn.Module):
    """Low-rank residual on new-token embeddings (naive baseline only). Starts as identity."""

    def __init__(self, width, rank=64):
        super().__init__()
        self.down = nn.Linear(width, rank, bias=False)
        self.up = nn.Linear(rank, width, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return x + self.up(self.down(x.float())).to(x.dtype)


class NewRowEmbedding(nn.Module):
    """Frozen native embedding for old IDs, trainable FP32 rows for new IDs."""

    def __init__(self, original, new_rows, adapter=None):
        super().__init__()
        self.original = original
        self.new_rows = new_rows
        self.adapter = adapter
        self.native_only = False

    def forward(self, ids):
        size = self.original.num_embeddings
        if self.native_only:
            return self.original(ids)
        old = self.original(ids.clamp(max=size - 1))
        new = F.embedding((ids - size).clamp(min=0), self.new_rows).to(old.dtype)
        if self.adapter is not None:
            new = self.adapter(new)
        return torch.where((ids >= size).unsqueeze(-1), new, old)


class NewRowHead(nn.Module):
    """Native output head with logits for the new tokens appended."""

    def __init__(self, original, new_rows):
        super().__init__()
        self.original = original
        self.new_rows = new_rows
        self.native_only = False

    def forward(self, hidden):
        logits = self.original(hidden)
        if self.native_only:
            return logits
        return torch.cat([logits, F.linear(hidden, self.new_rows.to(hidden.dtype))], dim=-1)


def add_new_rows(model, parts, tied, adapter_rank=0):
    """Append one row per new token, initialized as the mean of its native pieces' rows."""
    embedding, head = model.get_input_embeddings(), model.get_output_embeddings()
    ids = sorted(parts)
    assert ids == list(range(embedding.num_embeddings, embedding.num_embeddings + len(ids)))
    rows_in = torch.stack([embedding.weight[parts[i]].float().mean(0) for i in ids])
    rows_out = torch.stack([head.weight[parts[i]].float().mean(0) for i in ids])
    new_in = nn.Parameter(rows_in.detach())
    new_out = new_in if tied else nn.Parameter(rows_out.detach())
    adapter = RowAdapter(embedding.embedding_dim, adapter_rank).to(new_in.device) if adapter_rank else None
    model.set_input_embeddings(NewRowEmbedding(embedding, new_in, adapter))
    model.set_output_embeddings(NewRowHead(head, new_out))


def add_lora(model, rank=16):
    from peft import LoraConfig, get_peft_model
    model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.0,
                                             target_modules=LORA_TARGETS, bias='none',
                                             task_type='CAUSAL_LM'))
    for name, p in model.named_parameters():
        p.requires_grad_('.lora_' in name or name.endswith('.new_rows'))
    return model


def build_model(method, parts=None, seed=SEED):
    from transformers import AutoModelForCausalLM
    assert method in METHODS
    random.seed(seed)
    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION, dtype=torch.bfloat16,
                                                 attn_implementation='sdpa').to('cuda')
    model.requires_grad_(False)
    if method == 'naive':
        add_new_rows(model, parts, tied=True, adapter_rank=64)
    elif method in ('distill', 'ce-only'):
        add_new_rows(model, parts, tied=False)
        model = add_lora(model)
    elif method == 'native-lora':
        model = add_lora(model)
    return model


def backbone(model):
    """The transformer stack without the output head."""
    root = model.get_base_model() if hasattr(model, 'get_base_model') else model
    return root.model


@contextmanager
def native_mode(model):
    """Temporarily turn off new rows and LoRA, giving back the original model exactly."""
    parts = [m for m in (model.get_input_embeddings(), model.get_output_embeddings()) if hasattr(m, 'native_only')]
    for m in parts:
        m.native_only = True
    try:
        if hasattr(model, 'disable_adapter'):
            with model.disable_adapter():
                yield model
        else:
            yield model
    finally:
        for m in parts:
            m.native_only = False


def trainable_state(model):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


def save_checkpoint(path, model, method, stats):
    torch.save({'method': method, 'trainable': trainable_state(model), 'stats': stats}, path)


def load_checkpoint(path, parts=None):
    """Rebuild the model a checkpoint was trained from and load its trainable weights."""
    state = torch.load(path, map_location='cpu', weights_only=False)
    method = state.get('method') or method_from_old_run(state['run_spec'])
    model = build_model(method, parts)
    # Older checkpoints called the naive adapter `interface`.
    weights = {k.replace('.interface.down.0.', '.adapter.down.').replace('.interface.up.0.', '.adapter.up.'): v
               for k, v in state['trainable'].items()}
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    assert params.keys() == weights.keys(), 'checkpoint does not match the model'
    with torch.no_grad():
        for name, p in params.items():
            p.copy_(weights[name])
    return model, method


def method_from_old_run(spec):
    """Checkpoints from the original experiments stored settings instead of a method name."""
    if spec['native']:
        return 'native-lora'
    if spec.get('interface') == 'linear':
        return 'naive'
    return 'ce-only' if spec.get('hidden_weight') == 0 and spec.get('kl_weight') == 0 else 'distill'
