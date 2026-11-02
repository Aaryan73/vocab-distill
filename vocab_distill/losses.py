"""Training losses. Logits are computed in small chunks so the full [length, vocab] tensor never exists."""
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def next_token_loss(head, hidden, ids, chunk=16):
    """Mean cross-entropy of predicting ids[t + 1] from hidden[t], for one sequence."""
    states, targets = hidden[0, :-1], ids[0, 1:]

    def chunk_loss(s, t):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = head(s)
        return F.cross_entropy(logits.float(), t, reduction='sum')

    total = states.new_zeros((), dtype=torch.float32)
    for i in range(0, len(states), chunk):
        s, t = states[i:i + chunk], targets[i:i + chunk]
        # Recompute logits in the backward pass instead of keeping them.
        total = total + (checkpoint(chunk_loss, s, t, use_reentrant=False) if s.requires_grad else chunk_loss(s, t))
    return total / len(targets)


def state_gap(student, teacher):
    """Mean squared distance between L2-normalized hidden states."""
    return (F.normalize(student.float(), dim=-1) - F.normalize(teacher.float(), dim=-1)).square().sum(-1).mean()


def alignment_losses(student, teacher, ends, layers, native_head, kl_positions=128):
    """Match the student to the native teacher at every shared token boundary.

    student, teacher: model outputs with hidden_states. ends[j] is the teacher position
    where student token j ends, so student[j] and teacher[ends[j]] have read the same text.
    """
    index = torch.tensor(ends, device=student.last_hidden_state.device)
    s_final = student.last_hidden_state[0]
    t_final = teacher.last_hidden_state[0, index]

    hidden = sum(state_gap(student.hidden_states[l][0], teacher.hidden_states[l][0, index]) for l in layers) / len(layers)
    hidden = hidden + state_gap(s_final, t_final)

    # KL(teacher || student) on the native vocabulary, at a random subset of boundaries.
    pick = torch.randperm(len(ends), device=index.device)[:kl_positions]
    kl = s_final.new_zeros((), dtype=torch.float32)
    for i in range(0, len(pick), 16):
        idx = pick[i:i + 16]
        log_s = F.log_softmax(native_head(s_final[idx]).float(), -1)
        log_t = F.log_softmax(native_head(t_final[idx]).float(), -1)
        kl = kl + F.kl_div(log_s, log_t, reduction='sum', log_target=True)
    return hidden, kl / len(pick)
