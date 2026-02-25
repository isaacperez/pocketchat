import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
import yaml
from torch.utils.data import DataLoader, random_split

from pocketchat.constants import DEFAULT_EMBEDDING_DIM, DEFAULT_MATCHER_EPS, DEFAULT_VOCAB_SIZE
from pocketchat.data import CharCorpus, NextCharSequenceDataset, build_variable_length_collate_fn
from pocketchat.losses import next_char_cross_entropy_loss
from pocketchat.model import Cortex


@dataclass
class TrainConfig:
    seed: int = 42
    device: str = "auto"
    steps: int = 3000
    log_every: int = 100
    save_every: int = 500
    val_every: int = 500
    val_split: float = 0.05
    val_max_batches: int = 0
    show_val_completion: bool = False
    val_completion_chars: int = 200
    val_completion_max_prompt_chars: int = 120
    val_completion_decoding: str = "greedy"
    val_completion_temperature: float = 1.0
    val_completion_top_k: int = 0
    learning_rate: float = 5e-4
    weight_decay: float = 1e-5
    loss_horizon_decay: float = 1.0
    warmup_steps: int = 200
    min_lr_ratio: float = 0.1
    batch_size: int = 64
    min_seq_len: int = 16
    max_seq_len: int = 128
    max_chars: int = 0
    stride: int = 1
    drop_last: bool = False
    num_workers: int = 0
    pin_memory: bool = False
    vocab_size: int = DEFAULT_VOCAB_SIZE
    embedding_dim: int = DEFAULT_EMBEDDING_DIM
    hidden_state_dim: int | None = None
    num_hypotheses: int = 4
    num_parts: int = 2
    window_size: int = 8
    num_iterations: int = 2
    num_next_chars: int = 4
    hypothesis_dim: int | None = None
    observation_dim: int | None = None
    updated_hypothesis_dim: int | None = None
    matcher_eps: float = DEFAULT_MATCHER_EPS
    grad_clip_norm: float = 1.0


def compute_batch_loss(
    outputs: dict[str, torch.Tensor],
    batch_targets: torch.Tensor,
    pad_id: int,
    lengths: torch.Tensor,
    horizon_decay: float = 1.0,
) -> torch.Tensor:
    return next_char_cross_entropy_loss(
        next_char_logits=outputs["next_char_logits"],
        targets=batch_targets,
        ignore_index=pad_id,
        prediction_position=-1,
        lengths=lengths,
        horizon_decay=horizon_decay,
    )


def _collect_special_token_ids(corpus: CharCorpus) -> torch.Tensor:
    token_ids = [corpus.char_to_id[token] for token in corpus.special_tokens if token in corpus.char_to_id]
    if not token_ids:
        return torch.empty(0, dtype=torch.long)
    return torch.tensor(token_ids, dtype=torch.long)


def _decode_next_tokens(
    logits: torch.Tensor,
    decoding: str,
    temperature: float,
    top_k: int,
) -> torch.Tensor:
    if logits.ndim != 2:
        raise ValueError(f"logits must have shape [N, V], got shape={tuple(logits.shape)}.")
    if decoding == "greedy":
        return logits.argmax(dim=-1)

    if temperature <= 0:
        raise ValueError(f"temperature must be > 0 for stochastic decoding, got {temperature}.")
    scaled_logits = logits / temperature

    if decoding == "sample":
        probs = torch.softmax(scaled_logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    if decoding == "top_k":
        if top_k <= 0:
            raise ValueError(f"top_k must be > 0 when decoding='top_k', got {top_k}.")
        k = min(top_k, scaled_logits.shape[-1])
        top_values, top_indices = torch.topk(scaled_logits, k=k, dim=-1)
        probs = torch.softmax(top_values, dim=-1)
        sampled_local = torch.multinomial(probs, num_samples=1).squeeze(-1)
        return top_indices.gather(dim=-1, index=sampled_local.unsqueeze(-1)).squeeze(-1)

    raise ValueError(
        f"Unsupported decoding='{decoding}'. Expected one of: 'greedy', 'sample', 'top_k'."
    )


@torch.no_grad()
def generate_autoregressive_completion(
    model: Cortex,
    prompt_ids: torch.Tensor,
    num_generate_chars: int,
    device: torch.device,
    special_token_ids: torch.Tensor,
    decoding: str = "greedy",
    temperature: float = 1.0,
    top_k: int = 0,
) -> torch.Tensor:
    if prompt_ids.ndim != 1:
        raise ValueError(f"prompt_ids must have shape [L], got shape={tuple(prompt_ids.shape)}.")
    if prompt_ids.numel() == 0:
        raise ValueError("prompt_ids must contain at least one token.")
    if num_generate_chars <= 0:
        raise ValueError(f"num_generate_chars must be > 0, got {num_generate_chars}.")

    was_training = model.training
    model.eval()
    try:
        current_ids = prompt_ids.to(torch.long).cpu()
        generated_chunks: list[torch.Tensor] = []
        generated_count = 0

        while generated_count < num_generate_chars:
            model_input = current_ids.unsqueeze(0).to(device)
            lengths = torch.tensor([current_ids.numel()], dtype=torch.long, device=device)
            outputs = model(model_input, lengths=lengths)
            next_char_logits = cast(torch.Tensor, outputs["next_char_logits"])[0]  # [N_pred, V]

            masked_logits = next_char_logits.clone()
            if special_token_ids.numel() > 0:
                mask_ids = special_token_ids.to(device=masked_logits.device)
                masked_logits[:, mask_ids] = float("-inf")

                all_masked_rows = torch.isneginf(masked_logits).all(dim=-1)
                if torch.any(all_masked_rows):
                    masked_logits[all_masked_rows] = next_char_logits[all_masked_rows]

            block_predictions = _decode_next_tokens(
                logits=masked_logits,
                decoding=decoding,
                temperature=temperature,
                top_k=top_k,
            ).to(torch.long).cpu()
            remaining = num_generate_chars - generated_count
            block = block_predictions[:remaining]
            generated_chunks.append(block)
            generated_count += int(block.numel())
            current_ids = torch.cat([current_ids, block], dim=0)

        return torch.cat(generated_chunks, dim=0)
    finally:
        if was_training:
            model.train()


def print_validation_completion_preview(
    step: int,
    corpus: CharCorpus,
    prompt_ids: torch.Tensor,
    generated_ids: torch.Tensor,
    max_prompt_chars: int,
) -> None:
    prompt_text = corpus.decode(prompt_ids)
    generated_text = corpus.decode(generated_ids)

    if max_prompt_chars <= 0:
        prompt_tail = prompt_text
    else:
        prompt_tail = prompt_text[-max_prompt_chars:]
    if prompt_tail == "":
        prompt_tail = "<EMPTY>"
    if generated_text == "":
        generated_text = "<EMPTY>"

    print(f"Validation Completion | step={step:6d}")
    print(f"  prompt_tail: {repr(prompt_tail)}")
    print(f"  generated:   {repr(generated_text)}")
    print(f"  completion:  {repr(prompt_tail + generated_text if generated_text != '<EMPTY>' else '')}")


def save_checkpoint(
    checkpoint_path: Path,
    model: Cortex,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    config: TrainConfig,
    corpus: CharCorpus,
    train_loss: float,
    val_loss: float | None,
    best_val_loss: float,
) -> None:
    checkpoint = {
        "step": step,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
        "config": vars(config),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "vocab": {
            "char_to_id": corpus.char_to_id,
            "id_to_char": corpus.id_to_char,
            "special_tokens": corpus.special_tokens,
            "pad_id": corpus.pad_id,
            "unk_id": corpus.unk_id,
            "bos_id": corpus.bos_id,
            "eos_id": corpus.eos_id,
        },
    }
    torch.save(checkpoint, checkpoint_path)


def load_config(path: str) -> TrainConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if "seq_len" in raw:
        raise ValueError(
            "Config key 'seq_len' is deprecated. Use 'min_seq_len' and 'max_seq_len' instead."
        )
    missing_keys = [key for key in ("min_seq_len", "max_seq_len") if key not in raw]
    if missing_keys:
        raise ValueError(
            "Config must define both 'min_seq_len' and 'max_seq_len'. "
            f"Missing: {', '.join(missing_keys)}."
        )
    return TrainConfig(**raw)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train.")
    parser.add_argument("--text", required=True, help="Path to text file used as raw char corpus.")
    parser.add_argument(
        "--config",
        default="config/default.yaml",
        help="YAML config path.",
    )
    parser.add_argument(
        "--out",
        default="runs/",
        help="Output directory for checkpoints and summaries.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to checkpoint (.pt) to resume training from.",
    )
    return parser.parse_args()


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_model(config: TrainConfig, vocab_size: int, pad_id: int | None) -> Cortex:
    return Cortex(
        vocab_size=vocab_size,
        embedding_dim=config.embedding_dim,
        padding_idx=pad_id,
        hidden_state_dim=config.hidden_state_dim,
        num_hypotheses=config.num_hypotheses,
        num_parts=config.num_parts,
        window_size=config.window_size,
        num_iterations=config.num_iterations,
        num_next_chars=config.num_next_chars,
        hypothesis_dim=config.hypothesis_dim,
        observation_dim=config.observation_dim,
        updated_hypothesis_dim=config.updated_hypothesis_dim,
        matcher_eps=config.matcher_eps,
    )


def _estimate_cortex_activation_numel(
    model: Cortex,
    batch_size: int,
    seq_len: int,
) -> int:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}.")
    if seq_len <= 0:
        raise ValueError(f"seq_len must be > 0, got {seq_len}.")

    b = batch_size
    l = seq_len
    h = model.num_hypotheses
    k = model.num_parts
    w = model.window_size
    iters = model.num_iterations
    d = model.embedding_dim
    d_hidden = model.hidden_state_dim
    d_h = model.hypothesis_dim
    d_o = model.observation_dim
    d_u = model.updated_hypothesis_dim
    n_pred = model.num_next_chars
    vocab = model.vocab_size

    token_embeddings_numel = b * l * d
    hidden_state_numel = b * d_hidden

    per_iteration_numel = (
        (b * h * d_h)  # hypotheses
        + (b * h * k)  # center logits
        + (b * h * k)  # zoom logits
        + (b * h * k * d)  # references
        + (b * h * k * d)  # adapters
        + (b * h * k * w * d)  # glimpse windows
        + (b * h * k * w)  # matcher heatmaps
        + (b * h * d_o)  # observations
        + (b * h * d_u)  # updated hypotheses
        + (b * d_hidden)  # hidden-state delta
    )

    histories_numel = (
        (b * h * iters * d_u)  # hypothesis_update_history
        + (b * d_hidden * iters)  # hidden_state_delta_history
    )
    outputs_numel = (
        (b * n_pred * vocab)  # next_char_logits
        + (b * n_pred)  # next_char_predictions (int tensor, small compared to logits)
    )

    return token_embeddings_numel + hidden_state_numel + (iters * per_iteration_numel) + histories_numel + outputs_numel


def _format_num_bytes(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    units = ["KiB", "MiB", "GiB", "TiB"]
    value = float(num_bytes)
    unit_index = -1
    while value >= 1024.0 and unit_index + 1 < len(units):
        value /= 1024.0
        unit_index += 1
    return f"{value:.2f} {units[unit_index]}"


def print_model_architecture_and_params(model: Cortex, config: TrainConfig) -> None:
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    non_trainable_params = total_params - trainable_params
    total_param_bytes = sum(param.numel() * param.element_size() for param in model.parameters())
    trainable_param_bytes = sum(
        param.numel() * param.element_size() for param in model.parameters() if param.requires_grad
    )
    non_trainable_param_bytes = total_param_bytes - trainable_param_bytes
    buffer_bytes = sum(buffer.numel() * buffer.element_size() for buffer in model.buffers())
    float_bytes = next(model.parameters()).element_size()

    avg_seq_len = int(round((config.min_seq_len + config.max_seq_len) / 2))
    activation_numel_avg = _estimate_cortex_activation_numel(
        model=model,
        batch_size=config.batch_size,
        seq_len=avg_seq_len,
    )
    activation_numel_max = _estimate_cortex_activation_numel(
        model=model,
        batch_size=config.batch_size,
        seq_len=config.max_seq_len,
    )
    activation_bytes_avg = activation_numel_avg * float_bytes
    activation_bytes_max = activation_numel_max * float_bytes

    # Memory estimate:
    # - inference_model_bytes: parameters + buffers only
    # - estimated_training_bytes_adamw: adds gradients + Adam moments (exp_avg, exp_avg_sq)
    inference_model_bytes = total_param_bytes + buffer_bytes
    grad_bytes = trainable_param_bytes
    adamw_state_bytes = 2 * trainable_param_bytes
    estimated_training_bytes_adamw = inference_model_bytes + grad_bytes + adamw_state_bytes
    estimated_training_bytes_with_activations_avg = estimated_training_bytes_adamw + activation_bytes_avg
    estimated_training_bytes_with_activations_max = estimated_training_bytes_adamw + activation_bytes_max

    print("Model architecture:")
    print(model)
    print(
        "Model parameters:",
        f"total={total_params:,}",
        f"trainable={trainable_params:,}",
        f"non_trainable={non_trainable_params:,}",
    )
    print(
        "Model memory:",
        f"params_total={_format_num_bytes(total_param_bytes)}",
        f"params_trainable={_format_num_bytes(trainable_param_bytes)}",
        f"params_non_trainable={_format_num_bytes(non_trainable_param_bytes)}",
        f"buffers={_format_num_bytes(buffer_bytes)}",
        f"inference_model={_format_num_bytes(inference_model_bytes)}",
        f"estimated_training_adamw={_format_num_bytes(estimated_training_bytes_adamw)}",
    )
    print(
        "Activation estimate:",
        f"batch_size={config.batch_size}",
        f"seq_len_avg={avg_seq_len}",
        f"seq_len_max={config.max_seq_len}",
        f"num_iterations={model.num_iterations}",
        f"forward_activations_avg={_format_num_bytes(activation_bytes_avg)}",
        f"forward_activations_max={_format_num_bytes(activation_bytes_max)}",
    )
    print(
        "Total estimate (AdamW + forward activations):",
        f"avg_seq={_format_num_bytes(estimated_training_bytes_with_activations_avg)}",
        f"max_seq={_format_num_bytes(estimated_training_bytes_with_activations_max)}",
    )
    print(
        "Memory note:",
        "activation numbers are forward-pass estimates; totals still exclude backward graph retention details, "
        "temporary kernels/workspaces, dataloader tensors, and framework overhead.",
    )


def create_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    if total_steps <= 0:
        raise ValueError(f"total_steps must be > 0, got {total_steps}.")
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}.")
    if min_lr_ratio <= 0 or min_lr_ratio > 1:
        raise ValueError(f"min_lr_ratio must be in (0, 1], got {min_lr_ratio}.")

    effective_warmup_steps = min(warmup_steps, total_steps)

    def lr_lambda(current_step: int) -> float:
        if effective_warmup_steps > 0 and current_step < effective_warmup_steps:
            return float(current_step + 1) / float(effective_warmup_steps)
        if total_steps == effective_warmup_steps:
            return min_lr_ratio

        decay_progress = float(current_step - effective_warmup_steps) / float(total_steps - effective_warmup_steps)
        decay_progress = min(1.0, max(0.0, decay_progress))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def create_dataloaders(
    corpus: CharCorpus,
    config: TrainConfig,
) -> tuple[
    DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None,
]:
    if corpus.pad_id is None:
        raise ValueError("Corpus must contain a PAD token to build training targets.")

    full_dataset = NextCharSequenceDataset(
        token_ids=corpus.data,
        seq_len=config.max_seq_len,
        num_next_chars=config.num_next_chars,
        pad_id=corpus.pad_id,
        stride=config.stride,
    )
    num_total = len(full_dataset)
    if num_total == 0:
        raise ValueError("Dataset is empty. Adjust max_seq_len/stride/max_chars.")

    val_size = int(num_total * config.val_split) if config.val_split > 0 else 0
    if config.val_split > 0 and val_size == 0 and num_total > 1:
        val_size = 1
    if val_size >= num_total:
        val_size = num_total - 1
    train_size = num_total - val_size

    if val_size > 0:
        split_generator = torch.Generator().manual_seed(config.seed)
        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=split_generator,
        )
    else:
        train_dataset = full_dataset
        val_dataset = None

    train_collate_fn = build_variable_length_collate_fn(
        min_seq_len=config.min_seq_len,
        max_seq_len=config.max_seq_len,
        pad_id=corpus.pad_id,
        random_lengths=True,
        random_offset=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=config.drop_last,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=train_collate_fn,
    )
    val_loader = None
    if val_dataset is not None:
        val_collate_fn = build_variable_length_collate_fn(
            min_seq_len=config.max_seq_len,
            max_seq_len=config.max_seq_len,
            pad_id=corpus.pad_id,
            random_lengths=False,
            random_offset=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            collate_fn=val_collate_fn,
        )
    return train_loader, val_loader


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


@torch.no_grad()
def evaluate(
    model: Cortex,
    dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    device: torch.device,
    pad_id: int,
    horizon_decay: float = 1.0,
    max_batches: int = 0,
) -> tuple[float, torch.Tensor | None]:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    preview_prompt: torch.Tensor | None = None
    seen_samples = 0

    for batch_index, (batch_inputs, batch_targets, lengths) in enumerate(dataloader, start=1):
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)
        lengths = lengths.to(device)
        outputs = model(batch_inputs, lengths=lengths)
        loss = compute_batch_loss(
            outputs=outputs,
            batch_targets=batch_targets,
            pad_id=pad_id,
            lengths=lengths,
            horizon_decay=horizon_decay,
        )
        total_loss += float(loss.item())
        total_batches += 1

        # Reservoir sampling over validation samples for an unbiased random preview.
        batch_inputs_cpu = batch_inputs.detach().cpu()
        lengths_cpu = lengths.detach().cpu()
        for sample_index in range(batch_inputs_cpu.shape[0]):
            sample_len = int(lengths_cpu[sample_index].item())
            sample_prompt = batch_inputs_cpu[sample_index, :sample_len].clone()
            seen_samples += 1
            if random.randrange(seen_samples) == 0:
                preview_prompt = sample_prompt

        if max_batches > 0 and batch_index >= max_batches:
            break

    model.train()
    if total_batches == 0:
        raise ValueError("Validation dataloader produced zero batches.")
    return total_loss / total_batches, preview_prompt


def maybe_load_checkpoint(
    resume_path: str | None,
    model: Cortex,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
) -> tuple[int, float]:
    if resume_path is None:
        return 0, float("inf")

    checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    move_optimizer_state_to_device(optimizer, device)

    scheduler_state = checkpoint.get("scheduler_state_dict")
    if scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)

    start_step = int(checkpoint.get("step", 0))
    best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
    return start_step, best_val_loss


def train(text_path: str, config: TrainConfig, out_dir: Path, resume_path: str | None = None) -> None:
    if config.steps <= 0:
        raise ValueError(f"steps must be > 0, got {config.steps}.")
    if config.log_every <= 0:
        raise ValueError(f"log_every must be > 0, got {config.log_every}.")
    if config.save_every <= 0:
        raise ValueError(f"save_every must be > 0, got {config.save_every}.")
    if config.val_every <= 0:
        raise ValueError(f"val_every must be > 0, got {config.val_every}.")
    if config.batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {config.batch_size}.")
    if config.min_seq_len < 1:
        raise ValueError(f"min_seq_len must be >= 1, got {config.min_seq_len}.")
    if config.max_seq_len < config.min_seq_len:
        raise ValueError(
            f"max_seq_len must be >= min_seq_len ({config.min_seq_len}), got {config.max_seq_len}."
        )
    if config.grad_clip_norm < 0:
        raise ValueError(f"grad_clip_norm must be >= 0, got {config.grad_clip_norm}.")
    if config.val_split < 0 or config.val_split >= 1:
        raise ValueError(f"val_split must be in [0, 1), got {config.val_split}.")
    if config.val_max_batches < 0:
        raise ValueError(f"val_max_batches must be >= 0, got {config.val_max_batches}.")
    if config.val_completion_chars <= 0:
        raise ValueError(f"val_completion_chars must be > 0, got {config.val_completion_chars}.")
    if config.val_completion_max_prompt_chars <= 0:
        raise ValueError(
            "val_completion_max_prompt_chars must be > 0, "
            f"got {config.val_completion_max_prompt_chars}."
        )
    if config.val_completion_decoding not in {"greedy", "sample", "top_k"}:
        raise ValueError(
            "val_completion_decoding must be one of: 'greedy', 'sample', 'top_k'. "
            f"Got {config.val_completion_decoding}."
        )
    if config.val_completion_temperature <= 0:
        raise ValueError(
            f"val_completion_temperature must be > 0, got {config.val_completion_temperature}."
        )
    if config.val_completion_decoding == "top_k" and config.val_completion_top_k <= 0:
        raise ValueError(
            "val_completion_top_k must be > 0 when val_completion_decoding='top_k', "
            f"got {config.val_completion_top_k}."
        )
    if config.warmup_steps < 0:
        raise ValueError(f"warmup_steps must be >= 0, got {config.warmup_steps}.")
    if config.min_lr_ratio <= 0 or config.min_lr_ratio > 1:
        raise ValueError(f"min_lr_ratio must be in (0, 1], got {config.min_lr_ratio}.")
    if config.loss_horizon_decay <= 0 or config.loss_horizon_decay > 1:
        raise ValueError(
            f"loss_horizon_decay must be in (0, 1], got {config.loss_horizon_decay}."
        )

    device = resolve_device(config.device)
    print(f"Using device: {device}")
    set_seed(config.seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    corpus = CharCorpus.from_file(
        text_path,
        max_chars=config.max_chars,
        vocab_size=config.vocab_size,
    )
    print(
        "Loaded corpus:",
        f"tokens={corpus.data.numel()}",
        f"vocab_size={corpus.vocab_size}",
        f"batch_size={config.batch_size}",
        f"min_seq_len={config.min_seq_len}",
        f"max_seq_len={config.max_seq_len}",
        f"num_next_chars={config.num_next_chars}",
        f"loss_horizon_decay={config.loss_horizon_decay}",
        f"show_val_completion={config.show_val_completion}",
        f"val_completion_decoding={config.val_completion_decoding}",
    )
    if corpus.pad_id is None:
        raise ValueError("Corpus must contain a PAD token to train with ignore_index.")

    train_loader, val_loader = create_dataloaders(corpus=corpus, config=config)
    print(f"Num training windows: {len(train_loader.dataset)}")
    if val_loader is not None:
        print(f"Num validation windows: {len(val_loader.dataset)}")
    if len(train_loader) == 0:
        raise ValueError("Dataloader is empty. Adjust min_seq_len/max_seq_len/batch_size/drop_last.")
    if config.show_val_completion and val_loader is None:
        print("Validation completion preview disabled because validation split is 0.")

    special_token_ids = _collect_special_token_ids(corpus)

    model = create_model(config=config, vocab_size=corpus.vocab_size, pad_id=corpus.pad_id).to(device)
    print_model_architecture_and_params(model, config=config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = create_lr_scheduler(
        optimizer=optimizer,
        total_steps=config.steps,
        warmup_steps=config.warmup_steps,
        min_lr_ratio=config.min_lr_ratio,
    )

    start_step, best_val_loss = maybe_load_checkpoint(
        resume_path=resume_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
    )
    if start_step >= config.steps:
        raise ValueError(
            f"Checkpoint step ({start_step}) is >= configured total steps ({config.steps}). "
            "Increase steps to continue training."
        )
    if resume_path is not None:
        print(f"Resumed from checkpoint: {resume_path}")
        print(f"Start step: {start_step}")

    model.train()
    batch_iterator = iter(train_loader)
    running_loss = 0.0
    running_steps = 0
    last_loss_value = float("nan")
    last_val_loss: float | None = None

    for step in range(start_step + 1, config.steps + 1):
        try:
            batch_inputs, batch_targets, lengths = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(train_loader)
            batch_inputs, batch_targets, lengths = next(batch_iterator)

        batch_inputs = batch_inputs.to(device, non_blocking=config.pin_memory)
        batch_targets = batch_targets.to(device, non_blocking=config.pin_memory)
        lengths = lengths.to(device, non_blocking=config.pin_memory)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch_inputs, lengths=lengths)
        loss = compute_batch_loss(
            outputs=outputs,
            batch_targets=batch_targets,
            pad_id=corpus.pad_id,
            lengths=lengths,
            horizon_decay=config.loss_horizon_decay,
        )
        loss.backward()
        if config.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
        optimizer.step()
        scheduler.step()

        last_loss_value = float(loss.item())
        running_loss += last_loss_value
        running_steps += 1

        should_log = (step % config.log_every == 0) or (step == 1) or (step == config.steps)
        if should_log:
            avg_loss = running_loss / max(running_steps, 1)
            current_lr = float(optimizer.param_groups[0]["lr"])
            print(
                f"Step {step:6d}/{config.steps} | "
                f"loss={last_loss_value:.6f} | "
                f"avg_loss={avg_loss:.6f} | "
                f"lr={current_lr:.8f}"
            )
            running_loss = 0.0
            running_steps = 0

        should_validate = val_loader is not None and ((step % config.val_every == 0) or (step == config.steps))
        if should_validate:
            last_val_loss, preview_prompt = evaluate(
                model=model,
                dataloader=val_loader,
                device=device,
                pad_id=corpus.pad_id,
                horizon_decay=config.loss_horizon_decay,
                max_batches=config.val_max_batches,
            )
            print(f"Validation | step={step:6d} | val_loss={last_val_loss:.6f}")
            if config.show_val_completion and preview_prompt is not None:
                generated_ids = generate_autoregressive_completion(
                    model=model,
                    prompt_ids=preview_prompt,
                    num_generate_chars=config.val_completion_chars,
                    device=device,
                    special_token_ids=special_token_ids,
                    decoding=config.val_completion_decoding,
                    temperature=config.val_completion_temperature,
                    top_k=config.val_completion_top_k,
                )
                print_validation_completion_preview(
                    step=step,
                    corpus=corpus,
                    prompt_ids=preview_prompt,
                    generated_ids=generated_ids,
                    max_prompt_chars=config.val_completion_max_prompt_chars,
                )
            if last_val_loss < best_val_loss:
                best_val_loss = last_val_loss
                best_path = out_dir / "checkpoint_best.pt"
                save_checkpoint(
                    checkpoint_path=best_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=step,
                    config=config,
                    corpus=corpus,
                    train_loss=last_loss_value,
                    val_loss=last_val_loss,
                    best_val_loss=best_val_loss,
                )
                print(f"Saved new best checkpoint: {best_path}")

        should_save = (step % config.save_every == 0) or (step == config.steps)
        if should_save:
            checkpoint_path = out_dir / f"checkpoint_step_{step:07d}.pt"
            save_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=step,
                config=config,
                corpus=corpus,
                train_loss=last_loss_value,
                val_loss=last_val_loss,
                best_val_loss=best_val_loss,
            )
            print(f"Saved checkpoint: {checkpoint_path}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    train(args.text, config, Path(args.out), resume_path=args.resume)


if __name__ == "__main__":
    main()
