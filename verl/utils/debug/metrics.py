# Copyright 2025 Individual Contributor: TomQunChaoA
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import torch

from verl.protocol import DataProto

logger = logging.getLogger(__file__)


def _is_primary_process() -> bool:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0


def _maybe_emit_logprob_diff_report(
    rollout_log_probs: torch.Tensor,
    actor_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    responses: torch.Tensor | None,
) -> None:
    if not _is_primary_process():
        return

    threshold = 10.0
    topk = 3

    if rollout_log_probs.dim() == 1:
        rollout_log_probs = rollout_log_probs.unsqueeze(0)
        actor_log_probs = actor_log_probs.unsqueeze(0)
        response_mask = response_mask.unsqueeze(0)
        if responses is not None and responses.dim() == 1:
            responses = responses.unsqueeze(0)

    response_mask_bool = response_mask.bool()
    logprob_abs_diff = (actor_log_probs - rollout_log_probs).abs()
    valid_logprob_abs_diff = torch.masked_select(logprob_abs_diff, response_mask_bool)
    if valid_logprob_abs_diff.numel() == 0 or valid_logprob_abs_diff.max().item() < threshold:
        return

    mask_float = response_mask_bool.float()
    valid_counts = mask_float.sum(dim=-1).clamp_min(1.0)
    rollout_mean = (rollout_log_probs * mask_float).sum(dim=-1) / valid_counts
    actor_mean = (actor_log_probs * mask_float).sum(dim=-1) / valid_counts

    per_datum_max = logprob_abs_diff.masked_fill(~response_mask_bool, float("-inf")).max(dim=-1).values
    print("[ROLLOUT_LOGPROB_DEBUG] response logprob mismatch summary:", flush=True)
    for batch_idx in range(response_mask_bool.shape[0]):
        if not response_mask_bool[batch_idx].any():
            continue
        print(
            f"  datum {batch_idx}: response-target mean rollout {rollout_mean[batch_idx].item():.3f}, "
            f"actor {actor_mean[batch_idx].item():.3f}, max abs diff {per_datum_max[batch_idx].item():.3f}",
            flush=True,
        )

    masked_flat_diff = logprob_abs_diff.masked_fill(~response_mask_bool, float("-inf")).reshape(-1)
    topk = min(topk, int(response_mask_bool.sum().item()))
    top_values, top_indices = torch.topk(masked_flat_diff, k=topk)
    seq_len = response_mask_bool.shape[1]
    print("[ROLLOUT_LOGPROB_DEBUG] top response token examples:", flush=True)
    for rank, (value, flat_idx) in enumerate(zip(top_values.tolist(), top_indices.tolist(), strict=False), start=1):
        batch_idx = flat_idx // seq_len
        seq_idx = flat_idx % seq_len
        token_text = ""
        if responses is not None and responses.shape == rollout_log_probs.shape:
            token_text = f", token {int(responses[batch_idx, seq_idx].item())}"
        print(
            f"  {rank}. datum {batch_idx}, pos {seq_idx}{token_text}: "
            f"rollout {rollout_log_probs[batch_idx, seq_idx].item():.3f}, "
            f"actor {actor_log_probs[batch_idx, seq_idx].item():.3f}, abs diff {value:.3f}",
            flush=True,
        )


def calculate_token_list_diff(tensor1: torch.Tensor, tensor2: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # verify inputs
    if tensor1.numel() == 0 or tensor2.numel() == 0:
        return torch.zeros(tensor1.shape[0], dtype=torch.long, device=tensor1.device)
    if tensor1.shape != tensor2.shape or mask.shape != tensor1.shape or mask.shape != tensor2.shape:
        print(
            f"<WARN> dim of tensor1, tensor2, mask is not equal, {(tensor1.shape)=},{(tensor2.shape)=}, {(mask.shape)=}"
        )
        return torch.ones_like(tensor1)
    # transfer to same device
    if tensor2.device != tensor1.device:
        tensor2 = tensor2.to(tensor1.device)
    if mask.device != tensor1.device:
        mask = mask.to(tensor1.device)

    # calculate diff
    diff_mask = tensor1 != tensor2

    valid_diff_mask = diff_mask & (mask == 1)

    diff_counts = valid_diff_mask.sum(dim=1)

    return diff_counts


def pearson_correlation_coefficient(tensor1: torch.Tensor, tensor2: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # implemention of https://arxiv.org/pdf/2506.13585
    if tensor1.shape != tensor2.shape or mask.shape != tensor1.shape or mask.shape != tensor2.shape:
        return 0
    mt1 = torch.masked_select(tensor1, mask)
    mt2 = torch.masked_select(tensor2, mask)
    result = torch.corrcoef(torch.stack([mt1, mt2], dim=0))
    return result[0][1].detach().item()


def calculate_log_prob_diff(log_probs1: torch.Tensor, log_probs2: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    full_diff = torch.abs(log_probs1 - log_probs2)
    return torch.masked_select(full_diff, mask)


def calculate_debug_metrics(data: DataProto) -> dict:
    """
    calculate rollout vs actor logprobs diff, for debugging purpose

    Args:
        data: DataProto
            the data batch to calculate
            rollout_log_probs: log_probs record when rollout forward tokens
            old_log_probs(actor log probs): log_probs record when actor forward tokens
            loss_mask or attention_mask: to mask unrelated token
            responses: the response tokens, for calculating size
    Returns:
        dict: metrics
            "training/rollout_probs_diff_valid": 1->input is valid, 0->input is invalid
            "training/rollout_probs_diff_max": max value of probability diff of rollout vs. actor
            "training/rollout_probs_diff_mean": mean value of probability diff of rollout vs. actor
            "training/rollout_probs_diff_std": std value of probability diff of rollout vs. actor
            "training/rollout_logprobs_diff_valid": 1->input is valid, 0->input is invalid
            "training/rollout_logprobs_diff_max": max value of logprob diff of rollout vs. actor
            "training/rollout_logprobs_diff_mean": mean value of logprob diff of rollout vs. actor
            "training/rollout_logprobs_diff_std": std value of logprob diff of rollout vs. actor
            "training/rollout_actor_probs_pearson_corr": logprob's pearson corrcoef of rollout vs. actor, reference to https://arxiv.org/pdf/2506.13585
    """

    rollout_old_log_probs = data.batch["rollout_log_probs"]
    actor_old_log_probs = data.batch["old_log_probs"]
    if "response_mask" in data.batch:
        logger.debug("response mask found, use it to mask log probs")
        log_prob_mask = data.batch["response_mask"]
    elif "attention_mask" in data.batch:
        log_prob_mask = data.batch["attention_mask"]
    else:
        logger.warning(f"no mask info found, use all log probs, {(data.batch.keys())=}")
        log_prob_mask = torch.ones_like(rollout_old_log_probs)
    responses = data.batch["responses"]
    response_length = responses.size(1)

    response_mask = log_prob_mask[:, -response_length:]
    # calculate pearson corrcoef
    actor_probs = torch.exp(actor_old_log_probs)
    rollout_probs = torch.exp(rollout_old_log_probs)
    response_mask_bool = response_mask.bool()
    pearson_corrcoef = pearson_correlation_coefficient(actor_probs, rollout_probs, response_mask_bool)
    rollout_probs_diff = calculate_log_prob_diff(actor_probs, rollout_probs, response_mask_bool)
    rollout_logprobs_diff = calculate_log_prob_diff(actor_old_log_probs, rollout_old_log_probs, response_mask_bool)
    rollout_probs_std = torch.std(rollout_probs_diff).detach().item() if rollout_probs_diff.numel() > 1 else 0.0
    rollout_logprobs_std = (
        torch.std(rollout_logprobs_diff).detach().item() if rollout_logprobs_diff.numel() > 1 else 0.0
    )

    _maybe_emit_logprob_diff_report(
        rollout_log_probs=rollout_old_log_probs,
        actor_log_probs=actor_old_log_probs,
        response_mask=response_mask,
        responses=responses,
    )

    return {
        "training/rollout_probs_diff_valid": 1,
        "training/rollout_probs_diff_max": torch.max(rollout_probs_diff).detach().item(),
        "training/rollout_probs_diff_mean": torch.mean(rollout_probs_diff).detach().item(),
        "training/rollout_probs_diff_std": rollout_probs_std,
        "training/rollout_logprobs_diff_max": torch.max(rollout_logprobs_diff).detach().item(),
        "training/rollout_logprobs_diff_mean": torch.mean(rollout_logprobs_diff).detach().item(),
        "training/rollout_logprobs_diff_std": rollout_logprobs_std,
        "training/rollout_actor_probs_pearson_corr": pearson_corrcoef,
    }
