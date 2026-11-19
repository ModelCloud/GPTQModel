# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Clean-room implementation from cleanroom/locateanything/SPEC.md; see NOTICE.
"""Token selection and box/point/reference pattern handling for localization decoding."""

from __future__ import annotations

from typing import Any

import torch


_TOP_LEVEL_DEFAULTS = {
    "box_start_token_id": 151668,
    "box_end_token_id": 151669,
    "coord_start_token_id": 151677,
    "coord_end_token_id": 152677,
    "ref_start_token_id": 151672,
    "ref_end_token_id": 151673,
    "none_token_id": 4064,
}


def build_token_ids(config) -> dict:
    """Collect the special-token table the localization sampler works with."""
    token_ids = {name: int(getattr(config, name, default)) for name, default in _TOP_LEVEL_DEFAULTS.items()}
    text_config = getattr(config, "text_config", None)
    eos = getattr(text_config, "eos_token_id", None)
    if isinstance(eos, (list, tuple)):
        eos = eos[0] if eos else None
    token_ids["null_token_id"] = int(getattr(text_config, "null_token_id", 152678))
    token_ids["im_end_token_id"] = int(151645 if eos is None else eos)
    token_ids["switch_token_id"] = int(getattr(text_config, "switch_token_id", 152679))
    token_ids["default_mask_token_id"] = int(getattr(text_config, "text_mask_token_id", 151676))
    return token_ids


def _is_coord(token_id: int, token_ids: dict) -> bool:
    return token_ids["coord_start_token_id"] <= int(token_id) <= token_ids["coord_end_token_id"]


# --------------------------------------------------------------------------------------------
# Logit processors
# --------------------------------------------------------------------------------------------


def apply_repetition_penalty(logits: torch.Tensor, previous_ids, penalty: float) -> torch.Tensor:
    """Penalize every vocabulary entry already produced in the same batch row."""
    if penalty == 1.0 or previous_ids is None:
        return logits
    vocab = logits.shape[-1]
    previous_ids = torch.as_tensor(previous_ids, device=logits.device).long()
    if previous_ids.dim() == 1:
        previous_ids = previous_ids.unsqueeze(0)
    out = logits.clone()
    for row in range(out.shape[0]):
        ids = previous_ids[min(row, previous_ids.shape[0] - 1)]
        ids = torch.unique(ids[(ids >= 0) & (ids < vocab)])
        if ids.numel() == 0:
            continue
        selected = out[row, ..., ids]
        out[row, ..., ids] = torch.where(selected > 0, selected / penalty, selected * penalty)
    return out


def apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Nucleus filtering; the first entry crossing the threshold is kept."""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
    drop = cumulative > top_p
    drop[..., 1:] = drop[..., :-1].clone()
    drop[..., 0] = False
    drop_unsorted = drop.scatter(-1, sorted_indices, drop)
    return logits.masked_fill(drop_unsorted, torch.finfo(logits.dtype).min)


def apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Keep the k largest logits (ties at the threshold survive)."""
    k = min(int(top_k), logits.shape[-1])
    threshold = torch.topk(logits, k, dim=-1).values[..., -1:]
    return logits.masked_fill(logits < threshold, torch.finfo(logits.dtype).min)


def _process_logits(logits, previous_ids, generate_kwargs):
    penalty = generate_kwargs.get("repetition_penalty", 1.0)
    temperature = generate_kwargs.get("temperature", 0) or 0
    top_p = generate_kwargs.get("top_p")
    top_k = generate_kwargs.get("top_k")
    if penalty is not None and penalty != 1.0:
        logits = apply_repetition_penalty(logits, previous_ids, penalty)
    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = apply_top_p(logits, top_p)
    if top_k is not None:
        logits = apply_top_k(logits, top_k)
    return logits, temperature


def _choose(probs: torch.Tensor, temperature: float):
    if temperature > 0:
        try:
            flat = probs.reshape(-1, probs.shape[-1])
            ids = torch.multinomial(flat, num_samples=1).view(probs.shape[:-1])
        except Exception:
            ids = probs.argmax(dim=-1)
    else:
        ids = probs.argmax(dim=-1)
    confidence = probs.gather(-1, ids.unsqueeze(-1)).squeeze(-1)
    return confidence, ids


def sample_tokens(logits: torch.Tensor, previous_ids=None, **generate_kwargs):
    """Process logits and choose one token per position. Returns (probs, confidence, ids)."""
    processed, temperature = _process_logits(logits, previous_ids, generate_kwargs)
    probs = torch.softmax(processed, dim=-1)
    confidence, ids = _choose(probs, temperature)
    return probs, confidence, ids


# --------------------------------------------------------------------------------------------
# Box / reference decoding
# --------------------------------------------------------------------------------------------


def classify_box_frame(probs: torch.Tensor, token_ids: dict, start_thresh: float = 0.7, end_thresh: float = 0.2) -> str:
    """Classify a 6-position probability frame as empty_box, legal_box or illegal_box."""
    null = token_ids["null_token_id"]
    if (
        probs[0, token_ids["box_start_token_id"]] >= start_thresh
        and probs[1, token_ids["none_token_id"]] > 0.2
        and probs[2, token_ids["box_end_token_id"]] > 0.2
        and probs[3, null] > 0.1
        and probs[4, null] > 0.1
    ):
        return "empty_box"
    end_mass = probs[5, token_ids["box_end_token_id"]] + probs[5, null] + probs[5, token_ids["im_end_token_id"]]
    if end_mass >= end_thresh:
        return "legal_box"
    return "illegal_box"


def decode_bbox_avg(
    logits,
    probs,
    token_ids,
    keep_k=5,
    start_thresh=0.7,
    end_thresh=0.2,
    generation_mode="hybrid",
):
    """Decode a 6-position frame into [box_start, c1, c2, c3, c4, box_end], an empty box or None."""
    del logits  # ranking uses the probabilities
    device = probs.device
    frame = classify_box_frame(probs, token_ids, start_thresh=start_thresh, end_thresh=end_thresh)
    if frame == "empty_box":
        null = token_ids["null_token_id"]
        return torch.tensor(
            [token_ids["box_start_token_id"], token_ids["none_token_id"], token_ids["box_end_token_id"], null, null, null],
            dtype=torch.long,
            device=device,
        )
    if frame == "illegal_box":
        return None

    k = min(int(keep_k), probs.shape[-1])
    coords = []
    for position in range(1, 5):
        top_probs, top_ids = torch.topk(probs[position], k)
        ranked = [(int(i), float(p)) for i, p in zip(top_ids.tolist(), top_probs.tolist()) if _is_coord(i, token_ids)]
        if not ranked:
            return None
        coord, coord_prob = ranked[0]
        if generation_mode != "fast" and coord_prob < 0.9 and len(ranked) > 1:
            ids = [i for i, _ in ranked]
            if max(ids) - min(ids) > 60:
                coord = 0
        coords.append(coord)
    return torch.tensor(
        [token_ids["box_start_token_id"], *coords, token_ids["box_end_token_id"]],
        dtype=torch.long,
        device=device,
    )


def decode_ref(logits, probs, token_ids, keep_k=5, start_thresh=0.6):
    """Decode a reference frame [ref_start, t1, ...] using non-coordinate tokens, or None."""
    del logits
    if probs[0, token_ids["ref_start_token_id"]] < start_thresh:
        return None
    k = min(int(keep_k), probs.shape[-1])
    tokens = [token_ids["ref_start_token_id"]]
    for position in range(1, probs.shape[0]):
        top_ids = torch.topk(probs[position], k).indices.tolist()
        non_coord = [int(i) for i in top_ids if not _is_coord(i, token_ids)]
        if not non_coord:
            return None
        tokens.append(non_coord[0])
    return torch.tensor(tokens, dtype=torch.long, device=probs.device)


def select_multi_position(logits: torch.Tensor, previous_ids, token_ids: dict, **generate_kwargs):
    """Choose tokens for every position and try a structured box/ref decode per batch row.

    Returns (probs, confidence, ids, box); box is None for a single position.
    """
    keep_k_avg = generate_kwargs.get("keep_k_avg", 4)
    generation_mode = generate_kwargs.get("generation_mode", "hybrid")
    processed, temperature = _process_logits(logits, previous_ids, generate_kwargs)
    probs = torch.softmax(processed, dim=-1)
    confidence, ids = _choose(probs, temperature)
    if logits.shape[1] == 1:
        return probs, confidence, ids, None

    boxes = []
    for row in range(logits.shape[0]):
        box = decode_bbox_avg(
            processed[row], probs[row], token_ids,
            keep_k=keep_k_avg, start_thresh=0.7, end_thresh=0.2, generation_mode=generation_mode,
        )
        if box is None:
            box = decode_ref(processed[row], probs[row], token_ids, keep_k=5, start_thresh=0.6)
        if box is None:
            box = torch.tensor([0], dtype=torch.long, device=logits.device)
        boxes.append(box)
    return probs, confidence, ids, torch.stack(boxes).long()


# --------------------------------------------------------------------------------------------
# Pattern handling
# --------------------------------------------------------------------------------------------


def _result(kind: str, tokens, need_switch_to_ar: bool = False, is_terminal: bool = False) -> dict[str, Any]:
    return {
        "type": kind,
        "tokens": [int(t) for t in tokens],
        "need_switch_to_ar": need_switch_to_ar,
        "is_terminal": is_terminal,
    }


def handle_pattern(candidate, token_ids, generation_mode="hybrid") -> dict:
    """Classify a 6-token candidate and return the tokens to append."""
    if torch.is_tensor(candidate):
        candidate = candidate.reshape(-1).tolist()
    c = [int(t) for t in candidate]
    null = token_ids["null_token_id"]
    im_end = token_ids["im_end_token_id"]
    box_start = token_ids["box_start_token_id"]
    box_end = token_ids["box_end_token_id"]

    if c[0] == null or c[0] == im_end:
        return _result("im_end", [im_end], is_terminal=True)
    if c[0:2] == [box_start, token_ids["none_token_id"]]:
        return _result("empty_box", [box_start, token_ids["none_token_id"], box_end])
    if c[0] == box_start:
        n = 0
        for token in c[1:5]:
            if not _is_coord(token, token_ids):
                break
            n += 1
        if n == 4 and len(c) > 5 and c[5] == box_end:
            return _result("coord_box", c[:6])
        if n == 2 and len(c) > 3 and c[3] == box_end:
            return _result("point_box", c[:4])
        if generation_mode == "fast":
            return _result("coord_box", c[:6])
        return _result("error_box", c[: n + 1], need_switch_to_ar=True)

    tokens = c[: c.index(null)] if null in c else c
    if len(tokens) >= 2 and tokens[-1] == token_ids["ref_end_token_id"] and tokens[-2] == token_ids["ref_end_token_id"]:
        tokens = tokens[:-1]
    return _result("ref_object", tokens)


__all__ = [
    "apply_repetition_penalty",
    "apply_top_k",
    "apply_top_p",
    "build_token_ids",
    "classify_box_frame",
    "decode_bbox_avg",
    "decode_ref",
    "handle_pattern",
    "sample_tokens",
    "select_multi_position",
]
