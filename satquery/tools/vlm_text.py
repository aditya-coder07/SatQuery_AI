"""Shared "one prompt, one or two images, text answer" path over the VQA
tool's Qwen2.5-VL base for the caption and change-caption adapters.

The chat is built exactly as `training/train_vlm_sft.py::chat_for` builds
it for `<CAPTION>` / `<CHANGE_CAPTION>` rows (same system prompt, the same
question string the manifests carry), so the deployed caption is produced
under the prompt the adapter was trained and benchmarked with. The adapter
is attached to the shared base through `rs_vqa._ModelHandle.ensure_adapter`
and selected per call with `using()`, like the grounding adapter.
"""

from __future__ import annotations

import os
from pathlib import Path

CAPTION_QUESTION = "Describe this remote-sensing image in one sentence."
CHANGE_QUESTION = ("These two images show the same area at two different times. Describe what has "
                   "changed between them in one sentence. If nothing has changed, say so.")


def configured(env_adapter: str) -> bool:
    from satquery.tools import rs_vqa

    return bool(os.getenv(env_adapter)) and bool(os.getenv(rs_vqa.ENV_BASE))


def available(env_adapter: str, what: str) -> tuple[bool, str]:
    from satquery.tools import rs_vqa
    from satquery.tools.sidecars import readable_safetensors

    adapter = Path(os.environ[env_adapter])
    if not adapter.exists():
        return False, f"{what} adapter not found: {adapter}"
    ok, reason = readable_safetensors(adapter)
    if not ok:
        return False, reason
    if not Path(os.environ[rs_vqa.ENV_BASE]).exists():
        return False, f"base model not found: {os.environ[rs_vqa.ENV_BASE]}"
    try:
        import torch  # noqa: F401
    except ImportError:
        return False, "torch is not installed"
    return True, "ready"


def _sequence_mean_probability(torch, model, batch, new_tokens) -> float:
    """Mean probability the model assigns to each token of `new_tokens`,
    teacher-forced in one forward pass. Used for beam-searched output,
    where `generate()`'s per-step scores are beam log-scores over the
    live hypotheses rather than a distribution over the chosen token, so
    the greedy-path `_mean_token_probability` would read the wrong thing.
    Greedy output keeps the cheaper per-step path (identical result)."""
    if new_tokens.numel() == 0:
        return 0.0
    ids = torch.cat([batch["input_ids"], new_tokens.unsqueeze(0)], dim=1)
    extra = {k: v for k, v in batch.items() if k not in ("input_ids", "attention_mask")}
    attention = torch.ones_like(ids)
    with torch.no_grad():
        logits = model(input_ids=ids, attention_mask=attention, **extra).logits[0]
    start = batch["input_ids"].shape[1]
    probs = torch.softmax(logits[start - 1:start - 1 + new_tokens.shape[0]].float(), dim=-1)
    chosen = probs.gather(1, new_tokens.unsqueeze(1)).squeeze(1)
    return round(float(chosen.mean()), 6)


def generate(env_adapter: str, adapter_name: str, images: list, question: str,
             max_new_tokens: int = 64, num_beams: int = 1) -> tuple[str, float, str]:
    """Reply for `question` over `images` (PIL) -> (text, mean token probability, model card).

    `num_beams` 1 is greedy, the setting every VLM benchmark number in this
    repository was measured with and the deployed default: beam 5 was
    chosen on RSICD val and then did not hold on the official test
    (evaluation/caption_decoding.py, 2026-09-20 - BLEU-4 up, CIDEr-D and
    ROUGE-L down, captions more generic). The caption tool passes
    `SATQUERY_CAPTION_BEAMS` when set."""
    from satquery.tools import rs_vqa
    from training.train_vlm_sft import VQA_SYSTEM_PROMPT

    base = Path(os.environ[rs_vqa.ENV_BASE])
    adapter = Path(os.environ[env_adapter])
    handle = rs_vqa._ModelHandle.get(base, Path(os.environ[rs_vqa.ENV_ADAPTER]))
    handle.ensure_adapter(adapter_name, adapter)
    torch = handle.torch
    chat = [
        {"role": "system", "content": [{"type": "text", "text": VQA_SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "image", "image": im} for im in images]
                                    + [{"type": "text", "text": question}]},
    ]
    text = handle.processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    batch = handle.processor(text=[text], images=images, return_tensors="pt").to(handle.model.device)
    beams = max(1, int(num_beams))
    with torch.no_grad(), handle.using(adapter_name) as model:
        generated = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False,
                                   num_beams=beams, output_scores=(beams == 1),
                                   return_dict_in_generate=True)
        new_tokens = generated.sequences[0][batch["input_ids"].shape[1]:]
        if beams == 1:
            confidence = rs_vqa._mean_token_probability(torch, generated.scores)
        else:
            confidence = _sequence_mean_probability(torch, model, batch, new_tokens)
    reply = handle.processor.decode(new_tokens, skip_special_tokens=True).strip()
    return reply, confidence, f"{base.name} + {adapter_name} adapter ({adapter.name})"
