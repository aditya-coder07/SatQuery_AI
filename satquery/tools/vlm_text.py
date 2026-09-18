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


def generate(env_adapter: str, adapter_name: str, images: list, question: str,
             max_new_tokens: int = 64) -> tuple[str, float, str]:
    """Greedy reply for `question` over `images` (PIL) -> (text, mean token probability, model card)."""
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
    with torch.no_grad(), handle.using(adapter_name) as model:
        generated = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False,
                                   output_scores=True, return_dict_in_generate=True)
    reply = handle.processor.decode(generated.sequences[0][batch["input_ids"].shape[1]:],
                                    skip_special_tokens=True).strip()
    confidence = rs_vqa._mean_token_probability(torch, generated.scores)
    return reply, confidence, f"{base.name} + {adapter_name} adapter ({adapter.name})"
