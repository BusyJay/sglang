# SPDX-License-Identifier: Apache-2.0
"""Dependency-light Qwen3.8 model constraints and checkpoint filters."""

from __future__ import annotations

import re
import json
from pathlib import Path
from typing import Iterable, Iterator

QWEN38_ARCHITECTURE = "Qwen4ExpForConditionalGeneration"
QWEN38_EXPECTED = {
    "num_hidden_layers": 48,
    "num_experts": 512,
    "num_experts_per_tok": 10,
    "hidden_size": 2560,
    "moe_intermediate_size": 640,
}


def resolve_cached_checkpoint(model_path, *, revision=None, cache_dir=None):
    """Resolve a directory or HF repo ID without downloading or modifying cache.

    Keep the caller's model ID/revision intact: config and tokenizer use the
    normal SGLang/Transformers OFFLINE path, while the loader needs shard paths.
    """
    path = Path(model_path)
    if not path.is_dir():
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError

        try:
            path = Path(snapshot_download(
                repo_id=model_path, revision=revision, cache_dir=cache_dir,
                local_files_only=True,
            ))
        except LocalEntryNotFoundError as error:
            raise ValueError(
                f"checkpoint {model_path!r} revision {revision or 'main'!r} "
                "is not in the local Hugging Face cache; mount the existing "
                "cache or provide a local directory. No download was attempted."
            ) from error
    return path


def cached_checkpoint_weight_files(checkpoint_dir):
    """Reject partial snapshots before loading; never clean or repair HF cache."""
    path = Path(checkpoint_dir)
    index = path / "model.safetensors.index.json"
    if index.is_file():
        mapping = json.loads(index.read_text())["weight_map"]
        names = sorted(set(mapping.values()))
        if not names:
            raise ValueError(f"empty safetensors shard index in {path}")
        files = []
        for name in names:
            if not isinstance(name, str) or Path(name).name != name:
                raise ValueError(f"invalid safetensors shard name: {name!r}")
            shard = path / name
            if not shard.is_file():
                raise ValueError(
                    f"incomplete local checkpoint: missing shard {shard}; "
                    "no download or cache repair was attempted"
                )
            files.append(str(shard))
        return files
    files = sorted(str(shard) for shard in path.glob("*.safetensors") if shard.is_file())
    if not files:
        raise ValueError(f"no safetensors checkpoint files found in {path}")
    return files


def validate_cached_checkpoint_revision(checkpoint_dir, hf_config, pack_revision):
    """Detect known snapshot/config/pack revision mismatches without guessing.

    Arbitrary local directories may have no verifiable revision, just as before.
    Synthetic fixture labels are not treated as HF commit hashes.
    """
    path = Path(checkpoint_dir)
    snapshot = path.name if path.parent.name == "snapshots" else None
    config_revision = getattr(hf_config, "_commit_hash", None)
    known = [value for value in (snapshot, config_revision, pack_revision)
             if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value)]
    if len(set(known)) > 1:
        raise ValueError(
            "checkpoint/config/expert-pack revision mismatch: "
            f"snapshot={snapshot}, config={config_revision}, pack={pack_revision}. "
            "Use --revision matching the pack for config, tokenizer and weights."
        )


def is_routed_expert_weight(name: str) -> bool:
    return (
        bool(re.search(r"(?:^|\.)layers\.\d+\.mlp\.experts\.", name))
        and "mtp" not in name
    )


def should_skip_checkpoint_tensor(name: str, *, language_model_only: bool) -> bool:
    return (
        "mtp" in name
        or "rotary_emb.inv_freq" in name
        or is_routed_expert_weight(name)
        or (
            language_model_only
            and (name.startswith("visual.") or "model.visual." in name)
        )
    )


def filtered_safetensors_weights_iterator(
    files: Iterable[str],
    *,
    language_model_only: bool = True,
    lazy_ple: bool = False,
    audit=None,
) -> Iterator[tuple[str, object]]:
    """Check names before get_tensor(), so skipped 63+ GiB experts are untouched."""
    from safetensors import safe_open

    for filename in files:
        if lazy_ple:
            from sglang.srt.models.qwen4_exp_ple_lazy import read_ple_checkpoint_index

            ple_index = read_ple_checkpoint_index(str(filename))
        else:
            ple_index = {}
        with safe_open(str(filename), framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if should_skip_checkpoint_tensor(
                    name, language_model_only=language_model_only
                ):
                    if audit is not None:
                        category = (
                            "mtp"
                            if "mtp" in name
                            else (
                                "rotary"
                                if "rotary_emb.inv_freq" in name
                                else (
                                    "routed_experts"
                                    if is_routed_expert_weight(name)
                                    else "visual"
                                )
                            )
                        )
                        audit[category] = audit.get(category, 0) + 1
                    continue
                if audit is not None:
                    audit["lazy_ple" if name in ple_index else "read_tensor"] = (
                        audit.get("lazy_ple" if name in ple_index else "read_tensor", 0)
                        + 1
                    )
                if name in ple_index:
                    yield name, ple_index[name]
                else:
                    yield name, handle.get_tensor(name)


def validate_qwen38_model_config(hf_config) -> list[str]:
    architectures = list(getattr(hf_config, "architectures", []) or [])
    text_config = getattr(hf_config, "text_config", hf_config)
    errors: list[str] = []
    if QWEN38_ARCHITECTURE not in architectures:
        errors.append(
            f"architectures must contain {QWEN38_ARCHITECTURE!r}, got {architectures!r}"
        )
    for field, expected in QWEN38_EXPECTED.items():
        actual = getattr(text_config, field, None)
        if actual != expected:
            errors.append(f"{field} must be {expected}, got {actual!r}")
    return errors
