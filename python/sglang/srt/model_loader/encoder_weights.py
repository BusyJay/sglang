"""Selective safetensors loading for the Qwen4Exp encoder (no language reads)."""

import json
import os


def is_qwen4_encoder_source(source):
    config = getattr(getattr(source, "model_config", None), "hf_config", None)
    return (
        getattr(config, "encoder_only", False)
        and "Qwen4ExpForConditionalGeneration"
        in (getattr(config, "architectures", None) or ())
    )


def is_visual_weight(name):
    return name.startswith(("model.visual.", "visual."))


def select_encoder_safetensors(hf_folder, weight_files):
    """Use the local index, or headers when unindexed; never read tensor data."""
    index_path = os.path.join(hf_folder, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path) as stream:
            weight_map = json.load(stream)["weight_map"]
        required = {shard for name, shard in weight_map.items() if is_visual_weight(name)}
        available = {os.path.relpath(path, hf_folder): path for path in weight_files}
        missing = required - available.keys()
        if missing:
            raise ValueError(f"Encoder checkpoint is missing visual shards: {sorted(missing)}")
        selected = [available[name] for name in sorted(required)]
    else:
        from safetensors import safe_open

        selected = []
        for path in weight_files:
            with safe_open(path, framework="pt", device="cpu") as reader:
                if any(is_visual_weight(name) for name in reader.keys()):
                    selected.append(path)
    if not selected:
        raise ValueError("Encoder checkpoint contains no visual safetensors weights")
    return selected


def encoder_safetensors_weights_iterator(weight_files):
    """Read only visual tensors, including when a shard also contains language weights.

    Deliberately use CPU mmap without whole-shard prefetch, eager multithread
    materialization or GPU-direct staging of the full language checkpoint.
    """
    from safetensors import safe_open

    for path in weight_files:
        with safe_open(path, framework="pt", device="cpu") as reader:
            for name in reader.keys():
                if is_visual_weight(name):
                    yield name, reader.get_tensor(name)


def validate_encoder_weights(expected_names, loaded_names):
    missing = sorted(set(expected_names) - set(loaded_names))
    if missing:
        preview = ", ".join(missing[:16])
        raise ValueError(
            f"Incomplete Qwen4Exp encoder weights: {len(missing)} visual parameters "
            f"not loaded: {preview}"
        )
