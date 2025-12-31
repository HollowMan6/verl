# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
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

"""ModelOpt quantization helpers for Megatron models."""

from dataclasses import dataclass
from typing import Any, Callable, Optional

import modelopt.torch.quantization as mtq
from megatron.core.transformer.moe.router import TopKRouter


@dataclass
class ModelOptQuantizationConfig:
    """Configuration for applying NVIDIA ModelOpt quantization.

    This mirrors the knobs used in examples/quantization/quantize.py but is
    designed to be reusable inside verl training.
    """

    enabled: bool = False

    # One of: int8_sq, fp8, fp8_blockwise, int4_awq, w4a8_awq, nvfp4
    quant_cfg: str = "int4_awq"

    # PTQ calibration
    calib_size: int = 512

    # Optional prompt-based PTQ calibration (useful for training flows where the dataset
    # is not available at model-build time).
    # If provided and weight_only=False, we will run megatron_generate() over these prompts.
    ptq_prompts: str | None = None
    ptq_osl: int = 1

    # If True, disable activation/input quantization (weight-only path)
    weight_only: bool = True

    # If True, enable kv-cache quantization
    export_kv_cache_quant: bool = False

    # If True, force routing to all experts during calibration (MoE only)
    force_all_expert_routing: bool = False

    # If True, compress weights to low-bit (real quantization)
    compress: bool = True


def get_modelopt_torch_quantization_config(
    export_quant_cfg: str,
    export_kv_cache_quant: bool = False,
    weight_only: bool = False,
) -> dict[str, Any]:
    """Return a ModelOpt quantization config.

    Ported from examples/quantization/quantize.py.
    """

    quant_cfg_choices = {
        "int8_sq": mtq.INT8_SMOOTHQUANT_CFG,
        "fp8": mtq.FP8_DEFAULT_CFG,
        "fp8_blockwise": mtq.FP8_2D_BLOCKWISE_WEIGHT_ONLY_CFG,
        "int4_awq": mtq.INT4_AWQ_CFG,
        "w4a8_awq": mtq.W4A8_AWQ_BETA_CFG,
        "nvfp4": mtq.NVFP4_DEFAULT_CFG,
    }

    if export_quant_cfg not in quant_cfg_choices:
        raise ValueError(f"Unknown ModelOpt quant_cfg='{export_quant_cfg}'. Valid: {sorted(quant_cfg_choices.keys())}")

    mtq_config = quant_cfg_choices[export_quant_cfg]

    fp8_config = {"enable": True, "num_bits": (4, 3), "axis": None}
    fp4_config = {
        "num_bits": (2, 1),
        "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
        "axis": None,
        "enable": True,
    }

    # NOTE: quant_cfg is a nested dict that ModelOpt mutates/consumes.
    if export_quant_cfg == "fp8":
        # Enable Medusa heads and kv-cache quantization
        mtq_config["quant_cfg"]["*medusa_heads**"] = fp8_config
    if "fp4" in export_quant_cfg:
        # Enable Medusa heads and kv-cache quantization
        mtq_config["quant_cfg"]["*medusa_heads**"] = fp4_config
    if "awq" in export_quant_cfg:
        weight_quantizer = mtq_config["quant_cfg"]["*weight_quantizer"]
        if isinstance(weight_quantizer, list):
            weight_quantizer = weight_quantizer[0]
        weight_quantizer["block_sizes"][-1] = 128
    if export_kv_cache_quant:
        mtq_config["quant_cfg"]["*linear_qkv.output_quantizer"] = fp8_config
    if weight_only:
        mtq_config["quant_cfg"]["*input_quantizer"] = {"enable": False}

    return mtq_config


def apply_modelopt_quantization_with_config(
    config: ModelOptQuantizationConfig,
    ptq_forward_loop_func: Optional[Callable[[Any], Any]] = None,
) -> Callable[[list[Any]], list[Any]]:
    """Apply ModelOpt quantization to a (typically unwrapped) Megatron model.

    Args:
        models: The model instances to be quantized (*before* DDP wrapping).
        config: ModelOptQuantizationConfig.
        ptq_forward_loop_func: Optional PTQ calibration forward loop.
            Signature: (model) -> None

    Returns:
        The function that applies ModelOpt quantization.
    """

    def apply_modelopt_quantization(models: list[Any]) -> list[Any]:
        if not config.enabled:
            return models

        mtq_config = get_modelopt_torch_quantization_config(
            config.quant_cfg,
            export_kv_cache_quant=config.export_kv_cache_quant,
            weight_only=config.weight_only,
        )

        for model in models:
            if config.weight_only:
                mtq.quantize(model, mtq_config)
            else:
                if ptq_forward_loop_func is None:
                    raise ValueError(
                        "ModelOpt PTQ quantization requires `ptq_forward_loop_func` unless weight_only=True. "
                        "For verl training, either set weight_only=True or provide a PTQ forward loop callback."
                    )

                if hasattr(model, "calibration_mode"):
                    model.calibration_mode = True
                    mtq.quantize(model, mtq_config, ptq_forward_loop_func)
                    model.calibration_mode = False
                else:
                    mtq.quantize(model, mtq_config, ptq_forward_loop_func)

            if config.compress:
                mtq.compress(model)

        return models

    return apply_modelopt_quantization


def build_prompt_ptq_forward_loop(
    tokenizer: Any,
    prompts: str,
    calib_size: int,
    osl: int = 1,
    force_all_expert_routing: bool = False,
) -> Callable[[Any], None]:
    """Build a PTQ forward loop based on simple text prompts.

    This is a minimal, dependency-light alternative to dataset-based PTQ.
    """

    if calib_size <= 0:
        raise ValueError("calib_size must be > 0")
    if not prompts:
        raise ValueError("prompts must be non-empty")

    import torch
    from modelopt.torch.utils.plugins.megatron_generate import megatron_generate

    def _loop(model: Any) -> None:
        all_prompts = [p for p in prompts.split("|") if p]
        if not all_prompts:
            raise ValueError("prompts resolved to an empty list")

        # Optional MoE calibration behavior.
        if force_all_expert_routing:
            for _, module in model.named_modules():
                if isinstance(module, TopKRouter):
                    module.topk = module.num_experts

        # Cycle prompts if calib_size > len(all_prompts)
        for idx in range(calib_size):
            prompt = all_prompts[idx % len(all_prompts)]
            tokens = tokenizer(prompt, return_tensors="pt")
            input_ids = tokens.input_ids
            if not isinstance(input_ids, torch.Tensor):
                raise TypeError("tokenizer() must return a BatchEncoding with a Tensor `input_ids`")
            megatron_generate(model, input_ids.cuda(), osl=osl)

        # Restore MoE router topk if possible
        if force_all_expert_routing:
            for _, module in model.named_modules():
                if isinstance(module, TopKRouter):
                    module.topk = module.config.moe_router_topk

    return _loop
