import logging
import os
from collections import OrderedDict
from typing import Generator, Optional

import torch
from tensordict import TensorDict

from dinfer.decoding.serving import SamplingParams, init_generator
from dinfer.model import FusedOlmoeForCausalLM

from verl import DataProto
from verl.utils.torch_functional import get_response_mask
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.base import BaseRollout

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _resolve_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    dtype_map = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "half": torch.float16,
    }
    if dtype is None:
        return torch.float32
    dtype_lower = str(dtype).lower()
    if dtype_lower in dtype_map:
        return dtype_map[dtype_lower]
    raise ValueError(f"Unsupported dtype for DInfer rollout: {dtype}")


def _normalize_eos(eos_token_id: int | list[int] | tuple[int, ...] | None) -> Optional[int]:
    if eos_token_id is None:
        return None
    if isinstance(eos_token_id, (list, tuple)):
        if not eos_token_id:
            return None
        return int(eos_token_id[0])
    return int(eos_token_id)


class DInferRollout(BaseRollout):
    """Rollout engine that wraps dInfer diffusion LLM."""

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        device_mesh,
    ):
        super().__init__(config, model_config, device_mesh)

        self.torch_dtype = _resolve_dtype(config.dtype)
        self.device = self._infer_device()

        self.model = self._build_model(model_config)
        self.model.eval()
        self.model.requires_grad_(False)

        dinfer_kwargs = (config.engine_kwargs or {}).get("dinfer", {}) or {}
        sampling_kwargs = dinfer_kwargs.get("sampling", {}).copy() if dinfer_kwargs.get("sampling") else {}

        hf_pad_id = getattr(model_config.hf_config, "pad_token_id", None)
        hf_eos_id = getattr(model_config.hf_config, "eos_token_id", None)
        sampling_kwargs.setdefault("mask_id", hf_pad_id if hf_pad_id is not None else 0)
        sampling_kwargs.setdefault("eos_id", hf_eos_id if hf_eos_id is not None else 2)
        sampling_kwargs.setdefault("enable_torch_compile", False)

        try:
            self.sampling_params = SamplingParams(**sampling_kwargs)
        except TypeError as exc:
            raise ValueError(f"Invalid sampling parameters for dInfer rollout: {sampling_kwargs}") from exc

        self.block_length = int(dinfer_kwargs.get("block_length", config.response_length))
        self.block_length = max(1, min(self.block_length, config.response_length))
        self.primary_eos = _normalize_eos(self.sampling_params.eos_id)
        self._diffusion_init_failed = False
        self.generator = self._build_generator()

    def _infer_device(self) -> torch.device:
        if torch.cuda.is_available():
            try:
                infer_tp = self.device_mesh["infer_tp"]
                local_rank = infer_tp.get_local_rank() if hasattr(infer_tp, "get_local_rank") else 0
            except KeyError:
                local_rank = 0
            return torch.device(f"cuda:{local_rank}")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _build_model(self, model_config: HFModelConfig) -> torch.nn.Module:
        hf_config = model_config.hf_config
        model_path = getattr(model_config, "local_path", None)
        model: torch.nn.Module
        try:
            if model_path and os.path.isdir(model_path):
                model = FusedOlmoeForCausalLM.from_pretrained(
                    model_path,
                    config=hf_config,
                    torch_dtype=self.torch_dtype,
                )
            else:
                raise FileNotFoundError
        except (FileNotFoundError, OSError, ValueError):
            logger.debug("Falling back to random-initialized FusedOlmoeForCausalLM for rollout.")
            model = FusedOlmoeForCausalLM(config=hf_config)

        target_dtype = self.torch_dtype
        if self.device.type == "cpu" and target_dtype in (torch.float16, torch.bfloat16):
            target_dtype = torch.float32
        model = model.to(self.device, dtype=target_dtype)
        return model

    def _build_generator(self):
        if self.device.type != "cuda":
            self._diffusion_init_failed = True
            logger.info(
                "dInfer rollout generator falling back to greedy decoding because device %s is not CUDA.", self.device
            )
            return None
        try:
            generator = init_generator(self.model, self.sampling_params)
            self._diffusion_init_failed = False
            return generator
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to initialize dInfer diffusion generator, using greedy fallback. %s", exc)
            self._diffusion_init_failed = True
            return None

    async def resume(self, tags: list[str]):  # noqa: D401 - interface compliance
        """No-op for dInfer rollout; weights stay resident on device."""
        _ = tags

    async def release(self):  # noqa: D401 - interface compliance
        """Release temporary memory by clearing CUDA cache."""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    async def update_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        **kwargs,
    ):
        del kwargs
        state_dict = OrderedDict()
        for name, tensor in weights:
            if tensor.device != self.device:
                tensor = tensor.to(self.device, non_blocking=True)
            if tensor.dtype != self.torch_dtype:
                tensor = tensor.to(dtype=self.torch_dtype)
            state_dict[name] = tensor

        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.debug("Missing keys when loading rollout weights: %s", missing)
        if unexpected:
            logger.debug("Unexpected keys when loading rollout weights: %s", unexpected)
        self.model.eval()
        if not self._diffusion_init_failed:
            self.generator = self._build_generator()

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        input_ids = prompts.batch["input_ids"].to(self.device)
        attention_mask = prompts.batch["attention_mask"].to(self.device)
        position_ids = prompts.batch["position_ids"].to(self.device)
        batch_size = input_ids.size(0)

        pad_token_id = prompts.meta_info.get("pad_token_id", getattr(self.model_config.hf_config, "pad_token_id", 0))
        eos_token_id = prompts.meta_info.get("eos_token_id", getattr(self.model_config.hf_config, "eos_token_id", 2))

        responses = []
        prompts_output = []
        sequences = []
        attention_masks = []
        position_ids_out = []

        for idx in range(batch_size):
            prompt_tokens = input_ids[idx].unsqueeze(0)
            prompt_mask = attention_mask[idx].unsqueeze(0)
            prompt_pos = position_ids[idx].unsqueeze(0)

            valid_prompt_len = int(prompt_mask.sum().item())
            trimmed_prompt = prompt_tokens[:, -valid_prompt_len:]
            generated = self._generate(trimmed_prompt)
            response_tokens = generated[:, trimmed_prompt.size(1) :]

            response_tokens = self._pad_or_trim(response_tokens, self.config.response_length, pad_token_id)

            response_mask = get_response_mask(
                response_id=response_tokens,
                eos_token=eos_token_id,
                dtype=prompt_mask.dtype,
            )
            attention_full = torch.cat((prompt_mask, response_mask), dim=-1)

            delta_position = torch.arange(
                1,
                response_tokens.size(1) + 1,
                device=prompt_pos.device,
                dtype=prompt_pos.dtype,
            ).unsqueeze(0)
            response_position_ids = prompt_pos[:, -1:] + delta_position
            position_full = torch.cat((prompt_pos, response_position_ids), dim=-1)

            padded_prompt = prompt_tokens
            full_sequence = torch.cat((padded_prompt, response_tokens), dim=-1)

            responses.append(response_tokens.squeeze(0))
            prompts_output.append(padded_prompt.squeeze(0))
            sequences.append(full_sequence.squeeze(0))
            attention_masks.append(attention_full.squeeze(0))
            position_ids_out.append(position_full.squeeze(0))

        batch = TensorDict(
            {
                "input_ids": torch.stack(prompts_output, dim=0).to("cpu"),
                "responses": torch.stack(responses, dim=0).to("cpu"),
                "sequences": torch.stack(sequences, dim=0).to("cpu"),
                "attention_mask": torch.stack(attention_masks, dim=0).to("cpu"),
                "position_ids": torch.stack(position_ids_out, dim=0).to("cpu"),
            },
            batch_size=batch_size,
        )
        return DataProto(batch=batch)

    def _generate(self, prompt: torch.Tensor) -> torch.Tensor:
        if self.generator is not None:
            output = self.generator.generate(
                prompt,
                gen_length=self.config.response_length,
                block_length=self.block_length,
            )
            return output.to(prompt.device)
        return self._greedy_generate(prompt)

    def _greedy_generate(self, prompt: torch.Tensor) -> torch.Tensor:
        generated = prompt
        eos_token = self.primary_eos
        for _ in range(self.config.response_length):
            with torch.no_grad():
                outputs = self.model(generated)
                logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                generated = torch.cat((generated, next_token), dim=-1)
                if eos_token is not None and bool((next_token == eos_token).all()):
                    break
        return generated

    @staticmethod
    def _pad_or_trim(tensor: torch.Tensor, target_len: int, pad_token_id: int) -> torch.Tensor:
        current_len = tensor.size(1)
        if current_len > target_len:
            return tensor[:, :target_len]
        if current_len == target_len:
            return tensor
        pad = torch.full(
            (tensor.size(0), target_len - current_len),
            pad_token_id,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        return torch.cat((tensor, pad), dim=1)
