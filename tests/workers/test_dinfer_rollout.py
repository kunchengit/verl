import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import torch
from tensordict import TensorDict

from verl import DataProto
from verl.workers.config import RolloutConfig
from verl.workers.rollout.dinfer_rollout.dinfer_rollout import DInferRollout


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.vocab_size = 8

    def forward(self, input_ids, *args, **kwargs):
        batch, seq_len = input_ids.shape
        logits = torch.zeros(batch, seq_len, self.vocab_size, device=input_ids.device)
        logits[:, -1, 1] = 10.0 + self.bias
        return SimpleNamespace(logits=logits)

    __call__ = forward

    @property
    def device(self):
        return self.bias.device


class DummyModelFactory:
    def __new__(cls, *args, **kwargs):
        return DummyModel()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return DummyModel()


class _FakeDeviceMesh(dict):
    def __init__(self):
        super().__init__()
        self["infer_tp"] = self

    def get_local_rank(self):
        return 0


def _build_prompts(batch_size: int = 2, prompt_length: int = 4):
    input_ids = torch.tensor(
        [
            [0, 0, 5, 6],
            [0, 7, 8, 9],
        ],
        dtype=torch.long,
    )
    attention_mask = torch.tensor(
        [
            [0, 0, 1, 1],
            [0, 1, 1, 1],
        ],
        dtype=torch.long,
    )
    position_ids = torch.tensor(
        [
            [0, 1, 2, 3],
            [0, 1, 2, 3],
        ],
        dtype=torch.long,
    )
    batch = TensorDict(
        {
            "input_ids": input_ids[:batch_size],
            "attention_mask": attention_mask[:batch_size],
            "position_ids": position_ids[:batch_size],
        },
        batch_size=batch_size,
    )
    meta_info = {"pad_token_id": 0, "eos_token_id": 2}
    return DataProto(batch=batch, meta_info=meta_info)


def test_dinfer_rollout_greedy_generation():
    config = RolloutConfig(
        name="dinfer",
        mode="sync",
        dtype="float32",
        prompt_length=4,
        response_length=4,
        tensor_model_parallel_size=1,
        data_parallel_size=1,
        pipeline_model_parallel_size=1,
        engine_kwargs={"dinfer": {"sampling": {"mask_id": 0, "eos_id": 2, "enable_torch_compile": False}}},
    )
    hf_config = SimpleNamespace(pad_token_id=0, eos_token_id=2)
    model_config = SimpleNamespace(hf_config=hf_config, local_path=None)
    device_mesh = _FakeDeviceMesh()

    with patch(
        "verl.workers.rollout.dinfer_rollout.dinfer_rollout.FusedOlmoeForCausalLM",
        new=DummyModelFactory,
    ):
        rollout = DInferRollout(config=config, model_config=model_config, device_mesh=device_mesh)

    prompts = _build_prompts()
    output = rollout.generate_sequences(prompts)

    assert output.batch["responses"].shape == (2, 4)
    assert output.batch["sequences"].shape == (2, 8)
    assert torch.all(output.batch["responses"] == 1)
    assert torch.all(output.batch["attention_mask"][:, :4] == prompts.batch["attention_mask"])

    # Ensure weight updates execute without error
    weight_iter = iter([("bias", torch.ones(1))])
    asyncio.run(rollout.update_weights(weight_iter))

    output_after = rollout.generate_sequences(prompts)
    assert output_after.batch["responses"].shape == (2, 4)
