import torch
import pytest


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(7)
    torch.set_num_threads(2)


@pytest.fixture
def tiny_config():
    from schnitz.config import Config, ModelConfig, MemoryConfig, TrainConfig
    return Config(
        ModelConfig(backend="tiny", tiny_width=32, tiny_layers=1, tiny_heads=4,
                    gradient_checkpointing=False),
        MemoryConfig(key_dim=12, write_slots=2, read_slots=2, payload_dims=[24],
                     neighbors=[2], reader_width=24, reader_rounds=2, chunk_size=2,
                     checkpoint_chunks=False, storage_dtype="bfloat16"),
        TrainConfig(device="cpu", precision="fp32", steps=2, train_worlds=2, eval_worlds=1,
                    gradient_accumulation=1, distractors=1, threads=2, log_every=1,
                    live_fraction=0.5, learning_rate=1e-3, backbone_learning_rate=1e-3),
    )
