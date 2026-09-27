import json

import pytest
import torch
from safetensors.torch import save_file

from schnitz.offline_bank import assert_bank_writer_compatible
from schnitz.trajectories import file_sha256


def test_bank_evaluation_rejects_changed_writer_but_allows_frozen_reader_update(tmp_path):
    bank = tmp_path / 'bank'
    bank.mkdir()
    source = tmp_path / 'writer' / 'model.safetensors'
    source.parent.mkdir()
    save_file({'backbone.base.layer': torch.ones(2), 'backbone.bridge.weight': torch.ones(2),
               'write_slots': torch.ones(2), 'key_head.weight': torch.ones(2),
               'reader.weight': torch.ones(2)}, source)
    run = tmp_path / 'run'
    run.mkdir()
    (run / 'initialization.json').write_text(json.dumps({'checkpoint': str(source.parent)}))
    checkpoint = run / 'checkpoint'
    checkpoint.mkdir()
    current = checkpoint / 'model.safetensors'
    manifest = {'identity': {'writer_checkpoint_sha256': file_sha256(source)}}
    save_file({'backbone.base.layer': torch.ones(2), 'backbone.bridge.weight': torch.zeros(2),
               'write_slots': torch.ones(2), 'key_head.weight': torch.ones(2),
               'reader.weight': torch.zeros(2)}, current)
    assert_bank_writer_compatible(run, checkpoint, bank, manifest,
                                  training_bank_dir=str(bank))
    save_file({'backbone.base.layer': torch.ones(2), 'backbone.bridge.weight': torch.zeros(2),
               'write_slots': torch.ones(2), 'key_head.weight': torch.zeros(2),
               'reader.weight': torch.zeros(2)}, current)
    with pytest.raises(ValueError, match='writer parameters'):
        assert_bank_writer_compatible(run, checkpoint, bank, manifest,
                                      training_bank_dir=str(bank))
    with pytest.raises(ValueError, match='bank generation'):
        assert_bank_writer_compatible(run, checkpoint, bank, manifest,
                                      training_bank_dir=None)


def test_bank_evaluation_accepts_exact_frozen_writer_checkpoint(tmp_path):
    bank = tmp_path / 'bank'
    bank.mkdir()
    checkpoint = tmp_path / 'checkpoint'
    checkpoint.mkdir()
    model = checkpoint / 'model.safetensors'
    save_file({'write_slots': torch.ones(2)}, model)
    assert_bank_writer_compatible(tmp_path, checkpoint, bank,
                                  {'identity': {'writer_checkpoint_sha256': file_sha256(model)}},
                                  training_bank_dir=None)
