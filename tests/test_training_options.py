import json
import zipfile
import pytest
from pydantic import ValidationError
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app import storage as s
from backend.app.schemas import RunConfig, EvalConfig
from backend.training.algorithms import checkpoint_algorithm


def test_stage_validation_and_algorithm_parameters():
    assert RunConfig(algorithm='a2c', world=2, stage=3, n_steps=32, batch_size=64).stage == 3
    for payload in ({'world': 0}, {'world': 9}, {'stage': 5}, {'algorithm': 'unknown'}):
        with pytest.raises(ValidationError):
            RunConfig(**payload)
    with pytest.raises(ValidationError):
        RunConfig(n_steps=32, batch_size=64)
    assert EvalConfig().world is None
    assert EvalConfig(world=1, stage=2).stage == 2


def test_checkpoint_algorithm_default_and_marker(tmp_path):
    for data, expected in [({}, 'ppo'), ({'dashboard_algorithm': 'a2c'}, 'a2c')]:
        path = tmp_path / (expected + '.zip')
        with zipfile.ZipFile(path, 'w') as z:
            z.writestr('data', json.dumps(data))
        assert checkpoint_algorithm(path) == expected


def test_actual_device_resolution(monkeypatch):
    import torch
    from backend.app.hardware import resolve_device
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    assert resolve_device('cuda') == 'cpu'
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    assert resolve_device('cuda') == 'cuda'
    assert resolve_device('auto') == 'cuda'
    assert resolve_device('cpu') == 'cpu'


def test_creation_checks_algorithm_and_preserves_selected_stage(tmp_path, monkeypatch):
    from backend.app import main
    monkeypatch.setattr(s, 'DATA', tmp_path)
    monkeypatch.setattr(main, 'ready', lambda: None)
    calls = []
    monkeypatch.setattr(main, 'launch', lambda kind, config: calls.append(config) or config)
    with TestClient(app) as client:
        mismatch = client.post('/api/runs', json={'algorithm': 'a2c', 'resume_checkpoint': 'pretrained'})
        assert mismatch.status_code == 422
        assert not calls
        response = client.post('/api/runs', json={'algorithm': 'a2c', 'world': 1, 'stage': 2})
        assert response.status_code == 201
        assert calls[-1]['algorithm'] == 'a2c'
        assert calls[-1]['stage'] == 2
        response = client.post('/api/evaluations', json={'checkpoint_id': 'pretrained', 'world': 1, 'stage': 3, 'device': 'cuda'})
        assert response.status_code == 201
        assert calls[-1]['stage'] == 3
        assert calls[-1]['device'] == 'cuda'
