import os
import sys

import numpy as np
import pytest

os.environ.setdefault("SAM2_ALLOW_NO_MODEL", "1")
os.environ.setdefault("SAM2_SKIP_MODEL", "1")

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import app as app_module


class FakeTensor:
    def __init__(self, array):
        self._array = np.array(array)

    def cpu(self):
        return self

    def numpy(self):
        return self._array

    def __getitem__(self, idx):
        return FakeTensor(self._array[idx])

    def __gt__(self, other):
        return FakeTensor(self._array > other)


class FakePredictor:
    def __init__(self):
        self.calls = []

    def init_state(self, video_path):
        self.calls.append(("init_state", video_path))
        return {"video_path": video_path}

    def add_new_points_or_box(
        self, inference_state, frame_idx, obj_id, points, labels, clear_old_points
    ):
        self.calls.append(
            (
                "add_new_points_or_box",
                frame_idx,
                obj_id,
                clear_old_points,
                points.shape,
                labels.shape,
            )
        )
        logits = FakeTensor(np.ones((1, 1, 4, 5), dtype=np.float32))
        return None, [obj_id], logits

    def propagate_in_video(self, inference_state, start_frame_idx=None, reverse=False):
        self.calls.append(("propagate_in_video", start_frame_idx, reverse))
        logits = FakeTensor(np.ones((1, 1, 4, 5), dtype=np.float32))
        yield 0, [0], logits


@pytest.fixture
def fake_predictor(monkeypatch):
    fake = FakePredictor()
    monkeypatch.setattr(app_module, "predictor", fake)
    monkeypatch.setattr(app_module, "set_or_reset_timer", lambda *args, **kwargs: None)
    return fake


@pytest.fixture
def client(fake_predictor):
    app_module.inference_states.clear()
    return app_module.app.test_client()


@pytest.fixture
def app_state():
    app_module.inference_states.clear()
    return app_module
