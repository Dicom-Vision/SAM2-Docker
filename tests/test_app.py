import io
import json
import os
import zipfile

import numpy as np
import pytest


def seed_session(
    app_state,
    session_id,
    temp_dir,
    jpg_dir=None,
    points_history=None,
    n_frames=None,
    frame_shape=None,
):
    app_state.inference_states[session_id] = {
        "inference_state": {},
        "temp_dir": temp_dir,
        "points_history": points_history or {},
    }
    if jpg_dir is not None:
        app_state.inference_states[session_id]["jpg_dir"] = jpg_dir
    if n_frames is not None:
        app_state.inference_states[session_id]["n_frames"] = n_frames
    if frame_shape is not None:
        app_state.inference_states[session_id]["frame_shape"] = frame_shape


class DummyForm:
    def __init__(self, data=None):
        self._data = data

    def to_dict(self):
        return self._data


class DummyRequest:
    def __init__(self, json_data=None, form_data=None):
        self._json_data = json_data
        self.form = DummyForm(form_data)

    def get_json(self, silent=True):
        return self._json_data


class FakeScheduler:
    def __init__(self):
        self.jobs = {}
        self.removed = []
        self.added = []

    def start(self):
        return None

    def get_job(self, job_id):
        return self.jobs.get(job_id)

    def remove_job(self, job_id):
        self.removed.append(job_id)
        self.jobs.pop(job_id, None)

    def add_job(self, func, trigger, id, args, replace_existing):
        self.added.append((func, trigger, id, args, replace_existing))
        self.jobs[id] = True


class SimpleTensor:
    def __init__(self, array):
        self._array = np.array(array)

    def cpu(self):
        return self

    def numpy(self):
        return self._array

    def __getitem__(self, idx):
        return SimpleTensor(self._array[idx])

    def __gt__(self, other):
        return SimpleTensor(self._array > other)


def build_dicom_zip():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zipf:
        for name in ["a.dcm", "b.dcm", "c.dcm", "d.dcm"]:
            zipf.writestr(name, b"dicom")
    buffer.seek(0)
    return buffer


def test_add_points_records_history(client, app_state, tmp_path):
    session_id = "session-add"
    seed_session(app_state, session_id, str(tmp_path))
    payload = {
        "session_id": session_id,
        "frame_idx": "0",
        "obj_id": "1",
        "points": json.dumps([[[1, 2], [3, 4]]]),
        "labels": json.dumps([[1, 0]]),
    }

    response = client.post("/add_points", data=payload)

    assert response.status_code == 200
    history = app_state.inference_states[session_id]["points_history"][(0, 1)]
    assert history["points"] == [[1, 2], [3, 4]]
    assert history["labels"] == [1, 0]


def test_add_points_history_appends(client, app_state, tmp_path):
    session_id = "session-append"
    seed_session(app_state, session_id, str(tmp_path))
    payload = {
        "session_id": session_id,
        "frame_idx": "0",
        "obj_id": "1",
        "points": json.dumps([[[1, 2]]]),
        "labels": json.dumps([[1]]),
    }

    response = client.post("/add_points", data=payload)
    assert response.status_code == 200

    payload["points"] = json.dumps([[[3, 4]]])
    payload["labels"] = json.dumps([[0]])
    response = client.post("/add_points", data=payload)
    assert response.status_code == 200

    history = app_state.inference_states[session_id]["points_history"][(0, 1)]
    assert history["points"] == [[1, 2], [3, 4]]
    assert history["labels"] == [1, 0]


def test_add_points_invalid_session(client):
    payload = {
        "session_id": "missing",
        "frame_idx": "0",
        "obj_id": "1",
        "points": json.dumps([[[1, 2]]]),
        "labels": json.dumps([[1]]),
    }

    response = client.post("/add_points", data=payload)

    assert response.status_code == 400
    assert response.get_json()["error"] == "Invalid session_id"


def test_add_points_invalid_format(client, app_state, tmp_path):
    session_id = "session-format"
    seed_session(app_state, session_id, str(tmp_path))
    payload = {
        "session_id": session_id,
        "frame_idx": "0",
        "obj_id": "1",
        "points": json.dumps("bad"),
        "labels": json.dumps([1]),
    }

    response = client.post("/add_points", data=payload)

    assert response.status_code == 400
    assert response.get_json()["error"] == "Invalid points or labels format"


def test_add_points_length_mismatch(client, app_state, tmp_path):
    session_id = "session-mismatch"
    seed_session(app_state, session_id, str(tmp_path))
    payload = {
        "session_id": session_id,
        "frame_idx": "0",
        "obj_id": "1",
        "points": json.dumps([[[1, 2]]]),
        "labels": json.dumps([[1, 0]]),
    }

    response = client.post("/add_points", data=payload)

    assert response.status_code == 400
    assert response.get_json()["error"] == "Points and labels length mismatch"


def test_add_points_missing_fields_returns(client, app_state, tmp_path):
    session_id = "session-missing"
    seed_session(app_state, session_id, str(tmp_path))
    data = {
        "session_id": session_id,
        "points": json.dumps([[[1, 2]]]),
        "labels": json.dumps([[1]]),
    }

    response = client.post("/add_points", data=data)

    assert response.status_code == 400
    assert response.get_json()["error"] == "All fields are required"


def test_add_points_no_data(app_state, monkeypatch):
    monkeypatch.setattr(app_state, "request", DummyRequest(form_data=None))
    with app_state.app.app_context():
        response, status = app_state.add_points()

    assert status == 400
    assert response.get_json()["error"] == "No data provided"


def test_add_points_invalid_frame_idx(client, app_state, tmp_path):
    session_id = "session-bad-frame"
    seed_session(app_state, session_id, str(tmp_path))
    payload = {
        "session_id": session_id,
        "frame_idx": "bad",
        "obj_id": "1",
        "points": json.dumps([[[1, 2]]]),
        "labels": json.dumps([[1]]),
    }

    response = client.post("/add_points", data=payload)

    assert response.status_code == 400
    assert response.get_json()["error"] == "frame_idx and obj_id must be integers"


def test_add_points_invalid_json(client, app_state, tmp_path):
    session_id = "session-bad-json"
    seed_session(app_state, session_id, str(tmp_path))
    payload = {
        "session_id": session_id,
        "frame_idx": "0",
        "obj_id": "1",
        "points": "{",
        "labels": json.dumps([[1]]),
    }

    response = client.post("/add_points", data=payload)

    assert response.status_code == 400
    assert response.get_json()["error"] == "Invalid points or labels format"


def test_add_points_oom_exit(app_state, monkeypatch, tmp_path):
    class OOMPredictor:
        def add_new_points_or_box(self, **kwargs):
            raise RuntimeError("CUDA out of memory")

    session_id = "session-oom"
    seed_session(app_state, session_id, str(tmp_path))
    payload = {
        "session_id": session_id,
        "frame_idx": "0",
        "obj_id": "1",
        "points": json.dumps([[[1, 2]]]),
        "labels": json.dumps([[1]]),
    }

    monkeypatch.setattr(app_state, "predictor", OOMPredictor())

    def raise_exit(code):
        raise SystemExit(code)

    monkeypatch.setattr(os, "_exit", raise_exit)

    with app_state.app.test_client() as client:
        with pytest.raises(SystemExit):
            client.post("/add_points", data=payload)


def test_add_points_non_oom_raises(app_state, monkeypatch, tmp_path):
    class FailPredictor:
        def add_new_points_or_box(self, **kwargs):
            raise RuntimeError("boom")

    session_id = "session-fail"
    seed_session(app_state, session_id, str(tmp_path))
    payload = {
        "session_id": session_id,
        "frame_idx": "0",
        "obj_id": "1",
        "points": json.dumps([[[1, 2]]]),
        "labels": json.dumps([[1]]),
    }

    monkeypatch.setattr(app_state, "predictor", FailPredictor())
    monkeypatch.setattr(app_state.app, "testing", True)
    with app_state.app.test_client() as client:
        with pytest.raises(RuntimeError):
            client.post("/add_points", data=payload)


def test_undo_last_point_clears_when_empty(client, app_state, tmp_path):
    session_id = "session-undo"
    points_history = {
        (0, 0): {
            "points": [[10, 20]],
            "labels": [1],
            "format": "flat",
        }
    }
    seed_session(app_state, session_id, str(tmp_path), points_history=points_history)

    response = client.post(
        "/undo_last_point",
        data={"session_id": session_id, "frame_idx": "0", "obj_id": "0"},
    )

    assert response.status_code == 200
    assert response.mimetype == "application/octet-stream"
    zip_bytes = io.BytesIO(response.get_data())
    with zipfile.ZipFile(zip_bytes, "r") as zip_file:
        assert "masks.nii.gz" in zip_file.namelist()
    assert (0, 0) not in app_state.inference_states[session_id]["points_history"]


def test_undo_last_point_no_data(app_state, monkeypatch):
    monkeypatch.setattr(app_state, "request", DummyRequest(form_data=None))
    with app_state.app.app_context():
        response, status = app_state.undo_last_point()

    assert status == 400
    assert response.get_json()["error"] == "No data provided"


def test_undo_last_point_requires_fields(client):
    response = client.post("/undo_last_point", data={"session_id": "session-missing"})

    assert response.status_code == 400
    assert response.get_json()["error"] == "session_id, frame_idx, and obj_id are required"


def test_undo_last_point_invalid_session(client):
    response = client.post(
        "/undo_last_point",
        data={"session_id": "missing", "frame_idx": "0", "obj_id": "0"},
    )

    assert response.status_code == 400
    assert response.get_json()["error"] == "Invalid session_id"


def test_undo_last_point_no_points(client, app_state, tmp_path):
    session_id = "session-empty"
    seed_session(app_state, session_id, str(tmp_path), points_history={})

    response = client.post(
        "/undo_last_point",
        data={"session_id": session_id, "frame_idx": "0", "obj_id": "0"},
    )

    assert response.status_code == 400
    assert response.get_json()["error"] == "No points to undo"


def test_undo_last_point_invalid_frame_idx(client):
    response = client.post(
        "/undo_last_point",
        data={"session_id": "missing", "frame_idx": "bad", "obj_id": "0"},
    )

    assert response.status_code == 400
    assert response.get_json()["error"] == "frame_idx and obj_id must be integers"


def test_undo_last_point_corrupted_history(client, app_state, tmp_path):
    session_id = "session-corrupt"
    points_history = {
        (0, 0): {"points": [[1, 2], [3, 4]], "labels": [1], "format": "flat"}
    }
    seed_session(app_state, session_id, str(tmp_path), points_history=points_history)

    response = client.post(
        "/undo_last_point",
        data={"session_id": session_id, "frame_idx": "0", "obj_id": "0"},
    )

    assert response.status_code == 500
    assert response.get_json()["error"] == "Point history is corrupted"


def test_undo_last_point_returns_mask(client, app_state, tmp_path):
    session_id = "session-remaining"
    points_history = {
        (0, 0): {
            "points": [[10, 20], [30, 40]],
            "labels": [1, 0],
            "format": "flat",
        }
    }
    seed_session(app_state, session_id, str(tmp_path), points_history=points_history)

    response = client.post(
        "/undo_last_point",
        data={"session_id": session_id, "frame_idx": "0", "obj_id": "0"},
    )

    assert response.status_code == 200
    assert response.mimetype == "application/octet-stream"


def test_undo_last_point_oom_exit(app_state, monkeypatch, tmp_path):
    class OOMPredictor:
        def add_new_points_or_box(self, **kwargs):
            raise RuntimeError("CUDA out of memory")

    session_id = "session-undo-oom"
    points_history = {(0, 0): {"points": [[1, 2]], "labels": [1], "format": "flat"}}
    seed_session(app_state, session_id, str(tmp_path), points_history=points_history)

    monkeypatch.setattr(app_state, "predictor", OOMPredictor())

    def raise_exit(code):
        raise SystemExit(code)

    monkeypatch.setattr(os, "_exit", raise_exit)

    with app_state.app.test_client() as client:
        response = client.post(
            "/undo_last_point",
            data={"session_id": session_id, "frame_idx": "0", "obj_id": "0"},
        )

    assert response.status_code == 500


def test_undo_last_point_non_oom_raises(app_state, monkeypatch, tmp_path):
    class FailPredictor:
        def add_new_points_or_box(self, **kwargs):
            raise RuntimeError("boom")

    session_id = "session-undo-fail"
    points_history = {(0, 0): {"points": [[1, 2]], "labels": [1], "format": "flat"}}
    seed_session(app_state, session_id, str(tmp_path), points_history=points_history)

    monkeypatch.setattr(app_state, "predictor", FailPredictor())
    monkeypatch.setattr(app_state.app, "testing", True)
    with app_state.app.test_client() as client:
        with pytest.raises(RuntimeError):
            client.post(
                "/undo_last_point",
                data={"session_id": session_id, "frame_idx": "0", "obj_id": "0"},
            )


def test_undo_propagate_reapplies_points(client, app_state, fake_predictor, tmp_path):
    session_id = "session-propagate"
    jpg_dir = tmp_path / "jpgs"
    jpg_dir.mkdir()
    points_history = {
        (0, 0): {
            "points": [[1, 2], [3, 4]],
            "labels": [1, 0],
            "format": "flat",
        }
    }
    seed_session(
        app_state,
        session_id,
        str(tmp_path),
        jpg_dir=str(jpg_dir),
        points_history=points_history,
        n_frames=1,
    )

    response = client.post("/undo_propagate", data={"session_id": session_id})

    assert response.status_code == 200
    assert response.mimetype == "application/octet-stream"
    zip_bytes = io.BytesIO(response.get_data())
    with zipfile.ZipFile(zip_bytes, "r") as zip_file:
        assert "masks.nii.gz" in zip_file.namelist()
    assert fake_predictor.calls[0][0] == "init_state"


def test_get_server_status(client):
    response = client.post("/get_server_status", json={"api_key": "ignored"})

    assert response.status_code == 200
    assert response.get_json()["status"] == "happily running"


def test_file_too_large_handler(app_state):
    with app_state.app.app_context():
        response, status = app_state.file_too_large(Exception("big"))

    assert status == 413
    assert response.get_json()["error"] == "File is too large"


def test_clear_session_no_data(app_state, monkeypatch):
    monkeypatch.setattr(app_state, "request", DummyRequest(json_data=None, form_data=None))
    with app_state.app.app_context():
        response, status = app_state.clear_session()

    assert status == 400
    assert response.get_json()["error"] == "No data provided"


def test_clear_session_missing_session_id(client):
    response = client.post("/clear_session", data={"other": "value"})

    assert response.status_code == 400
    assert response.get_json()["error"] == "session_id is required"


def test_clear_session_invalid_session(client):
    response = client.post("/clear_session", data={"session_id": "missing"})

    assert response.status_code == 400
    assert response.get_json()["error"] == "Invalid session_id"


def test_clear_session_success(client, app_state, tmp_path):
    session_id = "session-clear"
    temp_dir = tmp_path / "temp"
    temp_dir.mkdir()
    app_state.inference_states[session_id] = {"temp_dir": str(temp_dir)}

    response = client.post("/clear_session", data={"session_id": session_id})

    assert response.status_code == 200
    assert response.get_json()["status"] == "cleared"
    assert session_id not in app_state.inference_states
    assert not temp_dir.exists()


def test_clear_session_json_success(client, app_state, tmp_path):
    session_id = "session-clear-json"
    temp_dir = tmp_path / "temp"
    temp_dir.mkdir()
    app_state.inference_states[session_id] = {"temp_dir": str(temp_dir)}

    response = client.post("/clear_session", json={"session_id": session_id})

    assert response.status_code == 200
    assert response.get_json()["status"] == "cleared"


def test_clear_session_resources_returns_false(app_state):
    assert app_state.clear_session_resources("missing") is False


def test_clear_session_resources_removes_job_and_cache(app_state, monkeypatch, tmp_path):
    class FakeCuda:
        def __init__(self):
            self.emptied = False

        def is_available(self):
            return True

        def empty_cache(self):
            self.emptied = True

    class FakeTorch:
        def __init__(self):
            self.cuda = FakeCuda()

    session_id = "session-resource"
    temp_dir = tmp_path / "temp"
    temp_dir.mkdir()
    fake_scheduler = FakeScheduler()
    job_id = f"session_cleanup_{session_id}"
    fake_scheduler.jobs[job_id] = True

    app_state.inference_states[session_id] = {"temp_dir": str(temp_dir)}

    monkeypatch.setattr(app_state, "scheduler", fake_scheduler)
    monkeypatch.setattr(app_state, "torch", FakeTorch())

    cleared = app_state.clear_session_resources(session_id, reason="manual")

    assert cleared is True
    assert job_id in fake_scheduler.removed
    assert not temp_dir.exists()


def test_clear_session_resources_handles_missing_dir(app_state, monkeypatch):
    session_id = "session-missing-dir"
    app_state.inference_states[session_id] = {"temp_dir": "missing"}
    fake_scheduler = FakeScheduler()
    monkeypatch.setattr(app_state, "scheduler", fake_scheduler)

    def raise_missing(path):
        raise FileNotFoundError()

    monkeypatch.setattr(app_state.shutil, "rmtree", raise_missing)

    cleared = app_state.clear_session_resources(session_id, reason="cleanup")

    assert cleared is True


def test_delete_session_noop(app_state):
    assert app_state.delete_session("missing") is None


def test_set_or_reset_timer_replaces_job(app_state, monkeypatch):
    fake_scheduler = FakeScheduler()
    job_id = "session_cleanup_test"
    fake_scheduler.jobs[job_id] = True
    monkeypatch.setattr(app_state, "scheduler", fake_scheduler)

    app_state.set_or_reset_timer("test", timeout_seconds=1)

    assert job_id in fake_scheduler.removed
    assert job_id in fake_scheduler.jobs


def test_flatten_format_helpers(app_state):
    assert app_state.flatten_points_labels("bad", "bad") == (None, None, None)

    flat_points, flat_labels, fmt = app_state.flatten_points_labels(
        [[[1, 2], [3, 4]]], [[1, 0]]
    )
    assert flat_points == [[1, 2], [3, 4]]
    assert flat_labels == [1, 0]
    assert fmt == "nested"

    flat_points, flat_labels, fmt = app_state.flatten_points_labels([[1, 2]], [1])
    assert fmt == "flat"
    assert app_state.format_points_labels(flat_points, flat_labels, fmt) == (
        flat_points,
        flat_labels,
    )
    assert app_state.format_points_labels(flat_points, flat_labels, "nested") == (
        [flat_points],
        [flat_labels],
    )


def test_parse_int(app_state):
    assert app_state.parse_int(None) is None
    assert app_state.parse_int("bad") is None
    assert app_state.parse_int("3") == 3


def test_initialize_video_no_zip(client):
    response = client.post("/initialize_video", data={})

    assert response.status_code == 400
    assert response.get_json()["error"] == "No zip file provided"


def test_initialize_video_empty_file_size(client, monkeypatch):
    zip_buffer = build_dicom_zip()

    def fake_getsize(path):
        return 0

    monkeypatch.setattr(os.path, "getsize", fake_getsize)

    response = client.post(
        "/initialize_video",
        data={"data_binary": (zip_buffer, "test.zip")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert response.get_json()["error"] == "File size is 0"


def test_initialize_video_success(client, app_state, monkeypatch):
    zip_buffer = build_dicom_zip()

    class FakeDicom:
        def __init__(self, pixel_array, attrs):
            self.pixel_array = pixel_array
            for key, value in attrs.items():
                setattr(self, key, value)

    dicoms = {
        "a.dcm": FakeDicom(np.ones((2, 2), dtype=np.float32), {"InstanceNumber": 1}),
        "b.dcm": FakeDicom(
            np.ones((2, 2), dtype=np.float32), {"ImagePositionPatient": [0, 0, 2]}
        ),
        "c.dcm": FakeDicom(np.ones((2, 2), dtype=np.float32), {"SliceLocation": 3}),
        "d.dcm": FakeDicom(np.ones((2, 2), dtype=np.float32), {}),
    }

    def fake_dcmread(path):
        return dicoms[os.path.basename(path)]

    monkeypatch.setattr(app_state.pydicom, "dcmread", fake_dcmread)

    response = client.post(
        "/initialize_video",
        data={
            "data_binary": (zip_buffer, "test.zip"),
            "ww": "100",
            "wl": "50",
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    data = response.get_json()
    assert "session_id" in data
    assert data["session_id"] in app_state.inference_states


def test_initialize_video_oom_exit(app_state, monkeypatch):
    class OOMPredictor:
        def init_state(self, video_path):
            raise RuntimeError("CUDA out of memory")

    zip_buffer = build_dicom_zip()

    class FakeDicom:
        def __init__(self):
            self.pixel_array = np.ones((2, 2), dtype=np.float32)

    def fake_dcmread(path):
        return FakeDicom()

    monkeypatch.setattr(app_state.pydicom, "dcmread", fake_dcmread)
    monkeypatch.setattr(app_state, "predictor", OOMPredictor())

    def raise_exit(code):
        raise SystemExit(code)

    monkeypatch.setattr(os, "_exit", raise_exit)

    with app_state.app.test_client() as client:
        with pytest.raises(SystemExit):
            client.post(
                "/initialize_video",
                data={"data_binary": (zip_buffer, "test.zip")},
                content_type="multipart/form-data",
            )


def test_initialize_video_non_oom_raises(app_state, monkeypatch):
    class FailPredictor:
        def init_state(self, video_path):
            raise RuntimeError("boom")

    zip_buffer = build_dicom_zip()

    class FakeDicom:
        def __init__(self):
            self.pixel_array = np.ones((2, 2), dtype=np.float32)

    def fake_dcmread(path):
        return FakeDicom()

    monkeypatch.setattr(app_state.pydicom, "dcmread", fake_dcmread)
    monkeypatch.setattr(app_state, "predictor", FailPredictor())

    monkeypatch.setattr(app_state.app, "testing", True)
    with app_state.app.test_client() as client:
        with pytest.raises(RuntimeError):
            client.post(
                "/initialize_video",
                data={"data_binary": (zip_buffer, "test.zip")},
                content_type="multipart/form-data",
            )


def test_convert_masks_to_nii(app_state):
    video_segments = {0: {0: np.ones((2, 2), dtype=np.uint8)}}
    nii = app_state.convert_masks_to_nii(video_segments, n_frames=1)
    assert nii.get_fdata().shape == (1, 2, 2)


def test_propagate_masks_no_data(app_state, monkeypatch):
    monkeypatch.setattr(app_state, "request", DummyRequest(form_data=None))
    with app_state.app.app_context():
        response, status = app_state.propagate_masks()

    assert status == 400
    assert response.get_json()["error"] == "No data provided"


def test_propagate_masks_requires_session(client):
    response = client.post("/propagate_masks", data={})

    assert response.status_code == 400
    assert response.get_json()["error"] == "session_id is required"


def test_propagate_masks_invalid_session(client):
    response = client.post("/propagate_masks", data={"session_id": "missing"})

    assert response.status_code == 400
    assert response.get_json()["error"] == "Invalid session_id"


def test_propagate_masks_success_with_reverse(app_state, monkeypatch, tmp_path):
    class ReversePredictor:
        def __init__(self):
            self.calls = []

        def propagate_in_video(self, inference_state, start_frame_idx=None, reverse=False):
            self.calls.append((start_frame_idx, reverse))
            logits = SimpleTensor(np.ones((1, 1, 2, 2), dtype=np.float32))
            if start_frame_idx is None:
                yield 1, [0], logits
            else:
                yield 0, [0], logits

    session_id = "session-prop"
    app_state.inference_states[session_id] = {
        "inference_state": {},
        "n_frames": 2,
    }
    predictor = ReversePredictor()
    monkeypatch.setattr(app_state, "predictor", predictor)
    monkeypatch.setattr(app_state, "set_or_reset_timer", lambda *args, **kwargs: None)

    with app_state.app.test_client() as client:
        response = client.post("/propagate_masks", data={"session_id": session_id})

    assert response.status_code == 200
    assert predictor.calls[0] == (None, False)
    assert predictor.calls[1] == (1, True)


def test_propagate_masks_success_without_reverse(app_state, monkeypatch, tmp_path):
    class SimplePredictor:
        def __init__(self):
            self.calls = []

        def propagate_in_video(self, inference_state, start_frame_idx=None, reverse=False):
            self.calls.append((start_frame_idx, reverse))
            logits = SimpleTensor(np.ones((1, 1, 2, 2), dtype=np.float32))
            yield 0, [0], logits

    session_id = "session-prop-no-reverse"
    app_state.inference_states[session_id] = {
        "inference_state": {},
        "n_frames": 1,
    }
    predictor = SimplePredictor()
    monkeypatch.setattr(app_state, "predictor", predictor)
    monkeypatch.setattr(app_state, "set_or_reset_timer", lambda *args, **kwargs: None)

    with app_state.app.test_client() as client:
        response = client.post("/propagate_masks", data={"session_id": session_id})

    assert response.status_code == 200
    assert predictor.calls == [(None, False)]


def test_undo_propagate_no_data(app_state, monkeypatch):
    monkeypatch.setattr(app_state, "request", DummyRequest(form_data=None))
    with app_state.app.app_context():
        response, status = app_state.undo_propagate()

    assert status == 400
    assert response.get_json()["error"] == "No data provided"


def test_undo_propagate_missing_session_id(client):
    response = client.post("/undo_propagate", data={})

    assert response.status_code == 400
    assert response.get_json()["error"] == "session_id is required"


def test_undo_propagate_invalid_session(client):
    response = client.post("/undo_propagate", data={"session_id": "missing"})

    assert response.status_code == 400
    assert response.get_json()["error"] == "Invalid session_id"


def test_undo_propagate_missing_jpg_dir(client, app_state, tmp_path):
    session_id = "session-no-jpg"
    seed_session(app_state, session_id, str(tmp_path))

    response = client.post("/undo_propagate", data={"session_id": session_id})

    assert response.status_code == 400
    assert response.get_json()["error"] == "No video data for session"


def test_undo_propagate_skips_empty_history(app_state, monkeypatch, tmp_path):
    class SimplePredictor:
        def init_state(self, video_path):
            return {"video_path": video_path}

        def add_new_points_or_box(self, **kwargs):
            return None

    session_id = "session-empty-history"
    jpg_dir = tmp_path / "jpgs"
    jpg_dir.mkdir()
    points_history = {(0, 0): None, (1, 1): {"points": [], "labels": [], "format": "flat"}}
    seed_session(
        app_state,
        session_id,
        str(tmp_path),
        jpg_dir=str(jpg_dir),
        points_history=points_history,
        n_frames=1,
        frame_shape=(2, 2),
    )

    monkeypatch.setattr(app_state, "predictor", SimplePredictor())
    monkeypatch.setattr(app_state, "set_or_reset_timer", lambda *args, **kwargs: None)

    with app_state.app.test_client() as client:
        response = client.post("/undo_propagate", data={"session_id": session_id})

    assert response.status_code == 200
    assert response.mimetype == "application/octet-stream"
    zip_bytes = io.BytesIO(response.get_data())
    with zipfile.ZipFile(zip_bytes, "r") as zip_file:
        assert "masks.nii.gz" in zip_file.namelist()


def test_undo_propagate_oom_exit(app_state, monkeypatch, tmp_path):
    class OOMPredictor:
        def init_state(self, video_path):
            raise RuntimeError("CUDA out of memory")

    session_id = "session-undo-prop-oom"
    jpg_dir = tmp_path / "jpgs"
    jpg_dir.mkdir()
    seed_session(app_state, session_id, str(tmp_path), jpg_dir=str(jpg_dir))

    monkeypatch.setattr(app_state, "predictor", OOMPredictor())

    def raise_exit(code):
        raise SystemExit(code)

    monkeypatch.setattr(os, "_exit", raise_exit)

    with app_state.app.test_client() as client:
        with pytest.raises(SystemExit):
            client.post("/undo_propagate", data={"session_id": session_id})


def test_undo_propagate_non_oom_raises(app_state, monkeypatch, tmp_path):
    class FailPredictor:
        def init_state(self, video_path):
            raise RuntimeError("boom")

    session_id = "session-undo-prop-fail"
    jpg_dir = tmp_path / "jpgs"
    jpg_dir.mkdir()
    seed_session(app_state, session_id, str(tmp_path), jpg_dir=str(jpg_dir))

    monkeypatch.setattr(app_state, "predictor", FailPredictor())

    monkeypatch.setattr(app_state.app, "testing", True)
    with app_state.app.test_client() as client:
        with pytest.raises(RuntimeError):
            client.post("/undo_propagate", data={"session_id": session_id})


def test_cuda_oom_guard_handles_exit(app_state, monkeypatch):
    def explode():
        raise RuntimeError("CUDA out of memory")

    wrapped = app_state.cuda_oom_guard(explode)

    def raise_exit(code):
        raise SystemExit(code)

    monkeypatch.setattr(os, "_exit", raise_exit)

    with pytest.raises(SystemExit):
        wrapped()


def test_cuda_oom_guard_reraises(app_state):
    def explode():
        raise RuntimeError("boom")

    wrapped = app_state.cuda_oom_guard(explode)
    with pytest.raises(RuntimeError):
        wrapped()
