import builtins
import importlib.util
import os
import runpy
import sys
import types

import pytest

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
APP_PATH = os.path.join(ROOT_DIR, "app.py")


class FakeScheduler:
    def __init__(self):
        self.started = False

    def start(self):
        self.started = True

    def get_job(self, job_id):
        return None

    def remove_job(self, job_id):
        return None

    def add_job(self, func, trigger, id, args, replace_existing):
        return None


class FakeIntervalTrigger:
    def __init__(self, seconds):
        self.seconds = seconds


def install_fake_apscheduler(monkeypatch):
    apscheduler = types.ModuleType("apscheduler")
    apscheduler.__path__ = []
    schedulers = types.ModuleType("apscheduler.schedulers")
    schedulers.__path__ = []
    triggers = types.ModuleType("apscheduler.triggers")
    triggers.__path__ = []
    background = types.ModuleType("apscheduler.schedulers.background")
    interval = types.ModuleType("apscheduler.triggers.interval")
    background.BackgroundScheduler = FakeScheduler
    interval.IntervalTrigger = FakeIntervalTrigger
    schedulers.background = background
    triggers.interval = interval
    apscheduler.schedulers = schedulers
    apscheduler.triggers = triggers

    monkeypatch.setitem(sys.modules, "apscheduler", apscheduler)
    monkeypatch.setitem(sys.modules, "apscheduler.schedulers", schedulers)
    monkeypatch.setitem(sys.modules, "apscheduler.schedulers.background", background)
    monkeypatch.setitem(sys.modules, "apscheduler.triggers", triggers)
    monkeypatch.setitem(sys.modules, "apscheduler.triggers.interval", interval)


def make_blocking_import(blocked):
    original_import = builtins.__import__

    def importer(name, globals=None, locals=None, fromlist=(), level=0):
        base = name.split(".")[0]
        if base in blocked or name in blocked:
            raise ImportError(f"blocked: {name}")
        return original_import(name, globals, locals, fromlist, level)

    return importer


def load_app_module(monkeypatch, module_name, env, importer=None, extra_modules=None):
    for key, value in env.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)

    if importer is not None:
        monkeypatch.setattr(builtins, "__import__", importer)

    if extra_modules:
        for name, module in extra_modules.items():
            monkeypatch.setitem(sys.modules, name, module)

    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, APP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_torch_module():
    torch = types.ModuleType("torch")

    class FakeProps:
        major = 8

    class FakeCuda:
        def is_available(self):
            return True

        def get_device_properties(self, index):
            return FakeProps()

        def empty_cache(self):
            return None

    class FakeAutocast:
        def __enter__(self):
            return self

    def autocast(device_type=None, dtype=None):
        return FakeAutocast()

    torch.cuda = FakeCuda()
    torch.autocast = autocast
    torch.bfloat16 = "bfloat16"
    torch.backends = types.SimpleNamespace(
        cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=False)),
        cudnn=types.SimpleNamespace(allow_tf32=False),
    )
    return torch


def fake_sam2_modules():
    sam2 = types.ModuleType("sam2")
    sam2.__path__ = []
    sam2.__file__ = "/tmp/sam2/__init__.py"

    build_sam = types.ModuleType("sam2.build_sam")

    def build_sam2_video_predictor(model_cfg, checkpoint):
        return "fake_predictor"

    def build_sam2(*args, **kwargs):
        return None

    build_sam.build_sam2_video_predictor = build_sam2_video_predictor
    build_sam.build_sam2 = build_sam2

    image_predictor = types.ModuleType("sam2.sam2_image_predictor")
    image_predictor.SAM2ImagePredictor = object
    sam2.build_sam = build_sam
    sam2.sam2_image_predictor = image_predictor

    return {
        "sam2": sam2,
        "sam2.build_sam": build_sam,
        "sam2.sam2_image_predictor": image_predictor,
    }


def test_import_handles_missing_torch_and_sam2(monkeypatch):
    install_fake_apscheduler(monkeypatch)
    importer = make_blocking_import({"torch", "sam2"})

    module = load_app_module(
        monkeypatch,
        "app_missing_torch",
        {"SAM2_ALLOW_NO_MODEL": "1", "SAM2_SKIP_MODEL": "1"},
        importer=importer,
    )

    assert module.torch is None
    assert module.SAM2_AVAILABLE is False


def test_import_raises_without_allow_no_model(monkeypatch):
    install_fake_apscheduler(monkeypatch)
    importer = make_blocking_import({"sam2"})

    with pytest.raises(ImportError):
        load_app_module(
            monkeypatch,
            "app_no_allow",
            {"SAM2_ALLOW_NO_MODEL": "0", "SAM2_SKIP_MODEL": "1"},
            importer=importer,
        )


def test_import_requires_model_when_skip_false(monkeypatch):
    install_fake_apscheduler(monkeypatch)
    importer = make_blocking_import({"sam2"})

    with pytest.raises(RuntimeError):
        load_app_module(
            monkeypatch,
            "app_requires_model",
            {"SAM2_ALLOW_NO_MODEL": "1", "SAM2_SKIP_MODEL": "0"},
            importer=importer,
        )


def test_import_with_fake_sam2_and_torch(monkeypatch):
    install_fake_apscheduler(monkeypatch)
    modules = {"torch": fake_torch_module()}
    modules.update(fake_sam2_modules())

    module = load_app_module(
        monkeypatch,
        "app_fake_modules",
        {"SAM2_ALLOW_NO_MODEL": "0", "SAM2_SKIP_MODEL": "0"},
        extra_modules=modules,
    )

    assert module.predictor == "fake_predictor"
    assert module.torch.backends.cuda.matmul.allow_tf32 is True
    assert module.torch.backends.cudnn.allow_tf32 is True


def test_main_runs_app(monkeypatch):
    install_fake_apscheduler(monkeypatch)
    monkeypatch.setenv("SAM2_ALLOW_NO_MODEL", "1")
    monkeypatch.setenv("SAM2_SKIP_MODEL", "1")

    import flask

    called = {}

    def fake_run(self, *args, **kwargs):
        called["run"] = True

    monkeypatch.setattr(flask.Flask, "run", fake_run)

    runpy.run_path(APP_PATH, run_name="__main__")

    assert called.get("run") is True
