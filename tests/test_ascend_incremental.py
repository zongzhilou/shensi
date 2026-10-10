"""Backend selection, command construction and GDAR math regression checks."""

import argparse
import copy
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from shensi.utils import accelerator, ascend_env
from shensi.utils.linalg import symmetric_eigh


def connection_source():
    """Load the real pure-Torch connection without Megatron's package-level specs."""
    path = (
        Path(__file__).resolve().parents[1] / "src/shensi/recipes/paper/"
        "gated_delta_attn_res/common/models/megatron/gdar_connection.py"
    )
    name = "_tested_gdar_connection"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def test_explicit_vendor_selection():
    assert accelerator.backend_name({"SHENSI_BACKEND": "ascend"}) == "npu"
    assert accelerator.backend_name({"SHENSI_BACKEND": "nvidia"}) == "cuda"
    assert accelerator.backend_name({"SHENSI_BACKEND": "cpu", "ASCEND_HOME_PATH": "/x"}) == "cpu"
    with pytest.raises(ValueError, match="SHENSI_BACKEND"):
        accelerator.backend_name({"SHENSI_BACKEND": "typo"})


def test_npu_visibility_and_platform():
    env = {
        "SHENSI_BACKEND": "npu",
        "ASCEND_RT_VISIBLE_DEVICES": "2,4",
        "CUDA_VISIBLE_DEVICES": "0",
        "VLLM_PLUGINS": "ascend_plugin",
    }
    assert accelerator.visible_devices(env) == 2
    accelerator.configure_verl_env(env)
    assert env["VERL_PLATFORM"] == "huawei"
    assert env["VLLM_PLUGINS"] == "ascend_plugin"
    with pytest.raises(ValueError, match="huawei"):
        accelerator.configure_verl_env({"SHENSI_BACKEND": "npu", "VERL_PLATFORM": "nvidia_noipc"})


def test_cuda_launch_defaults_unchanged():
    env = {"SHENSI_BACKEND": "cuda", "CUDA_VISIBLE_DEVICES": "3,5"}
    assert accelerator.visible_devices(env) == 2
    accelerator.configure_verl_env(env)
    assert env["CUDA_VISIBLE_DEVICES"] == "3,5"
    assert env["VERL_PLATFORM"] == "nvidia_noipc"
    assert env["VLLM_PLUGINS"] == ""


def test_hccl_parser_extension_is_npu_only(monkeypatch):
    parser = argparse.ArgumentParser()
    parser.add_argument("--distributed-backend", choices=("nccl", "gloo"))
    monkeypatch.setenv("SHENSI_BACKEND", "cuda")
    accelerator.extend_distributed_backend_choices(parser)
    with pytest.raises(SystemExit):
        parser.parse_args(["--distributed-backend", "hccl"])
    monkeypatch.setenv("SHENSI_BACKEND", "npu")
    accelerator.extend_distributed_backend_choices(parser)
    assert parser.parse_args(["--distributed-backend", "hccl"]).distributed_backend == "hccl"


def test_bootstrap_failure_is_not_silenced(monkeypatch):
    monkeypatch.setenv("SHENSI_BACKEND", "npu")
    monkeypatch.setattr(accelerator, "_megatron_ready", False)
    calls = []
    monkeypatch.setattr(accelerator, "device", lambda backend: calls.append(backend))

    def broken(name):
        calls.append(name)
        raise ImportError("incompatible vendor stack")

    monkeypatch.setattr(accelerator.importlib, "import_module", broken)
    with pytest.raises(ImportError, match="incompatible vendor stack"):
        accelerator.bootstrap_megatron()
    assert calls == ["npu", "megatron_adaptor"]
    assert not accelerator._megatron_ready


def test_cpu_rng_isolation(monkeypatch):
    monkeypatch.setenv("SHENSI_BACKEND", "cpu")
    torch.manual_seed(12)
    before = torch.random.get_rng_state().clone()
    with accelerator.isolated_rng(91):
        torch.randn(10)
    assert torch.equal(before, torch.random.get_rng_state())


def test_preflight_missing_stack_is_failure(monkeypatch, tmp_path):
    monkeypatch.setenv("ASCEND_HOME_PATH", str(tmp_path))
    monkeypatch.setattr(ascend_env, "_probe", lambda code: {"ok": False, "error": "no NPU"})
    result = ascend_env.inspect("reference")
    assert not result["ok"]
    assert not result["checks"]["torch_npu"]["ok"]
    assert "megatron" not in result["checks"]


def test_npu_command_and_smoke_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("SHENSI_BACKEND", "cpu")
    monkeypatch.setenv("SHENSI_FS", str(tmp_path))
    from shensi.recipes.paper.gated_delta_attn_res.common import config, runner
    from shensi.recipes.paper.gated_delta_attn_res.common.train.launcher import build_command

    captured = {}

    def capture(cfg, dry_run, watch=None):
        captured.update(cfg=cfg, dry_run=dry_run)
        return 0

    monkeypatch.setattr(runner, "run", capture)
    args = SimpleNamespace(
        config=None,
        profile="default",
        smoke=True,
        load="/resume",
        override=["train.model.hidden_size=64", "experiment.envs.SHENSI_BACKEND=npu"],
        dry_run=True,
    )
    assert config.train_from_args("stage1_pretrain", args) == 0
    assert captured["dry_run"] is True
    assert captured["cfg"]["train"]["model"]["hidden_size"] == 64
    cmd = build_command(captured["cfg"])
    assert cmd[cmd.index("--distributed-backend") + 1] == "nccl"
    assert cmd[cmd.index("--load") + 1] == "/resume"
    cuda_cfg = copy.deepcopy(captured["cfg"])
    cuda_cfg["experiment"]["envs"]["SHENSI_BACKEND"] = "cuda"
    assert "--distributed-backend" not in build_command(cuda_cfg)


def test_opd_smoke_inherits_tiny_not_million_step_default(monkeypatch, tmp_path):
    monkeypatch.setenv("SHENSI_BACKEND", "cpu")
    monkeypatch.setenv("SHENSI_FS", str(tmp_path))
    from shensi.recipes.paper.gated_delta_attn_res.common.config import smoke_config

    cfg = smoke_config("stage3_opd")
    assert cfg["train"]["model"]["train_iters"] == 5
    assert cfg["train"]["model"]["hidden_size"] == 256
    assert cfg["train"]["data"]["mock_data"]
    assert "base" not in cfg


@pytest.mark.parametrize("mode", ["cpu", "native"])
def test_whitening_math_cpu_cuda_path_preserved(monkeypatch, mode):
    monkeypatch.setenv("SHENSI_NPU_EIGH", mode)
    torch.manual_seed(8)
    a = torch.randn(2, 8, 8)
    a = a @ a.transpose(-1, -2) + torch.eye(8) * 0.01
    expected = torch.linalg.eigh(a)
    actual = symmetric_eigh(a)
    assert all(torch.equal(x, y) for x, y in zip(expected, actual, strict=True))


@pytest.mark.parametrize("whiten", ["full", "per_head", "diag", "off"])
def test_hf_mcore_connection_forward_and_gradients(whiten):
    mcore_read = connection_source()._depth_read
    from shensi.recipes.paper.gated_delta_attn_res.common.models.transformers.modeling_qwen3_gdar import (
        _depth_read as hf_read,
    )

    torch.manual_seed(18)
    a = torch.randn(8, 3, 16, requires_grad=True)
    q = torch.randn(8, 16, requires_grad=True)
    b, r = a.detach().clone().requires_grad_(), q.detach().clone().requires_grad_()
    kw = dict(eps=1e-6, heads=2, null=True, whiten=whiten, ridge=1e-3)
    actual, expected = mcore_read(a, q, **kw), hf_read(b, r, **kw)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(a.grad, b.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(q.grad, r.grad, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(importlib.util.find_spec("torch_npu") is None, reason="requires real NPU")
def test_real_npu_whitening_forward_backward():
    _depth_read = connection_source()._depth_read

    dev = accelerator.device("npu")  # Installed but unusable torch_npu is a failure, not a skip.
    torch.manual_seed(28)
    a = torch.randn(8, 3, 16, requires_grad=True)
    q = torch.randn(8, 16, requires_grad=True)
    b = a.detach().to(dev).requires_grad_()
    r = q.detach().to(dev).requires_grad_()
    kw = dict(eps=1e-6, heads=2, null=True, whiten="full", ridge=1e-3)
    cpu_out, npu_out = _depth_read(a, q, **kw), _depth_read(b, r, **kw)
    torch.testing.assert_close(npu_out.cpu(), cpu_out, atol=2e-3, rtol=2e-3)
    cpu_out.square().sum().backward()
    npu_out.square().sum().backward()
    torch.testing.assert_close(b.grad.cpu(), a.grad, atol=3e-3, rtol=3e-3)
    torch.testing.assert_close(r.grad.cpu(), q.grad, atol=3e-3, rtol=3e-3)
