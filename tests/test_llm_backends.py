"""云端与本地后端隔离 / Backend configuration and HTTP regression tests."""

from __future__ import annotations

import os
import sys
import json
import builtins
from pathlib import Path

import httpx
import pytest
from openai import OpenAI

sys.path.append(os.getcwd())

from shopharness import cli
from shopharness.config import Settings, load_cloud_settings
from shopharness.llm.base import LLMError, Message
from shopharness.llm.mock_client import MockLLM
from shopharness.llm.vllm_client import VLLMClient
from shopharness.llm.openai_client import OpenAIClient


@pytest.fixture(autouse = True)
def isolated_env(monkeypatch, tmp_path):
    """Use monkeypatch and tmp_path to isolate all private cloud configuration."""
    for name in ("LLM_BASE_URL", "LLM_API_KEY", "MODEL"):
        monkeypatch.delenv(name, raising = False)
    monkeypatch.chdir(tmp_path)


def write_env(path: Path) -> None:
    """Write fictional cloud settings to path."""
    path.write_text(
        "LLM_BASE_URL=https://cloud.example/api/v3\n"
        "LLM_API_KEY=test-secret\nMODEL=cloud-model\n",
        encoding = "utf-8",
    )


def mock_transport(monkeypatch, handler):
    """Route SDK requests to handler using monkeypatch, without network access."""
    def factory(**kwargs):
        """Build an SDK client from kwargs with an isolated transport."""
        return OpenAI(
            **kwargs,
            http_client = httpx.Client(transport = httpx.MockTransport(handler)),
            max_retries = 0,
        )
    monkeypatch.setattr("shopharness.llm.openai_client.OpenAI", factory)


def response(message: dict) -> httpx.Response:
    """Wrap message in a minimal successful Chat Completions response."""
    return httpx.Response(200, json = {"choices": [{"message": message}]})


def test_explicit_loading_and_precedence(monkeypatch, tmp_path):
    """Verify dotenv precedence and no implicit Settings mutation in tmp_path."""
    write_env(tmp_path / ".env")
    assert Settings().model == "Qwen/Qwen3-8B-FP8"
    assert "LLM_API_KEY" not in os.environ
    cloud = load_cloud_settings()
    assert cloud.model == "cloud-model"
    assert "test-secret" not in repr(cloud)
    monkeypatch.setenv("MODEL", "process-model")
    monkeypatch.setenv("LLM_API_KEY", "process-secret")
    monkeypatch.setenv("LLM_BASE_URL", "https://process.example/v1")
    cloud = load_cloud_settings()
    assert cloud.model == "process-model"
    assert cloud.base_url == "https://process.example/v1"
    assert cloud.api_key.get_secret_value() == "process-secret"
    assert load_cloud_settings(model = "cli-model").model == "cli-model"


def test_missing_and_empty_cloud_configuration(monkeypatch):
    """Check missing values and empty process overrides through monkeypatch."""
    assert load_cloud_settings() is None
    monkeypatch.setenv("LLM_API_KEY", "private-value")
    with pytest.raises(ValueError) as error:
        load_cloud_settings()
    assert "LLM_BASE_URL" in str(error.value)
    assert "MODEL" in str(error.value)
    assert "private-value" not in str(error.value)
    write_env(Path(".env"))
    monkeypatch.setenv("MODEL", "")
    with pytest.raises(ValueError, match = "MODEL"):
        load_cloud_settings()
    assert load_cloud_settings(model = "override").model == "override"


def test_credential_url_is_rejected_without_echo(monkeypatch):
    """Ensure a malformed URL supplied through monkeypatch cannot leak secrets."""
    write_env(Path(".env"))
    monkeypatch.setenv("LLM_BASE_URL", "https://user:private-secret@cloud.example/v1")
    with pytest.raises(ValueError) as error:
        load_cloud_settings()
    assert "private-secret" not in str(error.value)


def test_backend_selection_and_cloud_isolation(monkeypatch):
    """Select all backends and isolate cloud credentials using monkeypatch."""
    settings = Settings()
    assert isinstance(cli.make_llm(cli.parse_args([]), settings), MockLLM)
    write_env(Path(".env"))
    mock_transport(monkeypatch, lambda request: response({"content": "ok"}))
    cloud = cli.make_llm(cli.parse_args(["--model", "override"]), settings)
    assert type(cloud) is OpenAIClient
    assert cloud.model == "override"
    assert cloud.client.api_key == "test-secret"
    cloud.client.close()
    monkeypatch.setenv("LLM_BASE_URL", "invalid-cloud-url")
    assert isinstance(cli.make_llm(cli.parse_args(["--mock"]), settings), MockLLM)
    local = cli.make_llm(cli.parse_args(["--endpoint", "http://localhost:8000/v1"]), settings)
    assert isinstance(local, VLLMClient)
    assert local.model == settings.model
    assert local.client.api_key == "EMPTY"
    local.client.close()


@pytest.mark.parametrize("argv", [
    ["--mock", "--endpoint", "http://localhost:8000/v1"],
    ["--thinking"],
    ["--mock", "--thinking"],
])
def test_invalid_cli_combinations(argv):
    """Reject incompatible flags in argv before constructing a client."""
    with pytest.raises(SystemExit) as error:
        cli.parse_args(argv)
    assert error.value.code == 2


def test_cloud_tool_round_trip(monkeypatch):
    """Verify cloud authentication and tool result serialization with monkeypatch."""
    requests = []

    def handler(request):
        """Respond to request with a function call followed by a final answer."""
        assert request.headers["authorization"] == "Bearer test-secret"
        assert request.url.path == "/api/v3/chat/completions"
        body = json.loads(request.content)
        requests.append(body)
        assert "chat_template_kwargs" not in body
        assert "thinking" not in body
        if len(requests) == 1:
            return response({"role": "assistant", "content": None, "tool_calls": [{
                "id": "call-1",
                "type": "function",
                "function": {"name": "lookup", "arguments": '{"sku": "demo"}'},
            }]})
        assert body["messages"][-1] == {
            "role": "tool", "tool_call_id": "call-1", "content": "有库存",
        }
        assert body["messages"][-2]["tool_calls"][0]["id"] == "call-1"
        return response({"role": "assistant", "content": "有库存"})

    mock_transport(monkeypatch, handler)
    client = OpenAIClient("https://cloud.example/api/v3", "model", "test-secret")
    messages = [Message.user("查询库存")]
    tools = [{"type": "function", "function": {
        "name": "lookup", "parameters": {"type": "object"},
    }}]
    first = client.chat(messages, tools)
    assert first.tool_calls[0].arguments == {"sku": "demo"}
    assert requests[0]["tools"] == tools
    assert requests[0]["tool_choice"] == "auto"
    messages.extend([first, Message.tool("call-1", "lookup", "有库存")])
    assert client.chat(messages).content == "有库存"
    assert "tools" not in requests[-1]
    client.client.close()


@pytest.mark.parametrize("thinking", [False, True])
def test_vllm_options_remain_local(monkeypatch, thinking):
    """Check thinking template options and local credentials with monkeypatch."""
    def handler(request):
        """Inspect the vLLM-only fields on request."""
        assert request.headers["authorization"] == "Bearer EMPTY"
        assert json.loads(request.content)["chat_template_kwargs"] == {
            "enable_thinking": thinking,
        }
        return response({"role": "assistant", "content": "ok"})

    mock_transport(monkeypatch, handler)
    client = VLLMClient(enable_thinking = thinking)
    assert client.chat([Message.user("hello")]).content == "ok"
    client.client.close()


@pytest.mark.parametrize("status", [401, 429, 500])
def test_api_error_does_not_expose_credentials(monkeypatch, status):
    """Sanitize service errors with status using a monkeypatch transport."""
    mock_transport(monkeypatch, lambda request: httpx.Response(
        status, json = {"error": {"message": "test-secret"}},
    ))
    client = OpenAIClient("https://cloud.example/v1", "model", "test-secret")
    with pytest.raises(LLMError) as error:
        client.chat([Message.user("hello")])
    assert str(status) in str(error.value)
    assert "test-secret" not in str(error.value)
    assert error.value.__suppress_context__
    client.client.close()


@pytest.mark.parametrize("message", [
    {"role": "assistant", "tool_calls": [{
        "id": "call", "type": "function",
        "function": {"name": "lookup", "arguments": "invalid-json"},
    }]},
    {"role": "assistant", "tool_calls": [{
        "id": "call", "type": "function",
        "function": {"name": "lookup", "arguments": "[]"},
    }]},
])
def test_malformed_tool_response_is_llm_error(monkeypatch, message):
    """Wrap invalid tool arguments from message using a monkeypatch transport."""
    mock_transport(monkeypatch, lambda request: response(message))
    client = OpenAIClient("https://cloud.example/v1", "model", "test-secret")
    with pytest.raises(LLMError, match = "响应格式无效"):
        client.chat([Message.user("hello")])
    client.client.close()


def test_cli_starts_without_inference_packages(monkeypatch, tmp_path, capsys):
    """Start the CLI using monkeypatch, tmp_path and capsys without GPU imports."""
    original_import = builtins.__import__
    attempted = []

    def guarded_import(name, *args, **kwargs):
        """Reject GPU imports by name; forward remaining args and kwargs."""
        if name.split(".")[0] in {"vllm", "torch", "transformers"}:
            attempted.append(name)
            raise AssertionError(f"Unexpected inference import: {name}")
        return original_import(name, *args, **kwargs)

    write_env(tmp_path / ".env")
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(builtins, "input", lambda prompt: "退出")
    monkeypatch.setattr(sys, "argv", ["shopharness"])
    settings = Settings(
        db_path = str(tmp_path / "shop.db"),
        trace_dir = str(tmp_path / "traces"),
        skills_dir = str(Path(__file__).resolve().parents[1] / "skills"),
    )
    monkeypatch.setattr(cli, "Settings", lambda: settings)
    assert cli.main() == 0
    output = capsys.readouterr().out
    assert "云端 API 模式" in output
    assert "test-secret" not in output
    assert attempted == []
