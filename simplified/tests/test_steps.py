from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pytest
from nooa.unifiedllm import FakeLLMClient

from simplified.agents import (
    REQUIRED_RANGES,
    CandidateCritic,
    ChangeProposer,
    HotspotAnalyzer,
    HotspotCritic,
    HelloAgent,
    InputContractCritic,
    InputCritic,
    InputProposer,
    InstrumentationProposer,
    ReportBuilder,
    Router,
    TraceCritic,
)
from simplified.cli import (
    LocalSourceEnvironment,
    LocalTrainingEnvironment,
    _load,
    create_llm,
    main,
)
from simplified.types import (
    BenchmarkResult,
    CandidateResult,
    ChangeProposal,
    Critique,
    Hotspot,
    HotspotAnalysis,
    InstrumentationPlan,
    Route,
    TraceResult,
    TrainingRequest,
    TrainingSpec,
)


def _write(path: Path, text: str) -> None:
    path.write_text(text)


def _git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _write(repo / "README.md", "Run python bench.py to benchmark training.\n")
    _write(
        repo / "bench.py",
        'import json\nprint(json.dumps({"step_time_ms": 10.0, "correctness_value": 1.0}))\n',
    )
    _write(
        repo / "profile.py",
        "from pathlib import Path\n"
        "import json\n"
        'Path("trace.json").write_text("trace")\n'
        "print(json.dumps({"
        '"completed": True, "path": "trace.json", '
        f'"ranges": {REQUIRED_RANGES!r}, "summary": "forward took 80 percent"'
        "}))\n",
    )
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True
    )
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "fixture"], check=True)
    return repo


def _spec() -> TrainingSpec:
    return TrainingSpec(
        smoke_command=(sys.executable, "-c", "print('ok')"),
        benchmark_command=(sys.executable, "bench.py"),
        profile_command=(sys.executable, "profile.py"),
        correctness_tolerance=0.0,
    )


def test_load_training_request_accepts_plain_text_and_json(tmp_path: Path) -> None:
    plain_path = tmp_path / "request.txt"
    json_path = tmp_path / "request.json"
    description = (
        "Train GeoTransolver volume using the unified external aero recipe "
        "and the DrivAerML dataset."
    )
    _write(plain_path, f"  {description}\n")
    _write(
        json_path,
        TrainingRequest(description=description).model_dump_json(),
    )

    assert _load(str(plain_path), TrainingRequest) == TrainingRequest(
        description=description
    )
    assert _load(str(json_path), TrainingRequest) == TrainingRequest(
        description=description
    )


def test_create_llm_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}
    fake = FakeLLMClient()

    def factory(model, **options):
        captured.update(model=model, **options)
        return fake

    monkeypatch.setenv("SIMPLIFIED_MODEL", "openai/nvidia/example")
    monkeypatch.setenv("SIMPLIFIED_API_TOKEN", "secret")
    monkeypatch.setenv("SIMPLIFIED_API_BASE", "https://example.test/v1")
    monkeypatch.setenv("SIMPLIFIED_CLIENT_TYPE", "responses")
    monkeypatch.setattr("simplified.cli.get_llm_client", factory)

    assert create_llm() is fake
    assert captured == {
        "model": "openai/nvidia/example",
        "client_type": "responses",
        "api_key": "secret",
        "api_base": "https://example.test/v1",
    }


def test_create_llm_explicit_client_type_overrides_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def factory(model, **options):
        captured.update(model=model, **options)
        return FakeLLMClient()

    monkeypatch.setenv("SIMPLIFIED_CLIENT_TYPE", "completion")
    monkeypatch.setattr("simplified.cli.get_llm_client", factory)

    create_llm("openai/switchyard/openai/gpt-4o-mini", client_type="responses")

    assert captured["client_type"] == "responses"


def test_create_llm_rejects_invalid_environment_client_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SIMPLIFIED_CLIENT_TYPE", "invalid")

    with pytest.raises(ValueError, match="completion.*responses"):
        create_llm("openai/example")


def test_local_source_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _git_repo(tmp_path)
    source = LocalSourceEnvironment(repo)

    assert "bench.py" in source.list_files("*.py")
    assert any("bench.py" in match for match in source.search("step_time_ms"))
    assert "step_time_ms" in source.read_file("bench.py")

    monkeypatch.setattr("simplified.cli.shutil.which", lambda _name: None)
    assert "bench.py" in source.list_files("*.py")
    assert any("bench.py" in match for match in source.search("step_time_ms"))

    with pytest.raises(ValueError):
        source.read_file("../outside")


def test_input_critic_uses_bounded_evidence_and_one_predict_call(
    tmp_path: Path,
) -> None:
    repo = _git_repo(tmp_path)
    llm = FakeLLMClient.simple_message(
        '{"accepted":true,"feedback":"","requires_human":false}'
    )
    critic = InputCritic(llm=llm, source=LocalSourceEnvironment(repo))
    spec = _spec()

    evidence = critic._repository_evidence(spec)
    result = asyncio.run(critic.review(spec))

    assert len(evidence) <= 20_000
    assert "bench.py" in evidence
    assert result.accepted


def test_input_critic_uses_text_output_for_kimi(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _git_repo(tmp_path)
    llm = FakeLLMClient.simple_message(
        '{"accepted":true,"feedback":"","requires_human":false}'
    )
    llm.model = "openai/nvidia/moonshotai/kimi-k3-max-preview"
    output_models = []
    original_acall = llm.acall

    async def recording_acall(messages, tools=None, output_model=None, **kwargs):
        output_models.append(output_model)
        return await original_acall(
            messages, tools=tools, output_model=output_model, **kwargs
        )

    monkeypatch.delenv("SIMPLIFIED_STRUCTURED_OUTPUT", raising=False)
    monkeypatch.setattr(llm, "acall", recording_acall)
    critic = InputCritic(llm=llm, source=LocalSourceEnvironment(repo))

    result = asyncio.run(critic.review(_spec()))

    assert result.accepted
    assert output_models == [None]


def test_input_critic_falls_back_when_endpoint_rejects_json_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _git_repo(tmp_path)
    llm = FakeLLMClient.simple_message(
        '{"accepted":true,"feedback":"","requires_human":false}'
    )
    llm.model = "openai/custom/text-only-model"
    output_models = []
    original_acall = llm.acall

    async def rejecting_acall(messages, tools=None, output_model=None, **kwargs):
        output_models.append(output_model)
        if output_model is not None:
            raise RuntimeError(
                'unsupported response_format.type json_schema: only "text" is supported'
            )
        return await original_acall(
            messages, tools=tools, output_model=output_model, **kwargs
        )

    monkeypatch.delenv("SIMPLIFIED_STRUCTURED_OUTPUT", raising=False)
    monkeypatch.setattr(llm, "acall", rejecting_acall)
    critic = InputCritic(llm=llm, source=LocalSourceEnvironment(repo))

    result = asyncio.run(critic.review(_spec()))

    assert result.accepted
    assert output_models[0] is not None
    assert output_models[1] is None


def test_input_critic_evidence_reports_missing_command_paths(
    tmp_path: Path,
) -> None:
    repo = _git_repo(tmp_path)
    critic = InputCritic(
        llm=FakeLLMClient(),
        source=LocalSourceEnvironment(repo),
    )
    spec = _spec().model_copy(
        update={"smoke_command": (sys.executable, "missing-train.py")}
    )

    evidence = critic._repository_evidence(spec)

    assert "missing referenced paths:" in evidence
    assert "missing-train.py" in evidence


def test_real_training_commands_and_isolated_candidate(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path)
    environment = LocalTrainingEnvironment(
        repo, artifact_directory=tmp_path / "artifacts"
    )
    spec = _spec()

    assert environment.smoke(spec).completed
    baseline = environment.benchmark(spec)
    trace = environment.profile(
        spec,
        InstrumentationPlan(patch="", ranges=REQUIRED_RANGES),
    )
    proposal = ChangeProposal(
        patch=(
            "diff --git a/bench.py b/bench.py\n"
            "--- a/bench.py\n"
            "+++ b/bench.py\n"
            "@@ -1,2 +1,2 @@\n"
            " import json\n"
            '-print(json.dumps({"step_time_ms": 10.0, "correctness_value": 1.0}))\n'
            '+print(json.dumps({"step_time_ms": 5.0, "correctness_value": 1.0}))\n'
        ),
        rationale="Use the faster implementation.",
    )
    candidate = environment.benchmark_candidate(spec, proposal)

    assert baseline.step_time_ms == 10.0
    assert trace.completed and Path(trace.path).read_text() == "trace"
    assert candidate.benchmark and candidate.benchmark.step_time_ms == 5.0
    assert "10.0" in (repo / "bench.py").read_text()


def test_deterministic_agents_and_step_cli(tmp_path: Path) -> None:
    llm = FakeLLMClient()
    spec = _spec()
    trace = TraceResult(
        completed=True,
        path="trace.json",
        ranges=REQUIRED_RANGES,
        summary="forward took 80 percent",
    )
    hotspot = Hotspot(kind="model_forward_backward", evidence="forward took 80 percent")
    baseline = BenchmarkResult(step_time_ms=10, correctness_value=1)
    candidate = CandidateResult(
        completed=True,
        benchmark=BenchmarkResult(step_time_ms=5, correctness_value=1),
    )
    proposal = ChangeProposal(patch="diff", rationale="Faster operation")

    assert InputContractCritic(llm=llm).review(spec).accepted
    assert TraceCritic(llm=llm).review(trace, REQUIRED_RANGES).accepted
    route = Router(llm=llm).route(hotspot)
    assert route.skill == "physicsnemo-functionals-integrator"
    assert CandidateCritic(llm=llm).review(spec, baseline, candidate).accepted
    assert (
        ReportBuilder(llm=llm)
        .build(baseline, candidate.benchmark, hotspot, proposal)
        .speedup
        == 2
    )

    hotspot_path = tmp_path / "hotspot.json"
    output_path = tmp_path / "route.json"
    hotspot_path.write_text(hotspot.model_dump_json())
    main(
        [
            "route-hotspot",
            str(hotspot_path),
            "--no-trace",
            "-o",
            str(output_path),
        ]
    )
    assert Route.model_validate_json(output_path.read_text()) == route


def _real_model():
    model = os.getenv("SIMPLIFIED_TEST_MODEL") or os.getenv("SIMPLIFIED_MODEL")
    if not model:
        pytest.skip(
            "set SIMPLIFIED_TEST_MODEL or SIMPLIFIED_MODEL to run real-LLM functional tests"
        )
    return create_llm(model)


def _run_llm(call):
    async def run():
        llm = _real_model()
        try:
            return await call(llm)
        finally:
            await llm.aclose()

    return asyncio.run(run())


@pytest.mark.llm
def test_llm_hello() -> None:
    async def call(llm):
        return await HelloAgent(llm=llm).hello("PhysicsNeMo")

    result = _run_llm(call)
    assert result.message
    assert "physicsnemo" in result.message.lower()


@pytest.mark.llm
def test_llm_propose_inputs(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path)

    async def call(llm):
        return await InputProposer(
            llm=llm, source=LocalSourceEnvironment(repo)
        ).propose(
            TrainingRequest(
                description="Benchmark bench.py and preserve its correctness value."
            ),
            None,
        )

    assert isinstance(_run_llm(call), TrainingSpec)


@pytest.mark.llm
def test_llm_review_inputs(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path)

    async def call(llm):
        return await InputCritic(
            llm=llm,
            source=LocalSourceEnvironment(repo),
        ).review(_spec())

    assert isinstance(_run_llm(call), Critique)


@pytest.mark.llm
def test_llm_propose_instrumentation(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path)

    async def call(llm):
        return await InstrumentationProposer(
            llm=llm, source=LocalSourceEnvironment(repo)
        ).propose(_spec(), REQUIRED_RANGES, None)

    assert isinstance(_run_llm(call), InstrumentationPlan)


@pytest.mark.llm
def test_llm_analyze_hotspots() -> None:
    trace = TraceResult(
        completed=True,
        path="real-trace.json",
        ranges=REQUIRED_RANGES,
        summary="The forward range consumes 80% of each training step.",
    )

    async def call(llm):
        return await HotspotAnalyzer(llm=llm).analyze(trace, None)

    result = _run_llm(call)
    assert isinstance(result, HotspotAnalysis) and result.hotspots


@pytest.mark.llm
def test_llm_review_hotspots() -> None:
    trace = TraceResult(
        completed=True,
        path="real-trace.json",
        ranges=REQUIRED_RANGES,
        summary="The forward range consumes 80% of each training step.",
    )
    analysis = HotspotAnalysis(
        hotspots=(
            Hotspot(
                kind="model_forward_backward",
                evidence="The forward range consumes 80% of each training step.",
            ),
        )
    )

    async def call(llm):
        return await HotspotCritic(llm=llm).review(trace, analysis)

    assert isinstance(_run_llm(call), Critique)


@pytest.mark.llm
def test_llm_propose_change(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path)
    hotspot = Hotspot(kind="model_forward_backward", evidence="bench.py takes 10 ms")

    async def call(llm):
        return await ChangeProposer(
            llm=llm, source=LocalSourceEnvironment(repo)
        ).propose(
            _spec(),
            hotspot,
            Router(llm=llm).route(hotspot),
            None,
        )

    assert isinstance(_run_llm(call), ChangeProposal)


def test_training_commands_run_from_spec_working_directory(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path)
    recipe = repo / "examples" / "recipe"
    recipe.mkdir(parents=True)
    _write(
        recipe / "train.py",
        "import json, sys\n"
        "if sys.argv[1] == 'benchmark':\n"
        "    print(json.dumps({"
        "'step_time_ms': 7.0, 'correctness_value': 1.0"
        "}))\n"
        "else:\n"
        "    print('ok')\n",
    )
    spec = TrainingSpec(
        working_directory="examples/recipe",
        smoke_command=(sys.executable, "train.py", "smoke"),
        benchmark_command=(sys.executable, "train.py", "benchmark"),
        profile_command=(sys.executable, "train.py", "profile"),
        correctness_tolerance=0.0,
    )
    environment = LocalTrainingEnvironment(repo)

    assert environment.preflight(spec).completed
    assert environment.smoke(spec).completed
    assert environment.benchmark(spec).step_time_ms == 7.0


def test_command_preflight_rejects_wrong_or_escaping_working_directory(
    tmp_path: Path,
) -> None:
    repo = _git_repo(tmp_path)
    environment = LocalTrainingEnvironment(repo)
    wrong_script = _spec().model_copy(
        update={
            "smoke_command": (sys.executable, "missing/train.py"),
            "benchmark_command": (sys.executable, "bench.py"),
            "profile_command": (sys.executable, "profile.py"),
        }
    )
    escaping = _spec().model_copy(update={"working_directory": ".."})

    wrong_result = environment.preflight(wrong_script)
    escaping_result = environment.preflight(escaping)

    assert not wrong_result.completed
    assert "script does not exist relative to working_directory" in wrong_result.error
    assert not escaping_result.completed
    assert "escapes the repository" in escaping_result.error


def test_single_string_command_supports_leading_environment_assignment(
    tmp_path: Path,
) -> None:
    repo = _git_repo(tmp_path)
    recipe = repo / "examples" / "recipe"
    source = recipe / "src"
    source.mkdir(parents=True)
    _write(
        source / "train.py",
        "import json, os, sys\n"
        "assert os.environ['PYTHONPATH'] == sys.argv[2]\n"
        "if sys.argv[1] == 'benchmark':\n"
        "    print(json.dumps({"
        "'step_time_ms': 4.0, 'correctness_value': 1.0"
        "}))\n"
        "else:\n"
        "    print('ok')\n",
    )
    prefix = f"PYTHONPATH={repo} {sys.executable} src/train.py"
    spec = TrainingSpec(
        working_directory="examples/recipe",
        smoke_command=(f"{prefix} smoke {repo}",),
        benchmark_command=(f"{prefix} benchmark {repo}",),
        profile_command=(f"{prefix} profile {repo}",),
        correctness_tolerance=0.0,
    )
    environment = LocalTrainingEnvironment(repo)

    assert environment.preflight(spec).completed
    assert environment.smoke(spec).completed
    assert environment.benchmark(spec).step_time_ms == 4.0


def test_single_string_command_rejects_shell_operators(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path)
    environment = LocalTrainingEnvironment(repo)
    spec = _spec().model_copy(
        update={
            "smoke_command": (
                f"{sys.executable} -c \"print(1)\" && echo unsafe",
            )
        }
    )

    result = environment.preflight(spec)

    assert not result.completed
    assert "shell operators are not supported" in result.error


def test_command_preflight_rejects_shell_executable(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path)
    environment = LocalTrainingEnvironment(repo)
    spec = _spec().model_copy(
        update={
            "smoke_command": (
                "/bin/bash",
                "-c",
                "sed -i s/old/new/ README.md && python bench.py",
            )
        }
    )

    result = environment.preflight(spec)

    assert not result.completed
    assert "shell executables are not supported" in result.error
