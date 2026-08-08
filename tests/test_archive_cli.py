"""Global context: lock the explicit one-time evidence-refresh CLI boundary."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from training_facts_into_llms import archive_publishing, cli


def test_publish_existing_refresh_flag_defaults_off_and_requires_explicit_use() -> None:
    """Ordinary audit/publication commands retain their unchanged default path."""
    ordinary = cli.build_parser().parse_args(
        ["publish-existing", "--all", "--upload", "on"]
    )
    refresh = cli.build_parser().parse_args(
        [
            "publish-existing",
            "--all",
            "--upload",
            "on",
            "--refresh-evidence",
        ]
    )

    assert ordinary.refresh_evidence is False
    assert refresh.refresh_evidence is True


def test_refresh_evidence_with_upload_off_rejects_before_config_loading(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An invalid refresh cannot read `.env`, stage files, or reach a Hub boundary."""
    monkeypatch.setattr(
        cli,
        "_load_config",
        lambda root: (_ for _ in ()).throw(AssertionError("config was loaded")),
    )

    with pytest.raises(SystemExit) as raised:
        cli.main(
            [
                "publish-existing",
                "--all",
                "--upload",
                "off",
                "--refresh-evidence",
            ]
        )

    assert raised.value.code == 2
    assert "--refresh-evidence requires --upload on" in capsys.readouterr().err


def test_refresh_evidence_dispatch_logs_and_prints_only_sanitized_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The explicit command invokes only the evidence transaction and public receipt."""
    config = SimpleNamespace(log_dir=tmp_path / "logs")
    monkeypatch.setattr(cli, "_load_config", lambda root: config)
    calls: list[Any] = []
    payload = {
        "repo_id": "BurnyCoder/atemokoloporos-qwen3.5-0.8b-study-evidence",
        "repo_type": "dataset",
        "decision": "refresh",
        "previous_revision": "d6223aeac48c87faca586efec21cb48221f2640c",
        "revision": "final-public-sha",
        "changed_paths": [
            "EXPERIMENTS.md",
            "output/pdf/teaching-one-synthetic-fact-qwen35.pdf",
        ],
        "public": True,
        "ungated": True,
        "authenticated_hash_verification": True,
        "anonymous_hash_verification": True,
    }

    class Receipt:
        """Return the same narrow public mapping as EvidenceRefreshReceipt."""

        @staticmethod
        def to_dict() -> dict[str, Any]:
            return payload

    monkeypatch.setattr(
        archive_publishing,
        "refresh_historical_evidence",
        lambda current_config: calls.append(current_config) or Receipt(),
    )
    monkeypatch.setattr(
        archive_publishing,
        "publish_historical_archive",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("normal archive publisher was called")
        ),
    )
    summaries: list[dict[str, Any]] = []
    monkeypatch.setattr(cli, "_print_summary", summaries.append)

    assert (
        cli.main(
            [
                "publish-existing",
                "--all",
                "--upload",
                "on",
                "--refresh-evidence",
            ]
        )
        == 0
    )

    assert calls == [config]
    assert summaries == [payload]
    output = capsys.readouterr().out
    log_files = tuple((tmp_path / "logs").glob("*.jsonl"))
    assert len(log_files) == 1
    events = [
        json.loads(line)
        for line in log_files[0].read_text(encoding="utf-8").splitlines()
    ]
    assert [event["event"] for event in events] == [
        "historical_evidence_refresh_started",
        "historical_evidence_refresh_completed",
    ]
    assert events[-1]["receipt"] == payload
    serialized = json.dumps(events, sort_keys=True)
    assert "historical_evidence_refresh_completed" in output
    assert "staging_directory" not in serialized
    assert "HF_TOKEN" not in serialized
