import pytest

from scripts.publish_skills import registry_environment, validate_receipt


def test_publisher_passes_an_isolated_config_and_cleans_it(monkeypatch):
    import json
    import os
    from pathlib import Path

    monkeypatch.setenv("CLAWHUB_TOKEN", "test-token")
    monkeypatch.setenv("CLAWHUB_CONFIG_PATH", "existing-user-config.json")
    with registry_environment(True) as env:
        path = Path(env["CLAWHUB_CONFIG_PATH"])
        assert path.name == "config.json"
        assert path.read_text() and json.loads(path.read_text())["token"] == "test-token"
        assert "CLAWHUB_TOKEN" not in env
        assert os.environ["CLAWHUB_CONFIG_PATH"] == "existing-user-config.json"
    assert not path.exists()


def test_publisher_cleans_credentials_after_failure(monkeypatch):
    from pathlib import Path

    monkeypatch.setenv("CLAWHUB_TOKEN", "test-token")
    with pytest.raises(RuntimeError), registry_environment(True) as env:
        path = Path(env["CLAWHUB_CONFIG_PATH"])
        raise RuntimeError("registry failed")
    assert not path.exists()


def receipt(**changes):
    return {
        "ok": True,
        "status": "published",
        "slug": "glr-cli",
        "version": "0.13.2",
        "fingerprint": "a" * 64,
        "fileCount": 2,
        **changes,
    }


@pytest.mark.parametrize(
    "change",
    [
        {"ok": False},
        {"status": "pending-publication"},
        {"version": "0.1.0"},
        {"slug": "wrong"},
        {"fingerprint": "bad"},
        {"fileCount": 0},
    ],
)
def test_reject_invalid_publish_receipt(change):
    with pytest.raises(ValueError):
        validate_receipt(receipt(**change), slug="glr-cli", version="0.13.2", dry_run=False)


def test_valid_receipts():
    validate_receipt(receipt(), slug="glr-cli", version="0.13.2", dry_run=False)
    validate_receipt(
        receipt(status="would-publish"), slug="glr-cli", version="0.13.2", dry_run=True
    )
