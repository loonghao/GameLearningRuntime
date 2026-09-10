import pytest

from scripts.publish_skills import validate_receipt


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
