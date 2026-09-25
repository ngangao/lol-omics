import subprocess
from unittest.mock import MagicMock, patch

import pytest

from lol.fetch import FetchError, fetch_series_matrix


def test_fetch_series_matrix_calls_geofetch_with_correct_flags(tmp_path):
    """Regression test for two bugs found against a real geofetch install
    (never exercised before, since every prior dataset was downloaded by
    hand): (1) an earlier version called `geofetch -i <acc> -p <dest>`,
    which geofetch's argument parser rejects, since -p/--processed is a
    boolean flag with no argument of its own, and -m (metadata folder) is
    required but was never passed. (2) --processed alone defaults to
    sample-level processed files only; GSE161731 registers its series
    matrix at the series level, so nothing downloaded (geofetch reported
    "No files found" without erroring) until --data-source all was added.
    """
    with patch("lol.fetch.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        fetch_series_matrix("GSE00000", dest=tmp_path)

    called_args = mock_run.call_args[0][0]
    assert called_args[0] == "geofetch"
    assert "-i" in called_args and "GSE00000" in called_args
    assert "--processed" in called_args
    assert "--geo-folder" in called_args
    assert "-m" in called_args
    assert "--data-source" in called_args
    assert called_args[called_args.index("--data-source") + 1] == "all"
    # -p alone (without --processed spelled out) must NOT appear as a
    # bare flag expecting a destination argument right after it -- the
    # exact bug that broke on GSE161731.
    if "-p" in called_args:
        idx = called_args.index("-p")
        assert called_args[idx + 1] != str(tmp_path)


def test_fetch_series_matrix_raises_fetch_error_when_geofetch_missing(tmp_path):
    with (
        patch("lol.fetch.subprocess.run", side_effect=FileNotFoundError),
        pytest.raises(FetchError, match="not installed"),
    ):
        fetch_series_matrix("GSE00000", dest=tmp_path)


def test_fetch_series_matrix_raises_fetch_error_on_geofetch_failure(tmp_path):
    err = subprocess.CalledProcessError(1, "geofetch", stderr="network error")
    with (
        patch("lol.fetch.subprocess.run", side_effect=err),
        pytest.raises(FetchError, match="failed to download"),
    ):
        fetch_series_matrix("GSE00000", dest=tmp_path)
