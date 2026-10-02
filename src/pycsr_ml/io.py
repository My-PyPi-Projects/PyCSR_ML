"""Input discovery and resilient CSV/text loading."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional, Tuple, Union

import pandas as pd

SUPPORTED_EXTENSIONS = {".csv", ".txt"}


def _detect_encoding(path: Path) -> str:
    """Choose a practical encoding without adding a heavyweight dependency."""
    sample = path.read_bytes()[:1_048_576]
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            sample.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            continue
    return "latin-1"


def _sniff_delimiter(sample: str) -> Optional[str]:
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t;|").delimiter
    except csv.Error:
        return None


def load_dataset(input_path: Union[str, Path]) -> Tuple[pd.DataFrame, dict]:
    """Load a .csv or .txt input and return the frame plus ingestion metadata."""
    path = Path(input_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Input path is not a file: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError("Unsupported input type. Currently supported: .csv and .txt")
    if path.stat().st_size == 0:
        raise ValueError("Input file is empty.")

    encoding = _detect_encoding(path)
    with path.open("r", encoding=encoding, errors="replace") as stream:
        sample = stream.read(65536)
    delimiter = _sniff_delimiter(sample)
    load_mode = "delimited"

    if path.suffix.lower() == ".csv":
        delimiter = delimiter or ","
        frame = pd.read_csv(
            path, sep=delimiter, encoding=encoding, low_memory=False, on_bad_lines="skip"
        )
    elif delimiter:
        frame = pd.read_csv(
            path, sep=delimiter, encoding=encoding, low_memory=False, on_bad_lines="skip"
        )
    else:
        lines = path.read_text(encoding=encoding, errors="replace").splitlines()
        frame = pd.DataFrame({"text": [line for line in lines if line.strip()]})
        load_mode = "unstructured text (one row per non-empty line)"

    if frame.empty and len(frame.columns) == 0:
        raise ValueError("No data or columns could be read from the input file.")
    frame.columns = _unique_column_names(frame.columns)
    metadata = {
        "path": str(path),
        "name": path.name,
        "extension": path.suffix.lower(),
        "size_bytes": path.stat().st_size,
        "encoding": encoding,
        "delimiter": repr(delimiter) if delimiter else "n/a",
        "load_mode": load_mode,
        "malformed_row_policy": "skip rows with an invalid field count",
    }
    return frame, metadata


def _unique_column_names(columns) -> list[str]:
    seen: dict[str, int] = {}
    result = []
    for raw in columns:
        base = str(raw).strip() or "unnamed"
        count = seen.get(base, 0)
        result.append(base if count == 0 else f"{base}_{count}")
        seen[base] = count + 1
    return result
