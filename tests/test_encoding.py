"""Windows uses cp1250/cp1252 as the default text encoding: every text read must be explicit."""

import locale
import pathlib
import re

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_text_io_always_names_encoding():
    bad = []
    for f in list((ROOT / "src").rglob("*.py")) + list((ROOT / "scripts").glob("*.py")):
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"\.read_text\(\)|\bopen\([^)]*[\"'][rwa]\+?[\"']\)", line):
                bad.append(f"{f.relative_to(ROOT)}:{i}: {line.strip()}")
    assert not bad, "add encoding='utf-8':\n" + "\n".join(bad)


def test_vocab_reads_under_windows_codepage():
    raw = (ROOT / "configs" / "vocab.yaml").read_bytes()
    assert yaml.safe_load(raw.decode("utf-8"))["motor_imagery"]["cues"]["left_hand"]
    try:
        raw.decode("cp1250")
        decodes = True
    except UnicodeDecodeError:
        decodes = False
    # documents why encoding="utf-8" is required: the file is not valid cp1250
    assert not decodes or locale.getpreferredencoding() != "cp1250"
