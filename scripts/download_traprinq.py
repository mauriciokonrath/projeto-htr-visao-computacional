"""Baixa o modelo TraPrInq do Zenodo e confere o checksum."""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

URL = "https://zenodo.org/records/14538271/files/TraPrInq.mlmodel?download=1"
SHA256 = "e4aaad0b5a1956e46c94ddf14c1296ab3fa3422e3296704790ead1c8b3059241"
MD5 = "9e85b400cac29a66644ae50cc0a60e5b"


def digest(path, algorithm):
    value = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("models/htr/TraPrInq.mlmodel"))
    args = parser.parse_args()
    target = args.output
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        temporary = target.with_suffix(".mlmodel.partial")
        request = urllib.request.Request(URL, headers={"User-Agent": "htrcv"})
        with urllib.request.urlopen(request, timeout=120) as source, temporary.open("wb") as output:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                output.write(block)
        if digest(temporary, "sha256") != SHA256 or digest(temporary, "md5") != MD5:
            raise ValueError("Checksum do arquivo baixado divergente.")
        temporary.replace(target)
    if digest(target, "sha256") != SHA256:
        raise ValueError("Modelo local não corresponde ao checksum esperado.")
    target.with_suffix(".provenance.json").write_text(
        json.dumps(
            dict(source=URL, sha256=SHA256, zenodo_md5=MD5, size_bytes=target.stat().st_size),
            indent=2,
        )
        + "\n"
    )
    print(f"Modelo verificado: {target} (SHA-256 {SHA256})")


if __name__ == "__main__":
    main()
