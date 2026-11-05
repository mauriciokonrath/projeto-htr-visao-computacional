"""Baixa uma amostra do iForal fixada em um commit do repositório público."""

import argparse
import hashlib
import json
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath

COMMIT = "b09f7c75c97dfabe940d751c8ef88886a6a94341"
REPO = "Arch-W/iForal-Dataset"


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "htrcv"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def blob_hash(content):
    return hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/iforal"))
    parser.add_argument(
        "--documents", type=int, default=10, help="Primeira página de N documentos (padrão: 10)."
    )
    args = parser.parse_args()
    if args.documents < 1:
        parser.error("--documents deve ser positivo.")
    tree = json.loads(fetch(f"https://api.github.com/repos/{REPO}/git/trees/{COMMIT}?recursive=1"))
    if tree.get("truncated"):
        raise ValueError("Árvore incompleta; download interrompido.")
    blobs = {item["path"]: item for item in tree["tree"] if item["type"] == "blob"}
    xml_by_document = {}
    for path in sorted(blobs):
        parts = PurePosixPath(path).parts
        if len(parts) == 3 and parts[0] == "data" and path.endswith(".xml"):
            xml_by_document.setdefault(parts[1], path)
    chosen = list(xml_by_document.items())[: args.documents]
    records = []
    for doc, xml in chosen:
        image = str(PurePosixPath(xml).with_suffix(".jpg"))
        if image not in blobs:
            raise ValueError(f"Imagem esperada ausente: {image}; revisar a estrutura de origem.")
        for path in (xml, image):
            target = args.output / Path(*PurePosixPath(path).parts[1:])
            target.parent.mkdir(parents=True, exist_ok=True)
            payload = (
                target.read_bytes()
                if target.exists()
                else fetch(
                    f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{urllib.parse.quote(path)}"
                )
            )
            if blob_hash(payload) != blobs[path]["sha"]:
                raise ValueError(f"Hash divergente: {path}; arquivo local preservado.")
            if not target.exists():
                target.write_bytes(payload)
            records.append(
                dict(
                    path=str(target.relative_to(args.output)),
                    sha256=hashlib.sha256(payload).hexdigest(),
                    git_blob=blobs[path]["sha"],
                )
            )
        print(doc, flush=True)
    provenance = dict(
        repository=f"https://github.com/{REPO}",
        commit=COMMIT,
        selection="Primeira página de cada documento, em ordem lexicográfica.",
        documents=len(chosen),
        files=records,
    )
    (args.output / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n"
    )


if __name__ == "__main__":
    main()
