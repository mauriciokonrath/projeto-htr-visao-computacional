"""Leitura de anotações PAGE XML."""

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class Line:
    id: str
    region_id: str
    polygon: tuple[tuple[int, int], ...]
    baseline: tuple[tuple[int, int], ...]
    transcription: str | None

    @property
    def bbox(self):
        if not self.polygon:
            raise ValueError(f"Linha {self.id} sem polígono; baseline não é uma máscara exata.")
        x, y = zip(*self.polygon)
        return min(x), min(y), max(x) + 1, max(y) + 1


@dataclass(frozen=True)
class Page:
    image_filename: str
    width: int
    height: int
    region_ids: tuple[str, ...]
    lines: tuple[Line, ...]
    reading_order: str


def _points(element, width, height):
    if element is None:
        return ()
    points = tuple(tuple(map(int, pair.split(","))) for pair in element.get("points", "").split())
    if any(len(p) != 2 or not (0 <= p[0] <= width and 0 <= p[1] <= height) for p in points):
        raise ValueError("Coordenada PAGE fora da página ou malformada.")
    return points


def _custom_index(element):
    match = re.search(r"readingOrder\s*\{\s*index\s*:\s*(\d+)", element.get("custom", ""))
    return int(match.group(1)) if match else None


def parse_pagexml(path: str | Path) -> Page:
    # ElementTree não resolve entidades externas.
    root = ET.parse(path).getroot()
    page = root.find(".//{*}Page")
    if page is None:
        raise ValueError(f"{path}: elemento Page ausente.")
    width, height = int(page.attrib["imageWidth"]), int(page.attrib["imageHeight"])
    if min(width, height) <= 0:
        raise ValueError("Dimensões inválidas.")
    regions = list(page.findall(".//{*}TextRegion"))
    refs = page.findall("./{*}ReadingOrder/{*}OrderedGroup/{*}RegionRefIndexed")
    order_source = "xml_document_order_requires_review"
    if refs:
        ranks = {r.attrib["regionRef"]: int(r.attrib["index"]) for r in refs}
        if set(ranks) != {r.attrib["id"] for r in regions}:
            raise ValueError("ReadingOrder parcial ou aninhada: revisar antes da importação.")
        regions.sort(key=lambda r: ranks[r.attrib["id"]])
        order_source = "page_region_order_line_xml_requires_review"
    elif regions and all(_custom_index(r) is not None for r in regions):
        regions.sort(key=_custom_index)
        order_source = "custom_region_order_line_xml_requires_review"
    lines, ids = [], set()
    region_ids = [r.attrib["id"] for r in regions]
    if len(set(region_ids)) != len(region_ids):
        raise ValueError("TextRegion id duplicado.")
    for region in regions:
        elements = list(region.findall("./{*}TextLine"))
        if elements and all(_custom_index(e) is not None for e in elements):
            elements.sort(key=_custom_index)
        for element in elements:
            line_id = element.attrib["id"]
            if line_id in ids:
                raise ValueError(f"TextLine id duplicado: {line_id}")
            ids.add(line_id)
            polygon = _points(element.find("./{*}Coords"), width, height)
            baseline = _points(element.find("./{*}Baseline"), width, height)
            if polygon and (
                len(set(polygon)) < 3 or cv2.contourArea(np.array(polygon, np.int32)) == 0
            ):
                raise ValueError(f"Polígono degenerado: {line_id}")
            if not polygon and len(baseline) < 2:
                raise ValueError(f"Linha sem geometria válida: {line_id}")
            equivalents = sorted(
                element.findall("./{*}TextEquiv"), key=lambda e: int(e.get("index", "0"))
            )
            unicode_node = equivalents[0].find("./{*}Unicode") if equivalents else None
            transcription = None if unicode_node is None else "".join(unicode_node.itertext())
            lines.append(Line(line_id, region.attrib["id"], polygon, baseline, transcription))
    return Page(
        page.attrib["imageFilename"], width, height, tuple(region_ids), tuple(lines), order_source
    )


def page_mask(page: Page) -> np.ndarray:
    mask = np.zeros((page.height, page.width), dtype=np.uint8)
    for line in page.lines:
        if not line.polygon:
            raise ValueError(
                f"{line.id}: apenas baseline; aproximação não habilitada nesta versão."
            )
        cv2.fillPoly(mask, [np.array(line.polygon, dtype=np.int32)], 255)
    return mask
