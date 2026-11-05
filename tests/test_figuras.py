import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts/figuras.py"


def test_tex_figures_are_generated(tmp_path):
    subprocess.run(
        [sys.executable, str(SCRIPT), "--sem-pdf", "--output", str(tmp_path)], check=True
    )
    names = {p.name for p in tmp_path.iterdir()}
    assert names == {"cer_pipelines.tex", "fotometria_delta_cer.tex", "geometria_pares.tex"}
    tex = (tmp_path / "geometria_pares.tex").read_text(encoding="utf-8")
    assert tex.startswith("\\begin{tikzpicture}") and "(0,236.0000)" in tex
