"""Gera as figuras do relatório a partir de experiments/resultados.

Cada figura é salva em PDF (matplotlib) e em .tex (pgfplots) para inclusão
direta no LaTeX com \\input{figuras/<nome>.tex}. O .tex requer:

    \\usepackage{pgfplots}
    \\pgfplotsset{compat=1.17}

Uso: python scripts/figuras.py [--output figuras]
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments/resultados"

PIPELINES = [
    ("E1", "E1 Clássico"),
    ("E2", "E2 U-Net"),
    ("E3", "E3 Híbrido"),
    ("E0_native", "E0 BLLA"),
    ("E4_control", "E4 Controle"),
]
TREATMENTS = [("clahe", "CLAHE"), ("background", "Fundo"), ("background_clahe", "Fundo + CLAHE")]
GEOMETRIC = [
    ("H1_M1", "H1 M1"),
    ("H0_M1", "H0 M1"),
    ("H1_M0", "H1 M0"),
    ("H0_M0", "H0 M0"),
]
COLORS = ["1f4e79", "c55a11", "548235"]


def load(path):
    return json.loads((RESULTS / path).read_text())


ACCENTS = {"castreicao": "Castreição"}


def short_document(document_id):
    *name, year = document_id.removeprefix("foral_de_").split("_")
    words = [ACCENTS.get(w, w.title()) for w in name]
    return " ".join(words + [year])


def comma(value, digits=1):
    return f"{value:.{digits}f}".replace(".", ",")


# ---------------------------------------------------------------- dados


def pipeline_data():
    summary = load("avaliacao_final/summary.json")
    cer = [100 * summary[key]["metrics"]["cer"] for key, _ in PIPELINES]
    wer = [100 * summary[key]["metrics"]["wer"] for key, _ in PIPELINES]
    return [label for _, label in PIPELINES], {"CER": cer, "WER": wer}


def photometric_data():
    """Variação de CER (pontos percentuais) em relação ao original, por documento."""
    results = load("ablacoes/fotometria.json")
    documents = list(results["original"]["documents"])
    labels = [short_document(d) for d in documents] + ["Agregado"]
    series = {}
    for key, name in TREATMENTS:
        deltas = [
            100 * (results[key]["documents"][d]["cer"] - results["original"]["documents"][d]["cer"])
            for d in documents
        ]
        deltas.append(100 * (results[key]["micro"]["cer"] - results["original"]["micro"]["cer"]))
        series[name] = deltas
    return labels, series


def geometric_data():
    summary = load("ablacoes/geometria.json")
    total = summary["H1_M1"]["reference_count"]
    return (
        [label for _, label in GEOMETRIC],
        {"Pares": [summary[key]["matched"] for key, _ in GEOMETRIC]},
        total,
    )


# ---------------------------------------------------------------- pgfplots


def pgf_bars(labels, series, ylabel, *, ymin=None, ymax=None, digits=1):
    n = len(labels)
    axis = [
        "ybar",
        f"bar width={max(4, 18 // len(series))}pt",
        "width=\\linewidth",
        "height=6cm",
        f"xtick={{0,...,{n - 1}}}",
        "xticklabels={" + ",".join("{" + label + "}" for label in labels) + "}",
        f"ylabel={{{ylabel}}}",
        "/pgf/number format/use comma",
        f"/pgf/number format/precision={digits}",
        "/pgf/number format/fixed",
        "ymajorgrids",
        "grid style={gray!25}",
        "enlarge x limits=" + ("0.15" if n < 5 else "0.06"),
        "tick label style={font=\\footnotesize}",
        "label style={font=\\small}",
    ]
    if max(len(label) for label in labels) > 8:
        axis.append("x tick label style={rotate=30, anchor=east}")
    if any(v < 0 for values in series.values() for v in values):
        axis.append(
            "extra y ticks={0}, extra y tick labels={}, extra y tick style={grid=major, grid style={black!60}}"
        )
    if ymin is not None:
        axis.append(f"ymin={ymin}")
    if ymax is not None:
        axis.append(f"ymax={ymax}")
    if len(series) > 1:
        axis.append(
            "legend style={at={(0.5,1.02)}, anchor=south, legend columns=-1, "
            "draw=none, font=\\small, "
            "/tikz/every even column/.append style={column sep=8pt}}"
        )
    else:
        axis += ["nodes near coords", "every node near coord/.append style={font=\\scriptsize}"]
    lines = ["\\begin{tikzpicture}", "\\begin{axis}[", "  " + ",\n  ".join(axis), "]"]
    for color, (name, values) in zip(COLORS, series.items()):
        coords = " ".join(f"({i},{value:.4f})" for i, value in enumerate(values))
        rgb = ";".join(
            f"{c},{int(color[k : k + 2], 16)}" for c, k in zip(("red", "green", "blue"), (0, 2, 4))
        )
        lines.append(f"\\addplot[fill={{rgb,255:{rgb}}}, draw=none] coordinates {{{coords}}};")
    if len(series) > 1:
        lines.append("\\legend{" + ",".join(series) + "}")
    lines += ["\\end{axis}", "\\end{tikzpicture}"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- matplotlib


def mpl_bars(path, labels, series, ylabel, *, ymin=None, ymax=None, annotate=False):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    x = np.arange(len(labels))
    width = 0.8 / len(series)
    for i, (color, (name, values)) in enumerate(zip(COLORS, series.items())):
        offset = (i - (len(series) - 1) / 2) * width
        bars = ax.bar(x + offset, values, width, label=name, color="#" + color)
        if annotate:
            for bar, value in zip(bars, values):
                ax.annotate(
                    comma(value, 0) if float(value).is_integer() else comma(value),
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    xytext=(0, 2),
                    textcoords="offset points",
                )
    ax.set_xticks(x, labels, rotation=30 if max(map(len, labels)) > 8 else 0, ha="right")
    if max(map(len, labels)) <= 8:
        plt.setp(ax.get_xticklabels(), ha="center")
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(lambda v, _: comma(v, 0) if float(v).is_integer() else comma(v))
    ax.grid(axis="y", color="0.9")
    ax.set_axisbelow(True)
    if ymin is not None or ymax is not None:
        ax.set_ylim(ymin, ymax)
    if any(v < 0 for values in series.values() for v in values):
        ax.axhline(0, color="0.3", linewidth=0.8)
    if len(series) > 1:
        ax.legend(frameon=False, ncol=len(series), loc="lower center", bbox_to_anchor=(0.5, 1))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------- main


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=ROOT / "figuras")
    parser.add_argument("--sem-pdf", action="store_true", help="Gerar apenas os .tex")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    labels, series, total = geometric_data()
    figures = {
        "cer_pipelines": (*pipeline_data(), "Taxa de erro (\\%)", dict(ymin=0, ymax=110)),
        "fotometria_delta_cer": (
            *photometric_data(),
            "$\\Delta$ CER (p.p.)",
            dict(),
        ),
        "geometria_pares": (labels, series, f"Pares (de {total})", dict(ymin=0, ymax=total)),
    }
    for name, (labels, series, ylabel, limits) in figures.items():
        tex = pgf_bars(
            labels, series, ylabel, **limits, digits=0 if name == "geometria_pares" else 1
        )
        (args.output / f"{name}.tex").write_text(tex, encoding="utf-8")
        if not args.sem_pdf:
            mpl_label = ylabel.replace("\\%", "%").replace("$\\Delta$", "Δ")
            mpl_bars(
                args.output / f"{name}.pdf",
                labels,
                series,
                mpl_label,
                annotate=len(series) == 1,
                **limits,
            )
        print(args.output / name)


if __name__ == "__main__":
    main()
