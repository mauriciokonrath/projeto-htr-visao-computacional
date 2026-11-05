# Visão computacional para HTR em documentos históricos

[![CI](https://github.com/mauriciokonrath/projeto-htr-visao-computacional/actions/workflows/ci.yml/badge.svg)](https://github.com/mauriciokonrath/projeto-htr-visao-computacional/actions/workflows/ci.yml)

Trabalho da disciplina de Visão Computacional (Ciência da Computação, UFSC).
Autores: Maurício Konrath, Fernanda Larissa Müller e Isabella Viviani de Aquino.

O projeto compara diferentes formas de localizar linhas de texto em manuscritos históricos. O reconhecedor de texto (HTR) fica fixo, de modo que as diferenças de CER/WER refletem apenas as etapas de visão computacional. Os dados são do [iForal](https://github.com/Arch-W/iForal-Dataset) (forais portugueses) e o reconhecedor é o [TraPrInq](https://zenodo.org/records/14538271), executado no Kraken 5.3.0.

## Pipelines

| ID | Localização de linhas |
|---|---|
| E1 | Clássica: CLAHE, correção de inclinação (Canny + Hough), binarização adaptativa, morfologia e perfis de projeção por coluna |
| E2 | U-Net pequena treinada para detectar baselines, seguida de extração por componentes conexos |
| E3 | E2 + barreiras de coluna detectadas pelo método clássico |
| E0 | Segmentador neural BLLA nativo do Kraken (referência externa) |
| E4 | Controle: reconhecimento sobre a geometria anotada |

E1, E2 e E3 compartilham o mesmo pós-processamento: filtro de borda da página, guarda de 2 px e poligonização do Kraken. Também foram feitas duas ablações nas páginas de treino. A primeira é fotométrica (CLAHE e correção de fundo nos recortes); a segunda é geométrica (correção de inclinação e fechamento morfológico na pipeline clássica).

Os resultados estão em [RESULTADOS.md](RESULTADOS.md).

## Estrutura

```
src/htrcv/              biblioteca
  classical.py, layout.py      E1: segmentação clássica
  unet_pilot.py, pilot_data.py U-Net (E2): dados, treino e inferência em ladrilhos
  baseline_pilot.py            alvos de baseline e extração a partir do mapa da U-Net
  columns.py                   E3: barreiras de coluna
  native.py                    E0: segmentador BLLA
  postprocessing.py            filtro de borda e guarda
  recognition.py               poligonização e reconhecimento
  kraken_backend.py, oracle.py reconhecedor e controle E4
  photometry.py                CLAHE e correção de fundo
  line_matching.py, metrics.py pareamento por IoU, CER/WER
  final_evaluation.py          avaliação E0–E4
experiments/            scripts das avaliações e resultados agregados
scripts/                download, preparação e treino da U-Net, figuras
figuras/                gráficos dos resultados (PDF e pgfplots)
configs/                parâmetros de cada etapa
data/iforal-prepared/   manifesto, linhas e divisão treino/validação/teste
models/unet_piloto/     pesos e métricas da U-Net treinada
tests/                  testes (pytest)
```

## Instalação

Python 3.11. O ambiente básico, sem PyTorch e sem Kraken, é suficiente para a segmentação clássica, as ablações geométricas e os testes:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Para a U-Net, o BLLA e o reconhecimento de texto, instale o ambiente completo:

```bash
pip install -r requirements-htr.txt
pip install -e .
```

As versões estão fixadas porque mudam os resultados. Com OpenCV 5, por exemplo, a segmentação clássica gera um número diferente de candidatos.

## Dados e modelos

```bash
python scripts/download_iforal.py        # 10 documentos em data/iforal/, commit b09f7c7
python scripts/download_traprinq.py      # models/htr/TraPrInq.mlmodel, com conferência de SHA-256
```

O diretório `data/iforal-prepared/` já contém o manifesto usado nos experimentos. Ele foi gerado com `htrcv prepare --raw data/iforal --output data/iforal-prepared`. A divisão é feita por documento, com semente 42: sete documentos em treino, um em validação e dois em teste.

O checkpoint da U-Net (`models/unet_piloto/final.pt`, 122.673 parâmetros) está incluído. O modelo BLLA vem junto com o Kraken.

## Uso

Segmentação clássica (E1), sem HTR:

```bash
htrcv classical --manifest data/iforal-prepared/pages.csv --split test --output runs/classico
```

Avaliação final E0–E4 nas páginas de teste:

```bash
python experiments/avaliacao_final.py --output runs/avaliacao_final
```

Ablação geométrica:

```bash
python experiments/ablacoes.py geometria --output runs/ablacoes/geometria
```

Ablação fotométrica. A primeira etapa gera o controle E4 e os recortes de linha do treino:

```bash
htrcv oracle --manifest data/iforal-prepared/pages.csv --split train --output runs/e4_treino
python scripts/run_contrast_ablation.py --baseline runs/e4_treino \
    --manifest data/iforal-prepared/pages.csv --htr-config configs/htr.yaml \
    --preprocessing configs/contrast_ablation.yaml --output runs/contraste
python experiments/ablacoes.py fotometria --lines runs/contraste/control/per_line.csv \
    --output runs/ablacoes/fotometria
```

Treino da U-Net piloto:

```bash
python scripts/prepare_pilot.py --manifest data/iforal-prepared/pages.csv \
    --review configs/pilot_condado1491.json --output runs/piloto
python scripts/prepare_baseline_pilot.py --source runs/piloto \
    --config configs/pilot_baseline_geometry.json --output runs/piloto_baselines
python scripts/train_pilot.py --dataset runs/piloto_baselines \
    --config configs/pilot_baseline_unet.yaml --output runs/unet
```

Os scripts não sobrescrevem diretórios de saída existentes.

## Figuras

```bash
pip install matplotlib
python scripts/figuras.py
```

O script gera, em `figuras/`, um PDF e um `.tex` (pgfplots) para cada gráfico:

- `cer_pipelines`: CER e WER de cada pipeline;
- `fotometria_delta_cer`: variação do CER por documento em relação ao original;
- `geometria_pares`: pares obtidos em cada condição da ablação 2×2.

Para usar no LaTeX, carregue `\usepackage{pgfplots}` com `\pgfplotsset{compat=1.17}` e inclua com `\input{figuras/cer_pipelines.tex}`. A opção `--sem-pdf` gera só os `.tex`, sem precisar do matplotlib.

## Testes

```bash
pytest
ruff check . && ruff format --check .
```

Os testes rodam no ambiente mínimo. A avaliação E0–E4 é testada de ponta a ponta com modelos simulados (`tests/test_final_evaluation_pipeline.py`). O GitHub Actions executa lint e testes a cada push (`.github/workflows/ci.yml`).

## Limitações

- **Anotações parciais:** as métricas valem apenas para as linhas anotadas. Candidatos sem referência correspondente não são classificados como erro ou acerto.
- **Teste pequeno:** são duas páginas, e uma delas tem só cinco linhas anotadas.
- **U-Net piloto:** foi treinada em uma única página.
- **Pré-treino do reconhecedor:** não se sabe se o TraPrInq viu documentos do iForal no pré-treino.
