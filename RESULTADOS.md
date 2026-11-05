# Resultados

Os valores abaixo vêm de `experiments/resultados/`, e os gráficos correspondentes estão em `figuras/`. CER e WER são micro-médias: soma das edições dividida pela soma dos caracteres (ou palavras) de referência. Normalização do texto: NFC e espaços colapsados, preservando caixa, acentos e pontuação.

## Avaliação final (páginas de teste)

São duas páginas do split de teste do iForal (Abrantes 1217 e Castreição 1217), com 111 linhas anotadas e 6.146 caracteres de referência. O reconhecedor TraPrInq é o mesmo em todas as condições. Cada referência é pareada a no máximo um candidato (IoU de caixas ≥ 0,5). Referências sem par contam como deleção completa e permanecem no denominador.

| Pipeline | CER | WER | Pares | Omissões | Candidatos sem par |
|---|---:|---:|---:|---:|---:|
| E1 Clássico | 98,45% | 100,15% | 6 | 105 | 132 |
| E2 U-Net piloto | 53,71% | 92,81% | 96 | 15 | 194 |
| E3 Híbrido (barreiras de coluna) | 53,71% | 92,81% | 96 | 15 | 194 |
| E0 BLLA nativo (Kraken) | 46,16% | 91,73% | 103 | 8 | 116 |
| E4 Controle (geometria anotada) | 42,21% | 90,42% | 111 | 0 | 0 |

### Por documento

| Documento | Pipeline | CER | WER | Pares / referências |
|---|---|---:|---:|---:|
| Abrantes 1217 | E1 | 98,37% | 100,16% | 6 / 106 |
| Abrantes 1217 | E2 | 52,08% | 92,41% | 94 / 106 |
| Abrantes 1217 | E3 | 52,08% | 92,41% | 94 / 106 |
| Abrantes 1217 | E0 | 45,69% | 91,51% | 98 / 106 |
| Abrantes 1217 | E4 | 41,27% | 89,96% | 106 / 106 |
| Castreição 1217 | E1 | 100,00% | 100,00% | 0 / 5 |
| Castreição 1217 | E2 | 83,65% | 100,00% | 2 / 5 |
| Castreição 1217 | E3 | 83,65% | 100,00% | 2 / 5 |
| Castreição 1217 | E0 | 54,72% | 95,65% | 5 / 5 |
| Castreição 1217 | E4 | 59,43% | 98,55% | 5 / 5 |

### Observações

- **E2 e E3 empatados:** os resultados são idênticos porque o detector clássico de separadores não encontrou barreiras em nenhuma das duas páginas. A etapa híbrida ficou inativa neste teste.
- **E1:** gerou contornos que atravessam as duas colunas, e só 6 das 111 referências foram pareadas. As 105 omissões respondem por 5.832 das 6.051 edições de caracteres.
- **Peso dos documentos:** Abrantes concentra 106 das 111 linhas (5.828 dos 6.146 caracteres), o que faz o agregado refletir principalmente esse documento.
- **WER acima de 100%:** pode acontecer quando as inserções superam o número de palavras da referência.

## Ablação fotométrica (páginas de treino)

São os mesmos 259 recortes de linha (7 documentos, 14.991 caracteres, 3.188 palavras), extraídos com a geometria anotada. O tratamento é aplicado depois da extração, então a geometria não muda. A variante "original" reproduziu exatamente as 259 predições da execução anterior.

- **Correção de fundo:** fechamento morfológico com elemento quadrado de lado ≈ metade da altura do recorte, seguido de `round(255·I / max(B,1))`.
- **CLAHE:** `clipLimit = 2`, grade 8×8.

| Condição | CER micro | WER micro | CER macro | Edições de caracteres |
|---|---:|---:|---:|---:|
| Original | 41,41% | 88,96% | 39,85% | 6.208 |
| CLAHE | 39,14% | 88,33% | 37,95% | 5.867 |
| Correção de fundo | 39,07% | 88,21% | 37,98% | 5.857 |
| Fundo + CLAHE | 39,47% | 88,17% | 38,39% | 5.917 |

| Documento | Linhas | Original | CLAHE | Fundo | Fundo + CLAHE |
|---|---:|---:|---:|---:|---:|
| Castreição 1222 | 64 | 41,79% | 39,73% | 39,60% | 39,73% |
| Condado 1248 | 22 | 34,00% | 34,61% | 33,86% | 35,41% |
| Condado 1491 | 27 | 28,24% | 28,12% | 28,05% | 27,93% |
| Contrasta 1217 | 61 | 52,10% | 46,44% | 46,72% | 46,77% |
| Contrasta 1222 | 43 | 39,02% | 38,24% | 37,61% | 38,56% |
| Contrasta 1395 | 21 | 34,75% | 33,97% | 34,40% | 34,23% |
| Estremoz 1 1258 | 21 | 49,08% | 44,53% | 45,63% | 46,13% |

Mudanças pareadas em relação ao original:

| Condição | Linhas melhoraram | Pioraram | Empates | Δ edições |
|---|---:|---:|---:|---:|
| CLAHE | 143 | 73 | 43 | −341 |
| Correção de fundo | 143 | 70 | 46 | −351 |
| Fundo + CLAHE | 137 | 83 | 39 | −291 |

A correção de fundo melhorou o CER agregado nos sete documentos. A diferença em relação ao CLAHE é de apenas dez edições, e no CER macro a ordem se inverte.

## Ablação geométrica (páginas de treino)

Fatorial 2×2 na segmentação clássica. H é a correção de inclinação por Canny + Hough; M é o fechamento morfológico 1D do perfil horizontal. A avaliação usa as caixas candidatas, antes da poligonização e do reconhecimento, com pareamento um a um por IoU de caixas ≥ 0,5.

| Condição | Pares / 259 | Omissões | Candidatos |
|---|---:|---:|---:|
| H1 M1 (ambos ativos) | 236 | 23 | 534 |
| H0 M1 (sem Hough) | 165 | 94 | 483 |
| H1 M0 (sem fechamento) | 6 | 253 | 404 |
| H0 M0 (sem ambos) | 13 | 246 | 459 |

O fechamento morfológico é a etapa decisiva. A correção de inclinação só traz ganho quando o fechamento está ativo.

## U-Net piloto

A U-Net piloto foi treinada em uma ROI de uma única página (Condado 1491, 27 linhas, 600 passos). As métricas de treino estão em `models/unet_piloto/metrics.json`.
