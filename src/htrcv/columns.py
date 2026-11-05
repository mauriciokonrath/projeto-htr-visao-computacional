"""E3: barreiras de coluna estimadas na imagem e aplicadas ao mapa da U-Net."""

import numpy as np

from .classical import ClassicalConfig, segment


def barrier_mask(shape, transform, cuts):
    """Bloqueia uma faixa de +/-1 pixel ao redor de cada separador, na grade original.

    Como |a|+|b| <= sqrt(2) em uma rotação, nenhum passo 8-conexo atravessa a faixa.
    """
    matrix = np.asarray(transform, dtype=float)
    if matrix.shape != (2, 3) or not np.isfinite(matrix).all():
        raise ValueError("Transformação inválida.")
    if not np.allclose(matrix[:, :2] @ matrix[:, :2].T, np.eye(2), atol=1e-6):
        raise ValueError("A barreira exige rotação rígida, sem escala.")
    if not all(np.isfinite(cut) for cut in cuts):
        raise ValueError("Corte inválido.")
    y, x = np.ogrid[: shape[0], : shape[1]]
    work_x = matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]
    allowed = np.ones(shape, np.uint8)
    for cut in cuts:
        allowed[np.abs(work_x - cut) <= 1.0] = 0
    return allowed


def image_barriers(gray, config):
    result = segment(gray, ClassicalConfig(**config))
    cuts = result["layout"]["gutter_cuts_work"]
    allowed = barrier_mask(gray.shape, result["transform"], cuts)
    return allowed, dict(
        cuts_work=cuts,
        transform=result["transform"],
        applied_angle=result["applied_angle"],
        gutter_candidates=result["layout"]["gutter_candidates"],
        blocked_pixels=int(np.count_nonzero(allowed == 0)),
        half_width_work_pixels=1.0,
        references_used=False,
    )
