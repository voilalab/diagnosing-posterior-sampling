"""Measurement operator abstraction."""

from collections.abc import Callable
from dataclasses import dataclass

import torch

__all__ = ["AffineForwardModel", "ForwardModel"]


@dataclass
class ForwardModel:
    """Measurement operator :math:`A`."""

    fn: Callable[[torch.Tensor], torch.Tensor]
    derivative: Callable[[torch.Tensor], torch.Tensor]
    name: str = "unknown"
    description: str = ""
    is_linear: bool = False

    def __post_init__(self) -> None:
        if not self.description:
            self.description = self.name


class AffineForwardModel(ForwardModel):
    r"""Affine measurement operator :math:`\mathcal A(x) = M x + b`.

    Stores the matrix :math:`M` and bias :math:`b` as tensor attributes and
    auto-builds ``fn`` and ``derivative`` from them. Downstream code that
    needs the affine coefficients (e.g. closed-form posteriors against
    Gaussian or GMM priors) should dispatch on
    ``isinstance(forward_model, AffineForwardModel)`` and read
    ``forward_model.matrix`` / ``forward_model.bias`` directly rather than
    probing ``fn`` / ``derivative``.

    Args:
        matrix (torch.Tensor): ``()`` 0-D scalar for 1-D :math:`x`, or
            ``(m, d)`` for :math:`d`-dimensional :math:`x` and
            :math:`m`-dimensional :math:`y`.
        bias (torch.Tensor): ``()`` 0-D scalar paired with a 0-D ``matrix``,
            or ``(m,)`` paired with a ``(m, d)`` ``matrix``.
        name (str): operator name for logging/labels. Defaults to
            ``"affine"``.
        description (str): human-readable description; defaults to ``name``.

    Raises:
        ValueError: on shape mismatch between ``matrix`` and ``bias`` or if
            ``matrix`` is neither 0-D nor 2-D.
    """

    matrix: torch.Tensor
    bias: torch.Tensor

    def __init__(
        self,
        matrix: torch.Tensor,
        bias: torch.Tensor,
        name: str = "affine",
        description: str = "",
    ) -> None:
        if matrix.ndim == 0:
            if bias.ndim != 0:
                raise ValueError(
                    f"bias must be 0-D when matrix is 0-D, got shape {tuple(bias.shape)}.",
                )
        elif matrix.ndim == 2:
            if bias.shape != (matrix.shape[0],):
                raise ValueError(
                    f"bias must have shape ({matrix.shape[0]},) for matrix shape "
                    f"{tuple(matrix.shape)}, got {tuple(bias.shape)}.",
                )
        else:
            raise ValueError(
                f"matrix must be 0-D or 2-D, got shape {tuple(matrix.shape)}.",
            )
        self.matrix = matrix
        self.bias = bias

        if matrix.ndim == 0:

            def fn(x: torch.Tensor) -> torch.Tensor:
                return matrix * x + bias

            def derivative(x: torch.Tensor) -> torch.Tensor:
                return matrix.expand(x.shape[0], 1, 1)

        else:
            m_dim, d_dim = matrix.shape

            def fn(x: torch.Tensor) -> torch.Tensor:
                return x @ matrix.T + bias

            def derivative(x: torch.Tensor) -> torch.Tensor:
                return matrix.unsqueeze(0).expand(x.shape[0], m_dim, d_dim)

        super().__init__(
            fn=fn,
            derivative=derivative,
            name=name,
            description=description,
            is_linear=True,
        )
