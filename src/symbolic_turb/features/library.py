"""
------------
Builds a feature/library matrix Theta consisting of terms:

    (I1^a * I2^b) * Tn

for each tensor basis Tn in {T1, T2, T3} and for all exponent pairs (a,b)
such that:

    a + b <= max_degree

By default it *includes* the (a,b) = (0,0) term, i.e. the plain tensor basis Tn.
This matches your previous behavior (you had deg=0..max for I1^deg * Tn).

Shapes
------
Tn: (N, 3, 3)
I1,I2: (N, 1, 1)

Each feature column is flattened as (N*9,).
Theta returned is: (N*9, n_features)
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from symbolic_turb.core import FlowData


@dataclass(frozen=True)
class FeatureDescription:
    """Metadata for a single column in the library."""

    name: str  # ex: "I1^2 * I2 * T2"
    invariant_power: Tuple[int, ...]  # ex: (2, 1) meaning I1^2 * I2^1
    tensor_basis: int  # ex: 2 for T2


def _iter_total_degree_powers(
    n_invariants: int, max_degree: int, *, include_zero: bool
) -> Iterable[Tuple[int, ...]]:
    """
    Generate exponent tuples (a1, ..., ak) with total degree <= max_degree.
    If include_zero=True, includes the all-zero tuple (0,0,...,0).
    """
    if n_invariants <= 0:
        raise ValueError("n_invariants must be >= 1")
    if max_degree < 0:
        raise ValueError("max_degree must be >= 0")

    for exps in product(range(max_degree + 1), repeat=n_invariants):
        s = sum(exps)
        if s > max_degree:
            continue
        if not include_zero and s == 0:
            continue
        yield exps


def _monomial(invariants: Sequence[np.ndarray], powers: Sequence[int]) -> np.ndarray:
    """
    Compute monomial: prod_i invariants[i] ** powers[i]

    invariants are expected to be broadcastable together (and later with T).
    Typical invariant shape: (N,1,1).
    """
    if len(invariants) != len(powers):
        raise ValueError("len(invariants) must match len(powers)")

    out = np.ones_like(invariants[0])
    for I, p in zip(invariants, powers):
        if p:
            out = out * (I**p)
    return out


def _format_invariant_name(powers: Sequence[int]) -> str:
    """
    Convert (a,b,...) -> "I1^a * I2^b * ..."
    Skips zero powers. If all are zero, returns "1".
    """
    parts: List[str] = []
    for i, p in enumerate(powers, start=1):
        if p == 0:
            continue
        if p == 1:
            parts.append(f"I{i}")
        else:
            parts.append(f"I{i}^{p}")
    return " * ".join(parts) if parts else "1"


class CandidateLibrary:
    def __init__(self, max_degree: int = 2):
        self.max_degree = int(max_degree)
        self.feature_names: list[FeatureDescription] = []
        self._numerical_matrix: Optional[np.ndarray] = None
        self.include_invariant_constant: bool = True

    def fit_transform(self, flow_data: FlowData, threshold: float = 1e-5) -> np.ndarray:
        """
        Builds the library matrix Theta from flow data.

        Returns
        -------
        Theta: np.ndarray
            Shape: (N*9, n_features)
        """
        assert flow_data.T1.shape[0] != 0, "Tensor basis is empty"
        assert flow_data.I1.shape[0] != 0, "Invariant is empty"

        T_basis = [flow_data.T1, flow_data.T2, flow_data.T3]  # each (N,3,3)
        invariants = [flow_data.I1, flow_data.I2]  # each (N,1,1) typically

        N = T_basis[0].shape[0]
        for k, T in enumerate(T_basis, start=1):
            if T.shape != (N, 3, 3):
                raise ValueError(f"T{k} must have shape (N,3,3). Got {T.shape}.")
        for i, I in enumerate(invariants, start=1):
            if I.shape[0] != N:
                raise ValueError(
                    f"I{i} must have same N as T. Got {I.shape[0]} vs {N}."
                )

        # Precompute exponent tuples
        power_tuples = list(
            _iter_total_degree_powers(
                n_invariants=len(invariants),
                max_degree=self.max_degree,
                include_zero=self.include_invariant_constant,
            )
        )

        scalar_terms = [_monomial(invariants, p) for p in power_tuples]
        features_list: list[np.ndarray] = []
        descriptions: list[FeatureDescription] = []

        for tensor_idx, T in enumerate(T_basis, start=1):
            for powers, I_term in zip(power_tuples, scalar_terms):
                term = T * I_term  # (N,3,3)

                # SpaRTA magnitude filter
                if np.max(np.abs(term)) > threshold:
                    continue

                flat_term = term.reshape(-1)
                features_list.append(flat_term)

                inv_name = _format_invariant_name(powers)
                name = (
                    f"T{tensor_idx}"
                    if inv_name == "1"
                    else f"{inv_name} * T{tensor_idx}"
                )

                descriptions.append(
                    FeatureDescription(
                        name=name,
                        invariant_power=tuple(int(p) for p in powers),
                        tensor_basis=tensor_idx,
                    )
                )

        if not features_list:
            raise ValueError(
                "All candidates were filtered out! Check your data scaling."
            )

        self._numerical_matrix = np.stack(features_list, axis=1)
        self.feature_names = descriptions
        return self._numerical_matrix

    def get_feature_name(self, idx: int) -> str:
        return self.feature_names[idx].name

    def __len__(self):
        return len(self.feature_names)
