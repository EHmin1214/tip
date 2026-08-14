# -*- coding: utf-8 -*-
"""optimize — electrode and current optimisation engines.

Phase 1:  classic (two-channel exhaustive search, the baseline oracle).
Phase 2+: nsga (multi-objective GA), multichannel (LP), sequential.
"""
from .classic import optimize_classic, pareto_front, weighted_performance

__all__ = ["optimize_classic", "pareto_front", "weighted_performance"]
