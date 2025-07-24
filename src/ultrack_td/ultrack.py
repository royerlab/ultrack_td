from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from toolz import curry
from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.graph import BaseGraph
from tracksdata.nodes._base_nodes import BaseNodesOperator
from tracksdata.utils._multiprocessing import multiprocessing_apply


class UltrackCandidateNodes(BaseNodesOperator):
    def __init__(
        self,
        min_num_pixels: int,
        max_num_pixels: int,
        min_frontier: float | None,
    ):
        self._min_num_pixels = min_num_pixels
        self._max_num_pixels = max_num_pixels
        self._min_frontier = min_frontier

    def _init_nodes(self, graph: BaseGraph) -> None:
        raise NotImplementedError("Not implemented")

    def add_nodes(
        self,
        graph: BaseGraph,
        *,
        t: int | None = None,
        foreground: ArrayLike,
        contours: ArrayLike,
    ) -> None:
        self._init_nodes(graph)

        if t is None:
            time_points = graph.time_points()
        else:
            time_points = [t]

        node_ids = []
        for node_attrs in multiprocessing_apply(
            func=curry(self._add_nodes_per_time, foreground=foreground, contours=contours),
            sequence=time_points,
            desc="Adding nodes",
        ):
            node_ids.extend(graph.bulk_add_nodes(node_attrs))

    def _add_nodes_per_time(
        self,
        t: int,
        *,
        foreground: ArrayLike,
        contours: ArrayLike,
    ) -> list[dict[str, Any]]:
        foreground = np.asarray(foreground)
        contours = np.asarray(contours)

        node_attrs = []  # TODO _compute_nodes(contours)

        for node_attr in node_attrs:
            node_attr[DEFAULT_ATTR_KEYS.T] = t

        return node_attrs
