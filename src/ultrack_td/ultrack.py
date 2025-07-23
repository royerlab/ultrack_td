
import tracksdata as td
import higra as hg
import numpy as np
import scipy.ndimage as ndi
from toolz import curry
from numpy.typing import ArrayLike

from skimage.measure._regionprops import regionprops, RegionProperties

from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.graph import BaseGraph
from tracksdata.nodes._base_nodes import BaseNodesOperator
from tracksdata.nodes._mask import Mask
from tracksdata.utils._multiprocessing import multiprocessing_apply
from tracksdata.utils._logging import LOG


def _bbox_and_mask_from_leaves(
    nodes: dict[int, dict[str, Any]],
    tree: hg.Tree,
    node_idx: int,
) -> tuple[ArrayLike, ArrayLike]:

    crop_mask = self._parent.props.image
    tree = self._parent.tree

    size = tree.num_vertices()
    if self._parent._tree_buffer is None:
        not_selected = np.ones(size, dtype=bool)
    else:
        not_selected = self._parent._tree_buffer

    not_selected[self._h_node_index] = False
    leaves_labels = hg.reconstruct_leaf_data(tree, np.arange(size), not_selected)
    not_selected[self._h_node_index] = True  # resetting

    if self._parent._mask_buffer is None:
        mask = np.zeros(crop_mask.shape, dtype=int)
    else:
        mask = self._parent._mask_buffer

    mask[crop_mask] = leaves_labels
    binary_mask = mask == self._h_node_index
    mask[crop_mask] = 0  # resetting

    return self._reduce_mask(binary_mask)


def _bbox_and_mask_from_non_leaves(
    nodes: dict[int, dict[str, Any]],
    children: ArrayLike,
    tree: hg.Tree,
) -> tuple[ArrayLike, ArrayLike]:
    """
    Computes bounding-box and mask of non-leaf nodes using the
    previously data from its children. Much faster than computing from the hierarchy leaves.


    Parameters
    ----------
    nodes : dict[int, dict[str, Any]]
        Dictionary of nodes.
    children : ArrayLike
        Array of non-leaf children nodes.
    tree : hg.Tree
        Pre-computed hierarchy tree.

    Returns:
        Tuple[ArrayLike, ArrayLike]: Bounding-box and mask.
    """
    if tree is None:
        tree = self._parent.tree

    ndim = self._parent.props._ndim
    bbox = np.zeros(2 * ndim, dtype=int)
    bbox[:ndim] = np.iinfo(int).max

    for child in children:
        try:
            # this can happen due to the maximum size selecion
            child_bbox = nodes[child][DEFAULT_ATTR_KEYS.MASK].bbox
        except KeyError:
            return _bbox_and_mask_from_leaves(nodes, tree, child)

        for i in range(ndim):
            bbox[i] = min(bbox[i], child_bbox[i])
            bbox[i + ndim] = max(bbox[i + ndim], child_bbox[i + ndim])

    shape = tuple(M - m for m, M in zip(bbox[:ndim], bbox[ndim:]))
    mask = np.zeros(shape, dtype=bool)
    for child in children:
        child_bbox = nodes[child][DEFAULT_ATTR_KEYS.MASK].bbox
        slicing = []
        for i in range(ndim):
            start = child_bbox[i] - bbox[i]
            end = start + child_bbox[i + ndim] - child_bbox[i]
            slicing.append(slice(start, end))
        slicing = tuple(slicing)
        mask[slicing] |= self._parent._nodes[child].mask

    return bbox, mask


def _tree_node_mask(
    nodes: dict[int, dict[str, Any]],
    tree: hg.Tree,
    node_idx: int,
) -> Mask:

    children = tree.children(node_idx)
    num_leaf_children = (children < tree.num_leaves()).sum()

    if num_leaf_children > 0:
        return _bbox_and_mask_from_leaves(nodes, tree, node_idx)

    return _bbox_and_mask_from_non_leaves(
        nodes, children[children >= tree.num_leaves()], tree
    )


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
        self._hierarchy_fun = hg.watershed_hierarchy_by_area

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

    def _filter_contour_strength(
        self,
        tree: hg.Tree,
        alt: np.ndarray,
        frontier: np.ndarray,
    ) -> tuple[hg.Tree, np.ndarray, np.ndarray]:
        LOG.info("Filtering hierarchy by contour strength.")
        irrelevant_nodes = frontier < self._min_frontier

        if self._min_num_pixels is not None:
            # Avoid filtering nodes where merge leads to a node with maximum area above threshold
            parent_area = hg.attribute_area(tree)[tree.parents()]
            irrelevant_nodes[parent_area > self._min_num_pixels] = False

        tree, node_map = hg.simplify_tree(tree, irrelevant_nodes)
        return tree, alt[node_map], frontier[node_map]
        
    def _compute_nodes(self, obj: RegionProperties, **kwargs) -> list[dict[str, Any]]:

        contours = obj.intensity_image
        if contours.dtype == np.float16:
            contours = contours.astype(np.float32)

        if contours.size < 8:
            raise RuntimeError(f"Region too small. Size of {contours.size} found.")

        LOG.info("Creating graph from mask.")
        mask = obj.image
        graph, weights = mask_to_graph(mask, contours)

        LOG.info("Constructing hierarchy.")
        tree, alt = self._hierarchy_fun(graph, weights)

        LOG.info("Filtering small nodes of hierarchy.")
        tree, alt = hg.filter_small_nodes_from_tree(tree, alt, self._min_num_pixels)

        hg.set_attribute(graph, "no_border_vertex_out_degree", None)
        frontier = hg.attribute_contour_strength(tree, weights)
        hg.set_attribute(graph, "no_border_vertex_out_degree", 2 * mask.ndim)

        if self._min_frontier is not None:
            tree, alt, frontier = self._filter_contour_strength(
                tree,
                alt,
                frontier,
            )

        LOG.info("Filtering large nodes of hierarchy.")
        num_pixels = hg.attribute_area(tree)
        invalid_nodes = num_pixels > self._max_num_pixels

        extra_root = False
        if invalid_nodes.all():
            LOG.warning("All nodes are too large. Keeping root node.")
            invalid_nodes[-1] = False  # keeping root node
            extra_root = True

        tree, node_map = hg.simplify_tree(tree, invalid_nodes)
        alt = alt[node_map]
        frontier = frontier[node_map]
        num_pixels = num_pixels[node_map]

        if len(alt) == 0:
            raise RuntimeError("No nodes found.")
        
        nodes = {}

        for node_idx in tree.leaves_to_root_iterator():
            if num_pixels[node_idx] > self._max_num_pixels:
                LOG.warning(f"Node {node_idx} is too large even when 'extra_root' is {extra_root}. Skipping.")
                continue

            nodes[node_idx] = {
                "num_pixels": num_pixels[node_idx].item(),
                "altitude": alt[node_idx].item(),
                "frontier": frontier[node_idx].item(),
                DEFAULT_ATTR_KEYS.MASK: _tree_node_mask(nodes, tree, node_idx),
                **kwargs,
            }
        
        return list(nodes.values())


    def _add_nodes_per_time(
        self,
        t: int,
        *,
        foreground: ArrayLike,
        contours: ArrayLike,
    ) -> list[dict[str, Any]]:

        foreground = np.asarray(foreground)
        labels, num_labels = ndi.label(foreground)

        node_attrs = []

        for obj in regionprops(labels, contours, cache=True):
            node_attrs.extend(self._compute_nodes(obj, t=t))

        return node_attrs
