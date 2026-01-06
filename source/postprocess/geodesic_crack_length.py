import heapq
import importlib.util
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

_HAS_CKD_TREE = importlib.util.find_spec("scipy.spatial") is not None
if _HAS_CKD_TREE:
    from scipy.spatial import cKDTree  # type: ignore
else:
    cKDTree = None


def read_comsol_table(
    path: str,
    column_map: Optional[Dict[str, str]] = None,
    delimiter: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    Read a COMSOL-exported text table into numpy arrays.

    Parameters
    ----------
    path:
        Path to the exported table. The file is expected to contain a header
        row with at least the time, x, y and damage columns. Lines starting
        with ``#`` are treated as comments.
    column_map:
        Optional mapping to override column discovery,
        e.g. ``{"t": "Time", "d": "damage"}``.
    delimiter:
        Delimiter passed to ``numpy.genfromtxt``. By default whitespace is used.

    Returns
    -------
    dict
        Dictionary with keys ``t``, ``x``, ``y`` and ``d``.
    """
    with open(path, "r") as handle:
        lines = handle.readlines()

    header_line = None
    header_idx = None
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        header_line = stripped
        header_idx = idx
        break

    if header_line is None or header_idx is None:
        raise ValueError(f"Could not locate a header row in {path}")

    if delimiter is None:
        columns = [col for col in header_line.split() if col]
    else:
        columns = [col.strip() for col in header_line.split(delimiter) if col.strip()]

    raw_data = np.genfromtxt(
        path,
        delimiter=delimiter,
        comments="#",
        skip_header=header_idx + 1,
    )
    if raw_data.ndim == 1:
        raw_data = raw_data.reshape(1, -1)
    if raw_data.shape[1] != len(columns):
        raise ValueError(
            f"Header/data column mismatch in {path}: "
            f"{len(columns)} names vs {raw_data.shape[1]} columns"
        )
    table = dict(zip(columns, raw_data.T))

    def _choose(target: str, aliases: Sequence[str]) -> str:
        if column_map and target in column_map:
            return column_map[target]
        for candidate in columns:
            normalized = candidate.lower().replace(" ", "")
            if normalized == target or normalized in aliases:
                return candidate
        raise KeyError(f"Column for '{target}' not found in header: {columns}")

    time_key = _choose("t", aliases=("time",))
    x_key = _choose("x", aliases=("xcoord",))
    y_key = _choose("y", aliases=("ycoord",))
    d_key = _choose("d", aliases=("damage", "pf", "alpha"))

    return {
        "t": np.asarray(table[time_key], dtype=float),
        "x": np.asarray(table[x_key], dtype=float),
        "y": np.asarray(table[y_key], dtype=float),
        "d": np.asarray(table[d_key], dtype=float),
        "columns": np.array(columns),
    }


def build_knn_graph(
    points: np.ndarray,
    k: int = 8,
    use_kdtree: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a k-nearest-neighbour graph.

    Parameters
    ----------
    points:
        Array of shape (N, 2) containing point coordinates.
    k:
        Number of neighbours per node.
    use_kdtree:
        If True and SciPy is available, ``scipy.spatial.cKDTree`` is used.
        Otherwise a pure numpy implementation is used.

    Returns
    -------
    (indices, distances)
        ``indices`` has shape (N, k_eff) containing neighbour indices, and
        ``distances`` the corresponding edge lengths.
    """
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")
    n_points = points.shape[0]
    if n_points < 2:
        return np.empty((n_points, 0), dtype=int), np.empty((n_points, 0))
    k_eff = max(1, min(k, n_points - 1))

    if use_kdtree and _HAS_CKD_TREE and cKDTree is not None:
        tree = cKDTree(points)
        distances, indices = tree.query(points, k=k_eff + 1)
        indices = indices[:, 1:]
        distances = distances[:, 1:]
    else:
        diff = points[:, None, :] - points[None, :, :]
        distance_matrix = np.linalg.norm(diff, axis=2)
        indices = np.argsort(distance_matrix, axis=1)[:, 1 : k_eff + 1]
        row_indices = np.arange(n_points)[:, None]
        distances = distance_matrix[row_indices, indices]
    return indices, distances


def connected_component_from_root(
    neighbors: np.ndarray,
    root_index: int,
    valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Extract the connected component that contains ``root_index``.
    """
    n_nodes = neighbors.shape[0]
    if valid_mask is None:
        valid_mask = np.ones(n_nodes, dtype=bool)
    component = np.zeros(n_nodes, dtype=bool)
    if root_index < 0 or root_index >= n_nodes or not valid_mask[root_index]:
        return component
    stack = [root_index]
    component[root_index] = True
    while stack:
        node = stack.pop()
        for neighbor in neighbors[node]:
            if valid_mask[neighbor] and not component[neighbor]:
                component[neighbor] = True
                stack.append(neighbor)
    return component


def geodesic_length(
    neighbors: np.ndarray,
    distances: np.ndarray,
    root_index: int,
    allowed_mask: Optional[np.ndarray] = None,
) -> Tuple[float, np.ndarray]:
    """
    Run Dijkstra from root and return the maximum geodesic distance.
    """
    n_nodes = neighbors.shape[0]
    if allowed_mask is None:
        allowed_mask = np.ones(n_nodes, dtype=bool)
    dist_to = np.full(n_nodes, np.inf)
    if root_index < 0 or root_index >= n_nodes or not allowed_mask[root_index]:
        return 0.0, dist_to

    dist_to[root_index] = 0.0
    heap = [(0.0, root_index)]
    heapq.heapify(heap)
    while heap:
        current_dist, node = heapq.heappop(heap)
        if current_dist > dist_to[node]:
            continue
        for neighbor, edge_length in zip(neighbors[node], distances[node]):
            if not allowed_mask[neighbor]:
                continue
            candidate_dist = current_dist + float(edge_length)
            if candidate_dist < dist_to[neighbor]:
                dist_to[neighbor] = candidate_dist
                heapq.heappush(heap, (candidate_dist, neighbor))

    reachable = np.isfinite(dist_to) & allowed_mask
    if not reachable.any():
        return 0.0, dist_to
    return float(dist_to[reachable].max()), dist_to


def extract_a_t_geodesic(
    table_path: str,
    x_root: float,
    y_root: float,
    d_threshold: float,
    k: int = 8,
    y_band: Optional[float] = None,
    delimiter: Optional[str] = None,
    column_map: Optional[Dict[str, str]] = None,
    use_kdtree: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract crack length a(t) using geodesic distance from COMSOL table.

    For each time step the algorithm performs:
    1. Mask points with ``d >= d_threshold`` (and optional ``y_band``).
    2. Keep only the connected component that contains ``(x_root, y_root)``.
    3. Run Dijkstra on the subgraph and take the maximum geodesic distance.
    """
    table = read_comsol_table(table_path, column_map=column_map, delimiter=delimiter)
    t_all, x_all, y_all, d_all = (
        table["t"],
        table["x"],
        table["y"],
        table["d"],
    )
    times = np.unique(t_all)
    lengths = []
    root_xy = np.array([x_root, y_root])
    for t_value in times:
        time_mask = np.isclose(t_all, t_value)
        coords = np.column_stack((x_all[time_mask], y_all[time_mask]))
        damage = d_all[time_mask]
        if coords.size == 0:
            lengths.append(0.0)
            continue

        root_index = int(
            np.linalg.norm(coords - root_xy[None, :], axis=1).argmin()
        )
        active_mask = damage >= d_threshold
        if y_band is not None:
            active_mask &= np.abs(coords[:, 1] - y_root) <= y_band
        if active_mask.sum() < 2 or not active_mask[root_index]:
            lengths.append(0.0)
            continue

        keep_indices = np.nonzero(active_mask)[0]
        sub_coords = coords[active_mask]
        sub_root_index = int(np.where(keep_indices == root_index)[0][0])

        neighbors, dist_matrix = build_knn_graph(
            sub_coords,
            k=k,
            use_kdtree=use_kdtree,
        )
        component_mask = connected_component_from_root(
            neighbors,
            sub_root_index,
        )
        if component_mask.sum() < 2:
            lengths.append(0.0)
            continue
        max_distance, _ = geodesic_length(
            neighbors,
            dist_matrix,
            sub_root_index,
            allowed_mask=component_mask,
        )
        lengths.append(max_distance)

    return times, np.asarray(lengths, dtype=float)
