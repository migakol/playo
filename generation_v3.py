from __future__ import annotations
import os, csv, io, pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict, Counter

from visualize_levels import visualize_sample_level
from PIL import Image

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from typing import Iterable, List, Set, Dict
import pandas as pd
import networkx as nx

from render_graphs import df_to_32x32_images, one_room_df_to_32x32_images
from render_color_graphs import render_file_to_image, save_file_image_png

# ----------------------------
# 1) Build edges per file_id
# ----------------------------
def build_edges_for_file(df_file: pd.DataFrame) -> list[tuple[int, int]]:
    """
    Builds undirected edges (u, v) within this file, where connections are indices in df.index.
    Keeps only edges whose both endpoints are inside df_file.
    """
    file_idx = set(df_file.index.tolist())
    edges = set()

    for u, conn_list in df_file["connections"].items():
        if not isinstance(conn_list, (list, tuple, np.ndarray)):
            continue
        for v in conn_list:
            if v in file_idx and v != u:
                a, b = (u, v) if u < v else (v, u)
                edges.add((a, b))

    return sorted(edges)

def adjacency_from_edges(edges: list[tuple[int,int]]) -> dict[int, set[int]]:
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


# --------------------------------------------------------
# 2) Prune dead-end "center -> entrance" connections
#    (entrance leaf connected to a center, degree==1)
# --------------------------------------------------------
def prune_center_to_entrance_dead_ends(
    df_file: pd.DataFrame,
    edges: list[tuple[int,int]],
    center_label: str = "center",
    entrance_label: str = "entrance",
) -> list[tuple[int,int]]:
    adj = adjacency_from_edges(edges)

    node_type = df_file["node_type"].to_dict()
    to_remove = set()

    for u in list(adj.keys()):
        if node_type.get(u) != entrance_label:
            continue
        if len(adj[u]) != 1:
            continue
        only_neighbor = next(iter(adj[u]))
        if node_type.get(only_neighbor) == center_label:
            a, b = (u, only_neighbor) if u < only_neighbor else (only_neighbor, u)
            to_remove.add((a, b))

    pruned = [e for e in edges if e not in to_remove]
    return pruned


# --------------------------------------------------------
# 3) Canonicalize geometry (X,Z only), ignoring size
#    translate -> PCA rotate -> scale to unit box
# --------------------------------------------------------
def canonicalize_points_xz(points: np.ndarray) -> tuple[np.ndarray, float]:
    """
    points: (N,2) float array of X,Z.
    Returns:
      points_canon: normalized (N,2)
      size_scalar: a "size" measure BEFORE normalization (bbox diagonal)
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) == 0:
        return pts, 0.0

    # size (before normalization): bbox diagonal in XZ
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    size_scalar = float(np.linalg.norm(mx - mn))

    # translate to zero-mean
    centered = pts - pts.mean(axis=0, keepdims=True)

    # PCA rotation (2D)
    cov = np.cov(centered.T) if len(pts) > 1 else np.eye(2)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # sort by descending eigenvalue
    order = np.argsort(eigvals)[::-1]
    R = eigvecs[:, order]

    # fix reflection (make a proper rotation)
    if np.linalg.det(R) < 0:
        R[:, 1] *= -1.0

    rotated = centered @ R

    # scale to unit max dimension (bbox)
    mn2 = rotated.min(axis=0)
    mx2 = rotated.max(axis=0)
    span = (mx2 - mn2)
    scale = float(max(span[0], span[1], 1e-12))
    canon = rotated / scale

    return canon, size_scalar


# --------------------------------------------------------
# 4) Build a "letter signature" (shape-only)
#    - uses canonicalized coordinates rounded
#    - includes node_type so center/entrance matter
#    - uses edges as pairs of node "slots"
# --------------------------------------------------------
def shape_signature(
    df_file: pd.DataFrame,
    edges: list[tuple[int,int]],
    coord_round: int = 2,
) -> str:
    """
    Returns a stable string signature for grouping by shape.
    """
    idx = df_file.index.to_list()

    # canonicalize XZ
    pts = df_file[["x", "z"]].to_numpy(dtype=float)
    canon_pts, _ = canonicalize_points_xz(pts)

    # node "labels": (type, rounded_x, rounded_z)
    rounded = np.round(canon_pts, coord_round)
    node_desc = []
    for i, node_id in enumerate(idx):
        t = str(df_file.at[node_id, "node_type"])
        node_desc.append((node_id, t, float(rounded[i, 0]), float(rounded[i, 1])))

    # reorder nodes in a canonical way: sort by (type, x, z)
    node_desc_sorted = sorted(node_desc, key=lambda x: (x[1], x[2], x[3]))

    # map original node_id -> canonical slot 0..N-1
    slot_of = {node_id: k for k, (node_id, _, _, _) in enumerate(node_desc_sorted)}

    # remap edges into slots and sort
    edge_slots = []
    for u, v in edges:
        if u in slot_of and v in slot_of:
            a, b = slot_of[u], slot_of[v]
            if a != b:
                edge_slots.append((min(a, b), max(a, b)))
    edge_slots = sorted(set(edge_slots))

    # build signature string
    nodes_part = ";".join([f"{t}:{x:.{coord_round}f},{z:.{coord_round}f}" for _, t, x, z in node_desc_sorted])
    edges_part = ";".join([f"{a}-{b}" for a, b in edge_slots])
    return f"N[{nodes_part}]|E[{edges_part}]"


# --------------------------------------------------------
# 5) Build alphabet + sizes per letter
# --------------------------------------------------------
def build_alphabet_and_sizes(
    df: pd.DataFrame,
    center_label: str = "center",
    entrance_label: str = "entrance",
    coord_round: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      letters_df: one row per file with (file_id, letter_id, signature, size_scalar, n_nodes, n_edges)
      alphabet_df: one row per letter_id with counts and size distribution stats
    """
    rows = []

    for file_id, df_file in df.groupby("file_id", sort=False):
        df_file = df_file.copy()

        edges = build_edges_for_file(df_file)
        edges = prune_center_to_entrance_dead_ends(
            df_file, edges, center_label=center_label, entrance_label=entrance_label
        )

        sig = shape_signature(df_file, edges, coord_round=coord_round)

        # size scalar for distribution
        _, size_scalar = canonicalize_points_xz(df_file[["x", "z"]].to_numpy(dtype=float))

        rows.append({
            "file_id": file_id,
            "signature": sig,
            "size_scalar": size_scalar,
            "n_nodes": int(len(df_file)),
            "n_edges": int(len(edges)),
        })

    letters_df = pd.DataFrame(rows)

    # assign stable letter_id (A0, A1, ...)
    uniq_sigs = sorted(letters_df["signature"].unique())
    sig_to_letter = {sig: f"L{idx:04d}" for idx, sig in enumerate(uniq_sigs)}
    letters_df["letter_id"] = letters_df["signature"].map(sig_to_letter)

    # size distribution per letter
    alphabet_df = (letters_df
        .groupby("letter_id")
        .agg(
            n_files=("file_id", "count"),
            size_min=("size_scalar", "min"),
            size_p25=("size_scalar", lambda s: float(np.percentile(s, 25))),
            size_median=("size_scalar", "median"),
            size_p75=("size_scalar", lambda s: float(np.percentile(s, 75))),
            size_max=("size_scalar", "max"),
            size_mean=("size_scalar", "mean"),
            size_std=("size_scalar", "std"),
            n_nodes_mode=("n_nodes", lambda s: int(s.mode().iloc[0]) if len(s.mode()) else int(s.iloc[0])),
            n_edges_mode=("n_edges", lambda s: int(s.mode().iloc[0]) if len(s.mode()) else int(s.iloc[0])),
        )
        .reset_index()
        .sort_values(["n_files", "letter_id"], ascending=[False, True])
    )

    return letters_df, alphabet_df


# ----------------------------
# 6) (Optional) Histogram helper
# ----------------------------
def histogram_sizes_by_letter(letters_df: pd.DataFrame, letter_id: str, bins=20):
    import matplotlib.pyplot as plt

    data = letters_df.loc[letters_df["letter_id"] == letter_id, "size_scalar"].dropna().to_numpy()
    plt.figure()
    plt.hist(data, bins=bins)
    plt.title(f"Size distribution for {letter_id} (n={len(data)})")
    plt.xlabel("size_scalar (bbox diagonal in XZ)")
    plt.ylabel("count")
    plt.show()

def _orient(ax, ay, bx, by, cx, cy) -> float:
    # Cross product of (b-a) x (c-a)
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)

def _on_segment(ax, ay, bx, by, cx, cy) -> bool:
    # c lies on segment a-b (collinear assumed)
    return (min(ax, bx) <= cx <= max(ax, bx)) and (min(ay, by) <= cy <= max(ay, by))

def segments_intersect(
    a: Tuple[float, float],
    b: Tuple[float, float],
    c: Tuple[float, float],
    d: Tuple[float, float],
    *,
    eps: float = 1e-12,
    count_touching_as_intersection: bool = False,
) -> bool:
    """
    Proper segment intersection in 2D.
    If count_touching_as_intersection=False, then touching at endpoints is NOT counted.
    """
    ax, ay = a; bx, by = b; cx, cy = c; dx, dy = d

    o1 = _orient(ax, ay, bx, by, cx, cy)
    o2 = _orient(ax, ay, bx, by, dx, dy)
    o3 = _orient(cx, cy, dx, dy, ax, ay)
    o4 = _orient(cx, cy, dx, dy, bx, by)

    def sgn(x: float) -> int:
        if abs(x) <= eps:
            return 0
        return 1 if x > 0 else -1

    s1, s2, s3, s4 = sgn(o1), sgn(o2), sgn(o3), sgn(o4)

    # General case
    if s1 * s2 < 0 and s3 * s4 < 0:
        return True

    # Collinear / touching cases
    if s1 == 0 and _on_segment(ax, ay, bx, by, cx, cy):  # C on AB
        return count_touching_as_intersection
    if s2 == 0 and _on_segment(ax, ay, bx, by, dx, dy):  # D on AB
        return count_touching_as_intersection
    if s3 == 0 and _on_segment(cx, cy, dx, dy, ax, ay):  # A on CD
        return count_touching_as_intersection
    if s4 == 0 and _on_segment(cx, cy, dx, dy, bx, by):  # B on CD
        return count_touching_as_intersection

    return False


def find_crossing_connections(
    df: pd.DataFrame,
    *,
    eps: float = 1e-12,
    count_touching_as_intersection: bool = False,
    return_first_only: bool = False,
) -> Dict[Any, List[Tuple[Tuple[int, int], Tuple[int, int]]]]:
    """
    For each file_id, build segments from each node to its connected nodes,
    then check whether any two segments intersect.

    Returns:
      dict[file_id] -> list of pairs: [((i,j),(k,l)), ...] where i-j and k-l intersect

    Notes:
    - Assumes `connections` contains GLOBAL df indices.
    - Ignores duplicate undirected edges by normalizing (min(i,j), max(i,j)).
    - Does NOT count intersections where edges share an endpoint (unless you want it).
    """
    result: Dict[Any, List[Tuple[Tuple[int, int], Tuple[int, int]]]] = {}

    # Work per file
    file_cnt = -1
    for file_id, g in df.groupby("file_id"):
        file_cnt += 1
        idx_list = g.index.tolist()
        idx_set = set(idx_list)

        # if file_cnt != 62124:
        #     continue

        if file_cnt % 1000 == 0:
            print('Processing ', file_id)
        # Build unique undirected edges within this file
        edges: List[Tuple[int, int]] = []
        seen = set()
        for i, row in g.iterrows():
            conns = row["connections"] or []
            for j1 in conns:
                j = idx_list[j1]
                if j not in idx_set:
                    continue  # skip edges pointing outside the file
                a, b = (i, j) if i < j else (j, i)
                if a == b:
                    continue
                if (a, b) not in seen:
                    seen.add((a, b))
                    edges.append((a, b))

        if len(edges) < 2:
            continue

        # Cache coordinates
        xs = df["x"]
        ys = df["z"]
        def pt(i: int) -> Tuple[float, float]:
            return (float(xs.loc[i]), float(ys.loc[i]))

        crossings: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []

        # O(E^2) check
        for e_idx in range(len(edges)):
            i, j = edges[e_idx]
            a, b = pt(i), pt(j)

            for f_idx in range(e_idx + 1, len(edges)):
                k, l = edges[f_idx]

                # If they share an endpoint, skip (common in graphs)
                if len({i, j, k, l}) < 4:
                    continue

                c, d = pt(k), pt(l)

                if segments_intersect(
                    a, b, c, d,
                    eps=eps,
                    count_touching_as_intersection=count_touching_as_intersection,
                ):
                    crossings.append(((i, j), (k, l)))
                    if return_first_only:
                        result[file_id] = crossings
                        break
            if return_first_only and file_id in result:
                break

        if crossings:
            result[file_id] = crossings

    return result

def convert_samples(data_dir, output_pickle):
    """
    Analyze data and convert into readable python format (currently pandas), later we can convert into graph
    :param data_dir:
    :return:
    """

    file_arr = []
    type_arr = []
    x_arr = []
    y_arr = []
    z_arr = []
    conn_arr = []


    # for f_cnt, filename in enumerate(os.listdir(data_dir)):
    for f_cnt in range(100000):
        filename = f'/Users/michaelkolomenkin/Data/playo/files_for_yaron/sample_{f_cnt}.txt'
        if f_cnt % 1000 == 0:
            print('Processing ', f_cnt)
        if filename.endswith('.txt'):
            filepath = os.path.join(data_dir, filename)

            try:
                with open(filepath, 'r') as f:
                    lines = f.readlines()

                # Skip header lines
                data_lines = [line.strip() for line in lines if line.strip() and not line.strip().startswith('---')]

                node_types = {'center': 0, 'entrance': 0, 'corridor': 0}
                connections_count = []
                valid_lines = 0

                for line in data_lines:
                    # Skip header line if present
                    if 'node_type,pos,connections' in line:
                        continue

                    # Use CSV parsing to handle quoted fields properly
                    csv_reader = csv.reader(io.StringIO(line))
                    try:
                        parts = next(csv_reader)
                        if len(parts) < 4:
                            continue

                        node_type = parts[1]
                        connections = eval(parts[3])
                        center = eval(parts[2])

                        file_arr.append(f_cnt)
                        type_arr.append(node_type)
                        x_arr.append(center[0])
                        y_arr.append(center[1])
                        z_arr.append(center[2])
                        conn_arr.append(connections)

                        valid_lines += 1

                    except (ValueError, IndexError, SyntaxError) as e:
                        print(f"Error parsing line in {filename}: {line.strip()} - {e}")
                        continue

                if valid_lines > 0:
                    pass

            except Exception as e:
                print(f"Error processing {filename}: {e}")

    df = pd.DataFrame({
        'file_id': file_arr,
        'node_type': type_arr,
        'x': x_arr,
        'y': y_arr,
        'z': z_arr,
        'connections': conn_arr
    })

    with open(output_pickle, 'wb') as f:
        pickle.dump(df, f)


def analyze_stats(pickle_file):
    with open(pickle_file, 'rb') as f:
        df = pickle.load(f)

    rows_per_file = df.groupby("file_id").size()
    rows_per_file.value_counts().sort_index()

    rows_per_file_center = (
        df[df["node_type"] == "center"]
        .groupby("file_id")
        .size()
    )
    rows_per_file_center.value_counts().sort_index()
    # It can be seen that there are 5 rooms per file
    np.histogram(rows_per_file_center)

    floor_decimals = 3  #
    centers = df[df["node_type"].eq("center")].copy()
    centers["floor_y"] = centers["y"].round(floor_decimals)

    # how many distinct floors per file
    floors_per_file = centers.groupby("file_id")["floor_y"].nunique()

    # files where centers span more than 1 floor
    multi_floor_files = floors_per_file[floors_per_file > 1]
    num_multi_floor_files = int(multi_floor_files.size)

    print("Number of files where center nodes are on multiple floors:", num_multi_floor_files)

    crossings_by_file = find_crossing_connections(df, return_first_only=True)

    with open("/Users/michaelkolomenkin/Data/playo/output/crossings.pkl", 'wb') as f:
        pickle.dump(crossings_by_file, f)

    num_files_with_crossings = len(crossings_by_file)
    print(num_files_with_crossings) # list(crossings_by_file.items())[:3]

    # num_files_with_crossings = len(find_crossing_connections(df, return_first_only=True))

    pass

def vis_test():
    # sample_file = "/Users/michaelkolomenkin/Data/playo/files_for_yaron/sample_62124.txt"  # Change this to your file
    sample_file = "/Users/michaelkolomenkin/Data/playo/files_for_yaron/sample_33666.txt"  # Change this to your file
    nodes = visualize_sample_level(sample_file, "/Users/michaelkolomenkin/Data/playo/output")

    with open("/Users/michaelkolomenkin/Data/playo/output/crossings.pkl", 'rb') as f:
        crossings_by_file = pickle.load(f)
    pass


def alphabet_research():
    output_pickle = '/Users/michaelkolomenkin/Data/playo/data/samples.pkl'
    with open(output_pickle, 'rb') as f:
        df = pickle.load(f)
    letters_df, alphabet_df = build_alphabet_and_sizes(df, coord_round=2)
    pass

def _as_list(x: Any) -> List[int]:
    """Normalize df['connections'] cell to a python list of ints."""
    if x is None:
        return []
    if isinstance(x, float) and pd.isna(x):
        return []
    if isinstance(x, list):
        return x
    # if someone stored tuples/sets
    if isinstance(x, (tuple, set)):
        return list(x)
    raise TypeError(f"connections must be list/None/NaN, got {type(x)}: {x!r}")

def remove_dead_end_entrance_connections(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each file_id:
      If a node_type=='entrance' has exactly 1 connection AND it connects to a 'center'
      (and that connected node belongs to the same file_id),
      remove that connection from both sides, but keep the entrance row.
    Assumes connections store row indices (df.index values) of connected rows.
    """
    out = df.copy(deep=True)

    # Ensure every cell is a mutable list (not shared references)
    out["connections"] = out["connections"].apply(lambda v: list(_as_list(v)))

    for file_id, g in out.groupby("file_id", sort=False):
        # if file_id != 62124:
        #     continue
        list_idx = g.index
        idxs = set(list_idx)
        node_type_by_idx = g["node_type"].to_dict()

        entrance_idxs = g.index[g["node_type"] == "entrance"].tolist()
        for e_idx in entrance_idxs:
            e_conns = out.at[e_idx, "connections"]
            # consider only in-file connections
            e_conns_in_file = [list_idx[j] for j in e_conns if list_idx[j] in idxs]

            # "connected only to ONE center"
            if len(e_conns_in_file) == 1:
                j = e_conns_in_file[0]
                if node_type_by_idx.get(j) == "center":
                    # remove edge e_idx -> j
                    out.at[e_idx, "connections"] = [k for k in e_conns if list_idx[k] != j]

                    # remove reverse edge j -> e_idx (if present)
                    j_conns = out.at[j, "connections"]
                    out.at[j, "connections"] = [k for k in j_conns if list_idx[k] != e_idx]

    return out
def remove_dead_end_entrances_connected_only_to_one_center(
    df: pd.DataFrame,
    *,
    entrance_type: str = "entrance",
    center_type: str = "center",
    connections_col: str = "connections",
    file_id_col: str = "file_id",
    node_type_col: str = "node_type",
) -> pd.DataFrame:
    """
    Removes all 'entrance' nodes that are connected only to one node AND that node is a 'center'.
    Also removes those edges from every other node's connections.

    Assumptions:
    - `df[connections_col]` is a list[int] of *row indices* (df.index values) it connects to.
    - connections may be directed or asymmetric in the raw data; we compute degree using
      undirected view (union of outgoing+incoming within the same file).
    """

    df2 = df.copy(deep=True)

    # Ensure connections are lists (avoid None/NaN issues)
    def _as_list(x):
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return []
        return list(x)

    df2[connections_col] = df2[connections_col].apply(_as_list)

    to_remove: Set[int] = set()

    for fid, g in df2.groupby(file_id_col, sort=False):
        if fid % 1000 == 0:
            print('File ', fid)
        nodes: Set[int] = set(g.index)


        # Build undirected adjacency within this file
        adj: Dict[int, Set[int]] = {i: set() for i in nodes}

        for i, conn_list in g[connections_col].items():
            for j in conn_list:
                if j in nodes and j != i:
                    adj[i].add(j)
                    adj[j].add(i)

        # Identify dead-end entrances (degree 1) whose only neighbor is a center
        for i in nodes:
            if df2.at[i, node_type_col] != entrance_type:
                continue
            if len(adj[i]) != 1:
                continue
            (nbr,) = tuple(adj[i])
            if df2.at[nbr, node_type_col] == center_type:
                to_remove.add(i)

    if not to_remove:
        return df2  # nothing to do

    # 1) Remove references to removed nodes from ALL remaining nodes
    to_remove_set = set(to_remove)

    def _filter_conns(conn_list: List[int]) -> List[int]:
        return [j for j in conn_list if j not in to_remove_set]

    df2.loc[~df2.index.isin(to_remove_set), connections_col] = (
        df2.loc[~df2.index.isin(to_remove_set), connections_col]
        .apply(_filter_conns)
    )

    # 2) Drop the entrance rows
    df2 = df2.drop(index=list(to_remove_set))

    return df2


def generate_images(clean_pickle):
    with open(clean_pickle, 'rb') as f:
        df = pickle.load(f)

        # save_file_image_png(df, 62124, "/Users/michaelkolomenkin/Data/playo/images_for_yaron/file_62124.png")

    image_arr = one_room_df_to_32x32_images(df, room_id=-1)

    # for k in range(100000):
    #     image_arr = one_room_df_to_32x32_images(df, room_id=k)
    #     img = Image.fromarray(image_arr[k], mode="L")  # L = grayscale
    #     img.save(f"/Users/michaelkolomenkin/Data/playo/smaller_images/room_{k}.png")
    #
    # pass

if __name__ == "__main__":
    # main()
    samples_dir = '/Users/michaelkolomenkin/Data/playo/files_for_yaron'
    output_pickle = '/Users/michaelkolomenkin/Data/playo/data/samples.pkl'
    clean_pickle = '/Users/michaelkolomenkin/Data/playo/data/samples_clean.pkl'
    # convert_samples(samples_dir, output_pickle)

    # analyze_stats(output_pickle)
    # vis_test()

    # alphabet_research()

    # Remove dead end entrances to draw images
    with open(output_pickle, 'rb') as f:
        df = pickle.load(f)
    # # clean_df = remove_dead_end_entrances_connected_only_to_one_center(df)
    #
    # clean_df = remove_dead_end_entrance_connections(df)
    # #
    # with open(clean_pickle, 'wb') as f:
    #     pickle.dump(clean_df, f)
    # pass

    generate_images(clean_pickle)


