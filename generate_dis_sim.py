#!/usr/bin/env python3
"""
Generate a disease similarity (dis_sim.tsv) file using the DreamWalk approach.

Approach (mirrors macsbio/post-covid-kg DREAMwalk generate_dis_sim.py):
  1. Parse graph (1).txt to build disease–entity associations from graph edges.
     Disease nodes are identified by checking whether a node ID appears in the
     ORPHA hierarchy (as a child or parent).  Non-disease neighbours of a
     disease node become that disease's "gene-like" association set.
  2. When a disease has no graph-based associations (e.g. because the current
     graph does not yet contain ORPHA nodes), fall back to hierarchy-based
     associations: the disease's direct parent nodes in the ORPHA hierarchy,
     restricted to parents that are specific enough (fewer than
     MAX_PARENT_BREADTH children) to be discriminative.
  3. Compute pairwise Jaccard similarity efficiently using sparse-matrix
     operations (scipy) instead of a nested Python loop.
  4. Keep pairs whose similarity > threshold (default 0.4) and write both
     directions to dis_sim.tsv.

Output format (tab-separated, no header):
  Disease_1  Disease_2  3  jaccard_score  index
"""

from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix, csr_matrix


# Maximum number of children a hierarchy parent node may have while still
# being considered a discriminative feature (very broad categories are skipped).
MAX_PARENT_BREADTH: int = 500


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_dis_sim(
    graph_file: str,
    hierarchy_file: str,
    output_file: str,
    threshold: float = 0.4,
    max_parent_breadth: int = MAX_PARENT_BREADTH,
) -> None:
    """
    Generate the disease-similarity TSV.

    Parameters
    ----------
    graph_file         : path to graph (1).txt  (tab-separated DreamWalk network)
    hierarchy_file     : path to disease_orpha_hierarchy_combined.csv
    output_file        : path for the dis_sim.tsv output
    threshold          : Jaccard pairs at or below this value are discarded
                         (default 0.4, matching the DreamWalk reference)
    max_parent_breadth : hierarchy parents with more children than this value
                         are treated as too broad and excluded from the
                         association profile (default 500)
    """

    # ------------------------------------------------------------------
    # 1. Parse the graph (tab-separated: source, target, type, weight, idx)
    # ------------------------------------------------------------------
    graph_df = pd.read_csv(
        graph_file,
        sep="\t",
        header=None,
        names=["source", "target", "type", "weight", "idx"],
    )
    graph_df["source"] = graph_df["source"].astype(str)
    graph_df["target"] = graph_df["target"].astype(str)

    # Undirected adjacency: node -> set of direct neighbours
    adjacency: dict = defaultdict(set)
    for _, row in graph_df.iterrows():
        adjacency[row["source"]].add(row["target"])
        adjacency[row["target"]].add(row["source"])

    graph_nodes: set = set(adjacency.keys())
    print(f"Graph nodes  : {len(graph_nodes)}")
    print(f"Graph edges  : {len(graph_df)}")

    # ------------------------------------------------------------------
    # 2. Parse the disease hierarchy
    # ------------------------------------------------------------------
    hierarchy_df = pd.read_csv(hierarchy_file)
    hierarchy_df["child"]  = hierarchy_df["child"].astype(str)
    hierarchy_df["parent"] = hierarchy_df["parent"].astype(str)

    # parent_map: disease -> set of direct parent IDs
    parent_map: dict = defaultdict(set)
    for _, row in hierarchy_df.iterrows():
        parent_map[row["child"]].add(row["parent"])

    # All disease IDs in the hierarchy (leaf diseases as candidate output nodes)
    leaf_diseases: set = set(hierarchy_df["child"].unique())
    all_hierarchy_diseases: set = (
        leaf_diseases | set(hierarchy_df["parent"].unique())
    )
    print(f"Diseases in hierarchy : {len(all_hierarchy_diseases)}")

    # Identify which graph nodes are disease nodes (appear in hierarchy)
    disease_graph_nodes: set = graph_nodes & all_hierarchy_diseases
    print(f"Disease nodes in graph: {len(disease_graph_nodes)}")

    # Discriminative hierarchy parents (not overly broad categories)
    parent_child_counts = hierarchy_df["parent"].value_counts()
    specific_parents: set = set(
        parent_child_counts[parent_child_counts < max_parent_breadth].index
    )
    print(
        f"Specific hierarchy parents (< {max_parent_breadth} children): "
        f"{len(specific_parents)}"
    )

    # ------------------------------------------------------------------
    # 3. Build association profiles for each leaf disease
    #
    #    Primary   : graph neighbours that are NOT disease nodes
    #                (analogous to gene/protein associations in original DreamWalk)
    #    Fallback  : direct parents in the ORPHA hierarchy that are specific
    #                enough (fewer than max_parent_breadth children)
    # ------------------------------------------------------------------
    leaf_disease_list = sorted(leaf_diseases)
    disease_idx = {d: i for i, d in enumerate(leaf_disease_list)}

    # Build the union of all features across all diseases to create column index
    all_features: set = set()
    disease_profile_raw: dict = {}

    for disease in leaf_disease_list:
        profile: set = set()

        # Graph-based: neighbours of this disease node that are not diseases
        if disease in disease_graph_nodes:
            profile.update(
                nb for nb in adjacency[disease]
                if nb not in all_hierarchy_diseases
            )

        # Hierarchy fallback: specific direct parents
        if not profile:
            profile.update(
                p for p in parent_map.get(disease, set())
                if p in specific_parents
            )

        disease_profile_raw[disease] = profile
        all_features.update(profile)

    feature_list = sorted(all_features)
    feature_idx  = {f: i for i, f in enumerate(feature_list)}
    print(f"Feature dimensions    : {len(feature_list)}")

    # Diseases that have at least one association
    active_diseases = [d for d in leaf_disease_list if disease_profile_raw[d]]
    print(f"Diseases with associations: {len(active_diseases)}")

    if not active_diseases:
        print(
            "No disease associations found.  "
            "Ensure the graph contains ORPHA disease nodes connected to "
            "non-disease entities, or that the hierarchy file is populated."
        )
        pd.DataFrame().to_csv(output_file, sep="\t", index=False, header=False)
        return

    # ------------------------------------------------------------------
    # 4. Build sparse binary matrix: diseases × features
    # ------------------------------------------------------------------
    active_idx = {d: i for i, d in enumerate(active_diseases)}
    X = lil_matrix((len(active_diseases), len(feature_list)), dtype=np.int8)
    for disease in active_diseases:
        di = active_idx[disease]
        for feat in disease_profile_raw[disease]:
            if feat in feature_idx:
                X[di, feature_idx[feat]] = 1

    X = csr_matrix(X)
    row_sums = np.asarray(X.sum(axis=1)).flatten().astype(float)

    # ------------------------------------------------------------------
    # 5. Compute pairwise Jaccard similarity via sparse matrix multiply
    #    intersection[i,j] = (X @ X.T)[i,j]
    #    union[i,j]        = row_sums[i] + row_sums[j] - intersection[i,j]
    #    jaccard[i,j]      = intersection / union
    # ------------------------------------------------------------------
    print("Calculating Jaccard similarities …")
    XX = X.dot(X.T).tocoo()

    r_arr = np.array(XX.row)
    c_arr = np.array(XX.col)
    d_arr = np.array(XX.data, dtype=float)

    # Keep only upper-triangle pairs (i < j)
    tri_mask = r_arr < c_arr
    r_arr = r_arr[tri_mask]
    c_arr = c_arr[tri_mask]
    d_arr = d_arr[tri_mask]

    union_arr  = row_sums[r_arr] + row_sums[c_arr] - d_arr
    jaccard_arr = np.where(union_arr > 0, d_arr / union_arr, 0.0)

    sim_mask = jaccard_arr > threshold
    r_arr     = r_arr[sim_mask]
    c_arr     = c_arr[sim_mask]
    jaccard_arr = jaccard_arr[sim_mask]

    n_pairs = len(r_arr)
    print(f"Disease similarity pairs (Jaccard > {threshold}): {n_pairs}")

    # ------------------------------------------------------------------
    # 6. Build output dataframe (both directions, matching reference impl.)
    # ------------------------------------------------------------------
    d1_list  = [active_diseases[i] for i in r_arr]
    d2_list  = [active_diseases[j] for j in c_arr]

    rows_fwd = list(zip(d1_list, d2_list, [3] * n_pairs, jaccard_arr, range(0, 2 * n_pairs, 2)))
    rows_rev = list(zip(d2_list, d1_list, [3] * n_pairs, jaccard_arr, range(1, 2 * n_pairs + 1, 2)))

    all_rows = []
    for fwd, rev in zip(rows_fwd, rows_rev):
        all_rows.append(fwd)
        all_rows.append(rev)

    sim_df = pd.DataFrame(all_rows)
    sim_df.to_csv(output_file, sep="\t", index=False, header=False)
    print(f"Saved: {output_file}  ({len(sim_df)} rows, both directions)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    base = Path(__file__).resolve().parent

    generate_dis_sim(
        graph_file=str(base / "graph (1).txt"),
        hierarchy_file=str(base / "disease_orpha_hierarchy_combined.csv"),
        output_file=str(base / "dis_sim.tsv"),
        threshold=0.4,
    )
