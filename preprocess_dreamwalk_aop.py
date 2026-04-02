#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adapted preprocess_dreamwalk.py for the AOP Rare Disease KG.
- No drugs / ATC filtering (not present in this KG)
- Keeps: Disease, Gene (Protein), Pathway, AOP nodes
- Keeps all relations: associated_with, involves_gene, interacts_with,
                       part_of_pathway, adjacent
- Outputs preprocessed_graph.csv for DreamWalk
"""

import os
from pathlib import Path
import pandas as pd

path = Path(__file__).resolve().parent
os.chdir(path)

# ── 1. Load ──────────────────────────────────────────────────────────────────
data = pd.read_csv('aop_raredisease_kgpathway_neo4j.csv', low_memory=False)

# ── 2. Separate nodes and edges ───────────────────────────────────────────────
nodes = data[data['_start'].isna()].copy()
edges = data[data['_start'].notna()].copy()

print(f"Raw nodes : {len(nodes)}")
print(f"Raw edges : {len(edges)}")
print(f"\nNode label counts:\n{nodes['_labels'].value_counts()}")
print(f"\nEdge type counts:\n{edges['type'].value_counts()}")

# ── 3. Standardise node labels (mirror original script convention) ────────────
label_map = {
    ':disease' : ':Disease',
    ':gene'    : ':Protein',   # DreamWalk calls genes "Protein"
    ':pathway' : ':Pathway',
    ':AOP'     : ':AOP',
}
nodes['_labels'] = nodes['_labels'].replace(label_map)

# ── 4. Keep only the four node types we need ─────────────────────────────────
keep_labels = {':Disease', ':Protein', ':Pathway', ':AOP'}
nodes = nodes[nodes['_labels'].isin(keep_labels)].copy()

# ── 5. Keep only edges whose BOTH endpoints are in the retained node set ─────
valid_ids = set(nodes['_id'])
edges = edges[
    edges['_start'].isin(valid_ids) & edges['_end'].isin(valid_ids)
].copy()

# ── 6. No relation-type exclusions for this KG ───────────────────────────────
#    (all 5 relation types are biologically meaningful)
#    If you want to drop any, uncomment and edit:
# edges = edges[~edges['type'].isin(['adjacent'])]

print(f"\nAfter filtering:")
print(f"  Nodes : {len(nodes)}")
print(f"  Edges : {len(edges)}")
print(f"\nRetained edge types:\n{edges['type'].value_counts()}")

# ── 7. Add placeholder columns expected by DreamWalk downstream ──────────────
nodes['class']    = None
nodes['uniprotID'] = None

# ── 8. Rebuild a single dataframe (nodes first, then edges) ──────────────────
preprocessed = pd.concat([nodes, edges], ignore_index=True)

# ── 9. Save ───────────────────────────────────────────────────────────────────
preprocessed.to_csv('preprocessed_graph.csv', index=False)
print(f"\n✓ Saved preprocessed_graph.csv  ({len(preprocessed)} rows)")

# ── 10. Summary stats ─────────────────────────────────────────────────────────
print("\n── Final node counts ──────────────────────────────────────────────────")
final_nodes = preprocessed[preprocessed['_start'].isna()]
print(final_nodes['_labels'].value_counts().to_string())

print("\n── Final edge counts ──────────────────────────────────────────────────")
final_edges = preprocessed[preprocessed['_start'].notna()]
print(final_edges['type'].value_counts().to_string())
