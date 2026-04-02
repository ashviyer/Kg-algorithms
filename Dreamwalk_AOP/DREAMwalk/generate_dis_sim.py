import os
import pandas as pd
import numpy as np

# PATCHED: removed requests, json, sklearn imports (not needed)
# Original used hardcoded Windows paths to DisGeNET CSV files.
# Adapted: derive disease-gene associations directly from the KG
#          using the associated_with edges (disease → gene).

def map_id(row):
    if row['_labels'] == ':Disease':
        return row.get('diseaseID', row.get('id', None))
    else:
        return None

def jaccard_similarity(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    if union == 0:
        return 0.0
    return intersection / union

def save_dis_sim(kgfile, sim_file):
    kg_data = pd.read_csv(kgfile, low_memory=False)

    # ── Node map: _id → node identifier ──────────────────────────────────────
    nodes = kg_data[kg_data['_start'].isna()].copy()
    id_to_nodeid = dict(zip(nodes['_id'], nodes['id']))

    # ── Disease nodes ─────────────────────────────────────────────────────────
    disease_nodes = nodes[nodes['_labels'].isin([':Disease'])].copy()
    id_map = pd.DataFrame({
        'id'       : disease_nodes['_id'],
        'name'     : disease_nodes['name'],
        'mapped_id': disease_nodes.apply(map_id, axis=1),
    })
    diseases = id_map['mapped_id'].dropna().unique()
    print(f'Diseases found: {len(diseases)}')

    # ── PATCHED: Build disease→gene map from KG associated_with edges ─────────
    # Original: loaded three DisGeNET CSV files from hardcoded Windows paths
    # Adapted:  uses associated_with edges already in the KG
    edges = kg_data[kg_data['_start'].notna()].copy()
    assoc_edges = edges[edges['type'] == 'associated_with'].copy()

    disease_id_map = dict(zip(disease_nodes['_id'], disease_nodes.apply(map_id, axis=1)))

    disease_genes = {}
    for _, row in assoc_edges.iterrows():
        dis_id  = disease_id_map.get(row['_start'])
        gene_id = id_to_nodeid.get(row['_end'])
        if dis_id and gene_id:
            disease_genes.setdefault(dis_id, []).append(str(gene_id))

    n_with_genes = sum(1 for v in disease_genes.values() if len(v) > 0)
    print(f'Diseases with at least 1 gene: {n_with_genes}/{len(diseases)}')

    # ── Compute pairwise Jaccard similarity ───────────────────────────────────
    print('Calculating Jaccard similarities...')
    disease_list = [d for d in diseases if d in disease_genes]
    data = []

    # PATCHED: initialise sim_graph from scratch (not reading a pre-existing file)
    # Original tried to read sim_file before it existed, causing FileNotFoundError
    count = 0

    for i in range(len(disease_list)):
        d1 = disease_list[i]
        for j in range(i + 1, len(disease_list)):
            d2 = disease_list[j]
            sim = jaccard_similarity(disease_genes[d1], disease_genes[d2])
            if sim > 0.4:
                # Format: node1, node2, edgetype, weight, index
                data.append([d1, d2, 1, sim, count])
                data.append([d2, d1, 1, sim, count + 1])
                count += 2

    print(f'Disease similarity pairs (Jaccard > 0.4): {count // 2}')

    sim_df = pd.DataFrame(data, columns=['a', 'b', 'c', 'd', 'e'])
    sim_df.to_csv(sim_file, sep='\t', index=False, header=False)
    print(f'Disease similarity saved to: {sim_file}')
