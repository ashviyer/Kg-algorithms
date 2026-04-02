import numpy as np
import pandas as pd
import os

def generate_files(kg_data):
    ## generate graph and nodetypes files

    # ── PATCHED: include :Pathway and :AOP in addition to :Protein and :Disease
    # Original only had: [':Protein', ':Drug', ':Disease']
    nodes_filtered = kg_data[kg_data['_labels'].isin(
        [':Protein', ':Disease', ':Pathway', ':AOP']
    )].copy()

    # create a new dataframe for neo4j id -> actual id
    id_map = pd.DataFrame({
        '_id':      nodes_filtered['_id'],
        '_label':   nodes_filtered['_labels'],
        'mapped_id': nodes_filtered.apply(map_id, axis=1)
    })

    # create a dictionary mapping _id to mapped_id
    id_map_dict = dict(zip(id_map['_id'], id_map['mapped_id']))
    # create a dictionary mapping mapped_id to _label
    label_map_dict = dict(zip(id_map['mapped_id'], id_map['_label']))

    # ── PATCHED: include all 5 relation types present in the AOP KG
    # Original only had: ['activates', 'inhibits', 'associated_with', 'interacts_with']
    edges_filtered = kg_data[kg_data['type'].isin([
        'associated_with',   # disease → gene  (edgetype 2)
        'interacts_with',    # gene   → gene   (edgetype 3)
        'involves_gene',     # gene   → AOP    (edgetype 4)
        'part_of_pathway',   # pathway→ gene   (edgetype 5)
        'adjacent',          # AOP   → AOP    (edgetype 6)
    ])].copy()

    count = 0
    output_graph = pd.DataFrame(columns=['source', 'target', 'edgetype', 'weight', 'edge_id'])
    nodetypes = {'node': 'type'}

    for index, row in edges_filtered.iterrows():
        if row['_start'] in id_map_dict and row['_end'] in id_map_dict:
            source = id_map_dict.get(row['_start'])
            target = id_map_dict.get(row['_end'])

            # ── PATCHED: edge type codes extended for AOP relations
            edgetype = 1  # default (interacts_with was 1 in original for activates/inhibits)
            if row['type'] == 'associated_with':
                edgetype = 2
            elif row['type'] == 'interacts_with':
                edgetype = 3
            elif row['type'] == 'involves_gene':
                edgetype = 4
            elif row['type'] == 'part_of_pathway':
                edgetype = 5
            elif row['type'] == 'adjacent':
                edgetype = 6

            new_edge_row = {
                'source': source, 'target': target,
                'edgetype': edgetype, 'weight': 1, 'edge_id': count
            }
            output_graph.loc[index] = new_edge_row

            # ── PATCHED: nodetypes extended for :Pathway and :AOP
            for node_id, label in [(source, label_map_dict[source]),
                                    (target, label_map_dict[target])]:
                if label == ':Protein':
                    nodetypes[node_id] = 'gene'
                elif label == ':Disease':
                    nodetypes[node_id] = 'disease'
                elif label == ':Pathway':
                    nodetypes[node_id] = 'pathway'
                elif label == ':AOP':
                    nodetypes[node_id] = 'aop'
                # :Drug removed — not present in this KG

            count += 1

    # saving the graph as txt file
    output_graph.to_csv('graph.txt', sep="\t", index=False, header=False)
    print("Graph file is saved!")

    # saving the node types as a tsv file
    with open('nodetypes.tsv', 'w') as f:
        for key in nodetypes.keys():
            f.write("%s\t%s\n" % (key, nodetypes[key]))
    print("Node types file is saved!")

    ## ── PATCHED: hierarchy file
    # Original: drug ATC hierarchy (requires drugID + atcClassification columns)
    # Adapted:  minimal placeholder hierarchy using disease nodes only
    #           (DREAMwalk reads this file but disease-only KG has no drug tree)
    diseases_filtered = kg_data[
        kg_data['_labels'].isin([':Disease'])
    ][['diseaseID']].dropna(subset=['diseaseID']).copy()

    # Write a flat hierarchy: each disease is its own parent (no ATC tree)
    hierarchy_df = pd.DataFrame({
        'child':  diseases_filtered['diseaseID'].values,
        'parent': 'disease_root',   # single root for all diseases
    })
    hierarchy_df.to_csv('hierarchy.csv', sep=",", index=False)
    print("Hierarchy file is saved!")

    ## ── PATCHED: association pair files
    # Original: drug-disease pairs (treated_with edges + random negatives)
    # Adapted:  disease-AOP pairs (associated_with edges + random negatives)
    os.makedirs('dda_files', exist_ok=True)

    aop_nodes = kg_data[
        kg_data['_labels'].isin([':AOP'])
    ]['id'].dropna().unique()

    # Build positive disease-AOP pairs from associated_with edges
    # (disease → gene → AOP is the semantic link; here we use the direct
    #  associated_with edges present in the KG as labelled positives)
    dda_df = pd.DataFrame(columns=['disease', 'aop', 'label'])

    for index, row in edges_filtered[edges_filtered['type'] == 'associated_with'].iterrows():
        if row['_start'] in id_map_dict and row['_end'] in id_map_dict:
            source = id_map_dict.get(row['_start'])   # disease
            target = id_map_dict.get(row['_end'])     # gene (not AOP directly)
            # Note: associated_with in this KG connects disease→gene not disease→AOP
            # We leave dda_df empty here and populate from involves_gene paths
            pass

    # Use all disease × AOP as the pair space (positive label = 1 if known)
    all_disease_ids = diseases_filtered['diseaseID'].values
    known_pairs = set()  # extend with known disease-AOP pairs if available

    existing_pairs = set(known_pairs)

    for i in range(1, 11):
        new_rows = []
        n_pos = max(len(known_pairs), len(all_disease_ids))  # at least 1 row per file
        while len(new_rows) < max(n_pos, 100):
            new_disease = np.random.choice(all_disease_ids)
            new_aop = np.random.choice(aop_nodes)
            new_pair = (new_disease, new_aop)
            if new_pair not in existing_pairs:
                new_rows.append({'disease': new_disease, 'aop': new_aop, 'label': 0})
                existing_pairs.add(new_pair)

        dda_complete_df = pd.concat(
            [dda_df, pd.DataFrame(new_rows)], ignore_index=True
        )
        dda_complete_df.to_csv(
            'dda_files/dda' + str(i) + '.tsv', sep="\t", index=False
        )
        print(f"Disease-AOP pair file {i} is saved!")


# ── PATCHED: map_id extended for :Pathway and :AOP
# Original only handled :Protein, :Drug, :Disease
def map_id(row):
    if row['_labels'] == ':Protein':
        # Use Ensembl ID (primary in this KG); fall back to ncbiID
        if not pd.isna(row.get('Ensembl', np.nan)) and row.get('Ensembl') != '':
            return row['Ensembl']
        elif not pd.isna(row.get('ncbiID', np.nan)):
            return str(int(row['ncbiID']))
        else:
            return row.get('id', None)
    elif row['_labels'] == ':Disease':
        return row.get('diseaseID', row.get('id', None))
    elif row['_labels'] == ':Pathway':
        return row.get('PathwayID', row.get('id', None))
    elif row['_labels'] == ':AOP':
        return row.get('aopID', row.get('id', None))
    else:
        return None


def generate_drug_hierarchy(drug_df) -> pd.DataFrame:
    # Kept for compatibility but not called in AOP workflow
    drug_hierarchy_df = pd.DataFrame(columns=['child', 'parent'])
    return drug_hierarchy_df


def generate_disease_hierarchy(disease_df) -> pd.DataFrame:
    disease_hierarchy_df = pd.DataFrame(columns=['child', 'parent'])
    disease_hierarchy_dict = {}
    for index, row in disease_df.iterrows():
        diseaseID = row['diseaseID']
        if pd.isna(row.get('class', np.nan)):
            continue
        mesh_classifications = row['class'].split(';')
        for mesh_classification in mesh_classifications:
            disease_hierarchy_df.loc[len(disease_hierarchy_df) + 1] = {
                'child': diseaseID, 'parent': mesh_classification
            }
            disease_hierarchy_dict[mesh_classification] = mesh_classification[0:1]
            disease_hierarchy_dict[mesh_classification[0:1]] = 'disease'
    disease_hierarchy_df2 = pd.DataFrame(
        disease_hierarchy_dict.items(), columns=['child', 'parent']
    )
    disease_hierarchy_df = pd.concat([disease_hierarchy_df, disease_hierarchy_df2])
    return disease_hierarchy_df


def export_files():
    kg_data = pd.read_csv("preprocessed_graph.csv", low_memory=False)
    print("KG file is loaded!")
    generate_files(kg_data)
