"""
Adapted calculate_drug_scores.py for the AOP Rare Disease KG.

Original: find_candidates() returned top drug candidates for a query disease
          using Drug-Disease embedding differences scored by XGBoost models.

Adapted:  find_aop_candidates() returns top AOP event candidates for a query
          disease using Disease-AOP embedding differences scored by XGBoost
          models trained on disease-AOP pairs.
"""
import argparse
import os
import pickle
import pandas as pd
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--knowledge_graph_file', type=str, required=True)
    parser.add_argument('--embeddingf', type=str, required=True)
    parser.add_argument('--model_folder', type=str, required=True)
    parser.add_argument('--query_disease', type=str, required=True)
    parser.add_argument('--candidates_count', type=int, default=10)

    args = parser.parse_args()
    return {
        'kgfile'          : args.knowledge_graph_file,
        'embeddingf'      : args.embeddingf,
        'model_folder'    : args.model_folder,
        'query_disease'   : args.query_disease,
        'candidates_count': args.candidates_count,
    }

# PATCHED: map_id now handles Disease, Protein, Pathway, AOP
# Original only handled Drug and Disease
def map_id(row):
    label = row.get('_labels', '')
    if label == ':Disease':
        return row.get('diseaseID', row.get('id', None))
    elif label == ':Protein':
        ensembl = row.get('Ensembl', None)
        if pd.notna(ensembl) and ensembl != '':
            return ensembl
        ncbi = row.get('ncbiID', None)
        return str(int(ncbi)) if pd.notna(ncbi) else row.get('id', None)
    elif label == ':Pathway':
        return row.get('PathwayID', row.get('id', None))
    elif label == ':AOP':
        return row.get('aopID', row.get('id', None))
    return None

# PATCHED: process_aop_events replaces process_drugs
# Scores AOP events for a query disease using XGBoost models
def process_aop_events(aop_list, query_disease, embeddingf, model_list,
                       aop_name_dict, candidates_count):
    with open(embeddingf, 'rb') as fin:
        embedding_dict = pickle.load(fin)

    if query_disease not in embedding_dict:
        print(f'Warning: query disease {query_disease} not found in embeddings.')
        return pd.DataFrame()

    result_df = pd.DataFrame(columns=['aop_id', 'aop_name', 'avg_prob'])

    for aop in aop_list:
        if aop not in embedding_dict:
            continue

        prob_sum = 0
        n_models = len(model_list)
        for model in model_list:
            feature = embedding_dict[query_disease] - embedding_dict[aop]
            prob_sum += model.predict_proba([feature])[:, 1]

        prob_avg = prob_sum / n_models if n_models > 0 else 0
        result_df.loc[len(result_df) + 1] = {
            'aop_id'  : aop,
            'aop_name': aop_name_dict.get(aop, aop),
            'avg_prob': float(prob_avg),
        }

    candidates = result_df.sort_values(by='avg_prob', ascending=False).head(candidates_count)
    return candidates.reset_index(drop=True)

# PATCHED: find_aop_candidates replaces find_candidates
def find_aop_candidates(kgfile:str, embeddingf:str, model_folder:str,
                        query_disease:str, candidates_count:int=10):
    kg_data = pd.read_csv(kgfile, low_memory=False)

    nodes = kg_data[kg_data['_start'].isna()].copy()

    # PATCHED: filter for AOP nodes instead of Drug nodes
    aop_nodes = nodes[nodes['_labels'] == ':AOP'].copy()
    aop_nodes['mapped_id'] = aop_nodes.apply(map_id, axis=1)

    # PATCHED: name dict for AOP (id → name)
    aop_name_dict = dict(zip(aop_nodes['mapped_id'], aop_nodes['name']))

    aop_array = aop_nodes['mapped_id'].dropna().values

    # PATCHED: find AOP events already linked to the query disease
    # Original: found drugs already associated via INDICATES edges
    # Adapted:  finds AOP events reachable via disease→gene→AOP path
    edges = kg_data[kg_data['_start'].notna()].copy()
    id_to_nodeid = dict(zip(nodes['_id'], nodes.apply(map_id, axis=1)))

    # Disease → gene edges
    assoc = edges[edges['type'] == 'associated_with']
    disease_genes = set()
    for _, row in assoc.iterrows():
        if id_to_nodeid.get(row['_start']) == query_disease:
            disease_genes.add(row['_end'])   # _id of gene

    # Gene → AOP edges (involves_gene)
    involves = edges[edges['type'] == 'involves_gene']
    linked_aops = set()
    for _, row in involves.iterrows():
        if row['_start'] in disease_genes:
            aop_id = id_to_nodeid.get(row['_end'])
            if aop_id:
                linked_aops.add(aop_id)

    # Score all AOP events (including already linked ones for reference)
    aop_to_process = list(aop_array)  # score all 6 AOP events

    # Load all classifier models from model_folder
    model_files = sorted([
        f for f in os.listdir(model_folder)
        if f.endswith('.pkl') and f.startswith('clf')
    ])
    if not model_files:
        print(f'No clf*.pkl models found in {model_folder}')
        return pd.DataFrame()

    model_list = []
    for fname in model_files:
        fpath = os.path.join(model_folder, fname)
        with open(fpath, 'rb') as f:
            model_list.append(pickle.load(f))
    print(f'Loaded {len(model_list)} models from {model_folder}')

    candidates = process_aop_events(
        aop_to_process, query_disease, embeddingf,
        model_list, aop_name_dict, candidates_count,
    )

    # Add column indicating whether the AOP was already linked via genes
    candidates['already_linked'] = candidates['aop_id'].isin(linked_aops)

    print(f'\nTop AOP candidates for disease: {query_disease}')
    print(candidates.to_string(index=False))
    return candidates

# Kept for backward compatibility
def find_candidates(kgfile, embeddingf, model_folder, query_disease, candidates_count=10):
    return find_aop_candidates(kgfile, embeddingf, model_folder, query_disease, candidates_count)

if __name__ == '__main__':
    args = parse_args()
    results = find_aop_candidates(**args)
    results.to_csv('aop_candidates.csv', index=False)
    print('Saved to aop_candidates.csv')
