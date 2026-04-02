import argparse
import pickle
import numpy as np

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score

from DREAMwalk.utils import set_seed

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_file', type=str, required=True)
    parser.add_argument('--pair_file', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--model_checkpoint', type=str, default='clf.pkl')
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--valid_ratio', type=float, default=0.1)

    args = parser.parse_args()
    args = {
        'embeddingf': args.embedding_file,
        'pairf'     : args.pair_file,
        'seed'      : args.seed,
        'patience'  : args.patience,
        'modelf'    : args.model_checkpoint,
        'testr'     : args.test_ratio,
        'validr'    : args.valid_ratio,   # PATCHED: was 'validation_ratio' (typo in original)
    }
    return args

def split_dataset(pairf, embeddingf, validr, testr, seed):
    with open(embeddingf, 'rb') as fin:
        embedding_dict = pickle.load(fin)

    xs, ys = [], []
    with open(pairf, 'r') as fin:
        lines = fin.readlines()

    skipped = 0
    for line in lines[1:]:   # skip header
        line = line.strip().split('\t')
        # PATCHED: renamed drug→entity1, dis→entity2 to reflect Disease-AOP use case
        # Logic unchanged — positional TSV reading works for any entity pair
        entity1 = line[0]   # disease (or drug in original)
        entity2 = line[1]   # AOP     (or disease in original)
        label   = line[2]

        # PATCHED: skip pairs where either node has no embedding
        # Original would crash with KeyError on missing embeddings
        if entity1 not in embedding_dict or entity2 not in embedding_dict:
            skipped += 1
            continue

        xs.append(embedding_dict[entity1] - embedding_dict[entity2])
        ys.append(int(label))

    if skipped > 0:
        print(f'  Skipped {skipped} pairs with missing embeddings')

    # PATCHED: guard against all-same-label case (stratified split fails)
    # Original would crash if all labels were 0 (no known positives)
    unique_labels = set(ys)
    if len(unique_labels) < 2:
        raise ValueError(
            f'pair_file has only label(s) {unique_labels}. '
            'Need both positive (1) and negative (0) pairs to train. '
            'Add known disease–AOP positive pairs to the dda_files.'
        )

    x, y = {}, {}
    x['train'], x['test'], y['train'], y['test'] = train_test_split(
        xs, ys, test_size=testr, random_state=seed, stratify=ys)
    if validr > 0:
        x['train'], x['valid'], y['train'], y['valid'] = train_test_split(
            x['train'], y['train'],
            test_size=validr / (1 - testr),
            random_state=seed, stratify=y['train'])
    else:
        x['valid'], y['valid'] = [], []

    return x, y

def return_scores(target_list, pred_list):
    metric_list = [accuracy_score, roc_auc_score, average_precision_score, f1_score]
    scores = []
    for metric in metric_list:
        if metric in [roc_auc_score, average_precision_score]:
            scores.append(metric(target_list, pred_list))
        else:
            scores.append(metric(target_list, pred_list.round()))
    return scores

def predict_dda(embeddingf:str, pairf:str, modelf:str='clf.pkl', seed:int=42,
                validr:float=0.1, testr:float=0.1):
    set_seed(seed)
    x, y = split_dataset(pairf, embeddingf, validr, testr, seed)

    clf = XGBClassifier(
        base_score=0.5, booster='gbtree', eval_metric='error',
        objective='binary:logistic', gamma=0, learning_rate=0.1,
        max_depth=6, n_estimators=500, tree_method='auto',
        min_child_weight=4, subsample=0.8, colsample_bytree=0.9,
        scale_pos_weight=1, max_delta_step=1, seed=seed,
    )
    clf.fit(x['train'], y['train'])

    preds, scores = {}, {}
    for split in ['train', 'valid', 'test']:
        if len(x[split]) == 0:
            continue
        preds[split]  = clf.predict_proba(np.array(x[split]))[:, 1]
        scores[split] = return_scores(y[split], preds[split])
        print(f'{split.upper():5} set | Acc: {scores[split][0]*100:.2f}% | '
              f'AUROC: {scores[split][1]:.4f} | AUPR: {scores[split][2]:.4f} | '
              f'F1: {scores[split][3]:.4f}')

    import os
    os.makedirs(os.path.dirname(modelf) if os.path.dirname(modelf) else '.', exist_ok=True)
    with open(modelf, 'wb') as fw:
        pickle.dump(clf, fw)
    print(f'Saved XGBoost classifier: {modelf}')
    print('=' * 50)

if __name__ == '__main__':
    args = parse_args()
    predict_dda(**args)
