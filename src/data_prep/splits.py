import numpy as np, pandas as pd

def create_splits(df, drug_a_col='drug1_id', drug_b_col='drug2_id', seed=42):
    rng = np.random.default_rng(seed)
    all_drugs = pd.unique(df[[drug_a_col, drug_b_col]].values.ravel())
    n_holdout = int(0.15*len(all_drugs))
    holdout = set(rng.choice(all_drugs, size=n_holdout, replace=False))
    both = lambda r: r[drug_a_col] in holdout and r[drug_b_col] in holdout
    one = lambda r: (r[drug_a_col] in holdout) != (r[drug_b_col] in holdout)
    none_ = lambda r: r[drug_a_col] not in holdout and r[drug_b_col] not in holdout
    s1 = df[df.apply(both, axis=1)]
    s2 = df[df.apply(one, axis=1)]
    seen = df[df.apply(none_, axis=1)].sample(frac=1, random_state=seed)
    split = int(0.8*len(seen))
    return {'transductive_train': seen.iloc[:split], 'transductive_test': seen.iloc[split:],
            's1_test': s1, 's2_test': s2}
