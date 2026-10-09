import numpy as np
import pandas as pd
import pytest
from bank_marketing.core import BankFeatures, RAW_FEATURES, FORBIDDEN, chronological_splits, top_k_mask, proxy_profit


def example():
    return pd.DataFrame({'age':[35,51], 'job':['admin.','unknown'], 'marital':['single','married'], 'education':['university.degree','high.school'], 'default':['no','unknown'],'housing':['yes','no'],'loan':['no','no'],'pdays':[999,4],'previous':[0,2],'poutcome':['nonexistent','success'], 'duration':[0,9999], 'campaign':[1,9], 'y':['no','yes']})

def test_feature_whitelist_blocks_future_target_and_sentinel():
    x=example(); transform=BankFeatures().fit(x); a=transform.transform(x)
    assert not set(FORBIDDEN)&set(a.columns)
    assert pd.isna(a.pdays.iloc[0]) and a.pdays.iloc[1]==4
    assert a.previously_contacted.tolist()==[0,1]
    changed=x.assign(duration=999999,campaign=999,y='yes')
    pd.testing.assert_frame_equal(a,transform.transform(changed))

def test_missing_operational_features_fail_loudly():
    with pytest.raises(ValueError): BankFeatures().fit(example().drop(columns='age'))

def test_temporal_partitions_are_exhaustive_and_do_not_overlap():
    split=chronological_splits(41188); blocks=list(split.values())
    assert np.array_equal(np.concatenate(blocks),np.arange(41188))
    assert all(a[-1]<b[0] for a,b in zip(blocks,blocks[1:]))
    assert len(split['test'])==8238

def test_fixed_budget_and_ties_are_deterministic():
    p=np.array([.8,.8,.2,.1,.3])
    assert top_k_mask(p,.2).tolist()==[True,False,False,False,False]
    assert top_k_mask(p,0).sum()==0 and top_k_mask(p,1).sum()==5

def test_accounting_counts_every_selected_contact_cost():
    assert proxy_profit([1,0,1],[True,True,False],benefit=100,cost=5)==90
    assert proxy_profit([1,1],[False,False])==0
    assert proxy_profit([0,0],[True,True])==-10

def test_imputation_is_fitted_only_on_training_and_unknown_category_is_safe():
    from bank_marketing.core import pipeline
    from sklearn.linear_model import LogisticRegression
    train=example(); test=example().assign(age=100000,job='new_category')
    model=pipeline(LogisticRegression()).fit(train,[0,1])
    imputer=model.named_steps['preprocess'].named_transformers_['numeric'].named_steps['impute']
    original=imputer.statistics_.copy()
    probabilities=model.predict_proba(test)
    assert np.array_equal(imputer.statistics_,original)
    assert original[0]==43
    assert np.isfinite(probabilities).all()

def test_bundled_source_checksum_and_shape():
    import gzip,hashlib
    from pathlib import Path
    source=Path(__file__).resolve().parents[1]/'data/raw/bank-additional-full.csv.gz'
    raw=gzip.decompress(source.read_bytes())
    assert hashlib.sha256(raw).hexdigest()=='233e260d5d1d506a2c10381da5b8c2f75f2c08d1b373ca7b825e39ebe1bd30df'
    import io
    data=pd.read_csv(io.BytesIO(raw),sep=';')
    assert data.shape==(41188,21)
