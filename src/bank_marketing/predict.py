"""Batch scoring of a client/history CSV using the locally regenerated model."""
from pathlib import Path
import argparse,json
import pandas as pd
import joblib
from .core import RAW_FEATURES,top_k_mask

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--input',type=Path,required=True); parser.add_argument('--output',type=Path,required=True); parser.add_argument('--budget',type=float,default=.1); args=parser.parse_args()
    root=Path(__file__).resolve().parents[2]
    model=joblib.load(root/'models/bank_marketing.joblib')
    clients=pd.read_csv(args.input,sep=None,engine='python')
    missing=set(RAW_FEATURES)-set(clients.columns)
    if missing: raise ValueError(f'Missing required columns: {sorted(missing)}')
    p=model.predict_proba(clients[RAW_FEATURES])[:,1]
    output=pd.DataFrame({'input_row':range(len(clients)),'subscription_score':p,'select_under_budget':top_k_mask(p,args.budget).astype(int)})
    args.output.parent.mkdir(parents=True,exist_ok=True); output.to_csv(args.output,index=False)
    print(f'Scored {len(clients)} rows; output={args.output}')

if __name__=='__main__': main()
