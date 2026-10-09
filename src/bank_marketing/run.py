"""One command builds the analytical evidence and local model artifact."""
from __future__ import annotations
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/tfm-matplotlib')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import argparse, hashlib, json, platform, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.metrics import average_precision_score, roc_curve, precision_recall_curve, ConfusionMatrixDisplay, confusion_matrix, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
import sklearn, scipy, joblib
from .core import *

plt.rcParams.update({'figure.dpi': 130, 'savefig.dpi': 150, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': .15, 'font.size': 10})
COLORS = ['#194b68', '#df8542', '#4d8d79', '#985681']

def savefig(fig, path):
    fig.tight_layout(); fig.savefig(path, bbox_inches='tight'); plt.close(fig)

def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=lambda v: int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else str(v))+'\n',encoding='utf-8')

def bootstrap(y,p,repetitions=250):
    """Row bootstrap is descriptive; dependence may make intervals too narrow."""
    rng=np.random.default_rng(SEED); rows=[]
    for _ in range(repetitions):
        ix=rng.integers(0,len(y),len(y))
        if len(np.unique(y[ix]))<2: continue
        stats=evaluate(y[ix],p[ix]); rows.append({key:stats[key] for key in ['roc_auc','average_precision','brier','precision_at_10pct','lift_at_10pct']})
    frame=pd.DataFrame(rows)
    return pd.DataFrame({'metric':frame.columns,'lower_95':frame.quantile(.025).values,'upper_95':frame.quantile(.975).values,'method':'row bootstrap; conditional/descriptive','replications':len(frame)})

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]); args=parser.parse_args(); root=args.root.resolve()
    figures=root/'reports/figures'; tables=root/'reports/tables'; models=root/'models'
    for folder in [figures,tables,models]: folder.mkdir(parents=True,exist_ok=True)
    path=root/'data/raw/bank-additional-full.csv'
    if not path.exists():
        archive=path.with_suffix(path.suffix+'.gz')
        if not archive.exists(): raise FileNotFoundError('Source missing: see data/DATA_SOURCES.md and scripts/download_data.py.')
        import gzip
        path.write_bytes(gzip.decompress(archive.read_bytes()))
    df=pd.read_csv(path,sep=';')
    expected=RAW_FEATURES + ['duration','campaign','contact','month','day_of_week','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed','y']
    if len(df)!=41188 or set(df.columns)!=set(expected): raise ValueError('Unexpected source schema/row count.')
    if set(df.y.unique())!={'yes','no'}: raise ValueError('Invalid target labels.')
    if not df.age.between(0,120).all() or not df.previous.ge(0).all(): raise ValueError('Invalid client data ranges.')
    y=df.y.eq('yes').astype(int); X=df.drop(columns='y')
    splits=chronological_splits(len(df))
    summaries=[{'partition':name,'start_source_row':int(ix.min()),'end_source_row':int(ix.max()),'n':len(ix),'positive_n':int(y.iloc[ix].sum()),'prevalence':float(y.iloc[ix].mean())} for name,ix in splits.items()]
    pd.DataFrame(summaries).to_csv(tables/'partitions.csv',index=False)
    unknown=pd.Series({col:int(df[col].eq('unknown').sum()) for col in df.select_dtypes(include='object') if col!='y'},name='unknown_n').to_frame(); unknown['unknown_pct']=unknown.unknown_n/len(df)*100; unknown.to_csv(tables/'unknown_values.csv')
    df.describe(include='all').T.to_csv(tables/'data_dictionary_statistics.csv')
    quality={'source_file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'rows':len(df),'columns':len(df.columns),'null_cells':int(df.isna().sum().sum()),'duplicate_full_rows':int(df.duplicated().sum()),'positive_n':int(y.sum()),'prevalence':float(y.mean()),'duplicates_action':'Preserved: no customer ID; identical attributes may represent distinct observations. Original order retained.','unknown_cells':int(unknown.unknown_n.sum())}
    write_json(root/'reports/data_quality.json',quality)
    fig,axes=plt.subplots(1,3,figsize=(13,3.8)); axes[0].bar(['No suscribe','Suscribe'],[int((y==0).sum()),int(y.sum())],color=COLORS[:2]); axes[0].set_title('Objetivo: depósito a plazo'); axes[0].set_ylabel('Observaciones')
    block=pd.DataFrame({'block':np.minimum(np.arange(len(df))*20//len(df),19),'y':y}).groupby('block').y.mean(); axes[1].plot(block.index+1,block.values*100,'o-',color=COLORS[0]); axes[1].set_title('La prevalencia cambia con el orden'); axes[1].set_xlabel('Bloques cronológicos (5% de filas)'); axes[1].set_ylabel('Suscripciones (%)')
    axes[2].hist(df.age,bins=np.arange(15,102,4),color=COLORS[2]); axes[2].set_title('Edad: dispersión y cola derecha'); axes[2].set_xlabel('Edad'); savefig(fig,figures/'01_data_overview.png')
    month=df.groupby('month').y.agg(n='size',conversion=lambda v:v.eq('yes').mean()); month.to_csv(tables/'monthly_descriptive.csv')
    fig,ax=plt.subplots(figsize=(8,4)); unknown.unknown_pct.sort_values().plot.barh(ax=ax,color=COLORS[0]); ax.set_xlabel('Porcentaje de valores unknown'); ax.set_title('Ausencias codificadas: no confundir con NaN'); savefig(fig,figures/'02_unknown_values.png')
    candidates={
        'dummy_prior':DummyClassifier(strategy='prior'),
        'logistic_C0.1':LogisticRegression(C=.1,max_iter=1200,random_state=SEED),
        'logistic_C1':LogisticRegression(C=1.,max_iter=1200,random_state=SEED),
        'random_forest_leaf10':RandomForestClassifier(n_estimators=180,min_samples_leaf=10,max_features=.7,n_jobs=2,random_state=SEED),
        'random_forest_leaf30':RandomForestClassifier(n_estimators=180,min_samples_leaf=30,max_features=.7,n_jobs=2,random_state=SEED),
        'hist_gradient_leaf15':HistGradientBoostingClassifier(max_iter=160,max_leaf_nodes=15,l2_regularization=2.,learning_rate=.07,early_stopping=False,random_state=SEED),
        'hist_gradient_leaf31':HistGradientBoostingClassifier(max_iter=160,max_leaf_nodes=31,l2_regularization=5.,learning_rate=.07,early_stopping=False,random_state=SEED)}
    fitted={}; rows=[]; train=splits['train']; selection=splits['selection']
    for name,estimator in candidates.items():
        start=time.perf_counter(); m=pipeline(estimator); m.fit(X.iloc[train],y.iloc[train]); prob=m.predict_proba(X.iloc[selection])[:,1]
        fitted[name]=m; metrics=evaluate(y.iloc[selection],prob)
        if name=='dummy_prior': metrics.update(precision_at_10pct=float(y.iloc[selection].mean()), recall_at_10pct=int(top_k_mask(prob).sum())/len(prob), lift_at_10pct=1.)
        rows.append({'model':name,**metrics,'fit_seconds':time.perf_counter()-start}); print(f'Validated {name}: AP={rows[-1]["average_precision"]:.4f}',flush=True)
    selection_results=pd.DataFrame(rows).sort_values('average_precision',ascending=False); selection_results.to_csv(tables/'model_selection.csv',index=False)
    winner=selection_results.loc[selection_results.model!='dummy_prior'].iloc[0].model
    # Forward-chaining diagnostic only; never uses the final test.
    from sklearn.base import clone
    folds=[]; pretest=np.concatenate([train,selection])
    for fold,(a,b) in enumerate(TimeSeriesSplit(n_splits=3).split(pretest),start=1):
        m=clone(fitted[winner]); m.fit(X.iloc[pretest[a]],y.iloc[pretest[a]]); p=m.predict_proba(X.iloc[pretest[b]])[:,1]
        folds.append({'fold':fold,'train_n':len(a),'validation_n':len(b),'train_end_row':int(pretest[a[-1]]),'validation_start_row':int(pretest[b[0]]),'validation_end_row':int(pretest[b[-1]]),**evaluate(y.iloc[pretest[b]],p)})
    pd.DataFrame(folds).to_csv(tables/'forward_validation.csv',index=False)
    winner_model=clone(fitted[winner]); winner_model.fit(X.iloc[pretest],y.iloc[pretest])
    calibration=splits['calibration']; yc=y.iloc[calibration].to_numpy()
    calibrated=CalibratedClassifierCV(FrozenEstimator(winner_model),method='sigmoid'); calibrated.fit(X.iloc[calibration],yc)
    pc=calibrated.predict_proba(X.iloc[calibration])[:,1]
    thresholds=np.linspace(.01,.75,150); benefits=100.; cost=5.
    curve=pd.DataFrame({'threshold':thresholds,'proxy_profit':[proxy_profit(yc,pc>=t,benefits,cost) for t in thresholds],'contact_rate':[(pc>=t).mean() for t in thresholds]}); curve.to_csv(tables/'threshold_selection.csv',index=False)
    best_threshold=float(curve.sort_values(['proxy_profit','threshold'],ascending=[False,False]).iloc[0].threshold)
    # All design choices are now fixed. Read the held-out test for evaluation.
    test=splits['test']; yt=y.iloc[test].to_numpy(); Xt=X.iloc[test]; p=calibrated.predict_proba(Xt)[:,1]; uncal=winner_model.predict_proba(Xt)[:,1]
    test_rows=[]
    for name,m in fitted.items():
        metrics=evaluate(yt,m.predict_proba(Xt)[:,1])
        if name=='dummy_prior': metrics.update(precision_at_10pct=float(yt.mean()), recall_at_10pct=int(top_k_mask(p).sum())/len(p), lift_at_10pct=1.)
        test_rows.append({'model':name,'training_scope':'initial 60%; benchmark only',**metrics})
    test_rows.extend([{'model':winner+'_refit_uncalibrated','training_scope':'first 70%',**evaluate(yt,uncal,.5)}, {'model':winner+'_refit_sigmoid','training_scope':'first 70%; calibration next 10%',**evaluate(yt,p,best_threshold)}])
    pd.DataFrame(test_rows).to_csv(tables/'test_metrics.csv',index=False)
    intervals=bootstrap(yt,p); intervals.to_csv(tables/'bootstrap_intervals.csv',index=False)
    selected=top_k_mask(p,.10); policy=[]
    for name,mask in [('Nadie',np.zeros(len(yt),bool)),('Todos',np.ones(len(yt),bool)),('Top 10% (presupuesto fijo)',selected),('Umbral validado',p>=best_threshold),('Umbral teórico coste/beneficio',p>=cost/benefits)]:
        policy.append({'policy':name,'contacts':int(mask.sum()),'observed_acceptances_in_selected':int(yt[mask].sum()),'precision_selected':float(yt[mask].mean()) if mask.sum() else 0.,'proxy_profit':proxy_profit(yt,mask,benefits,cost),'benefit_assumption':benefits,'cost_assumption':cost})
    k=int(selected.sum()); policy.append({'policy':'Aleatorio 10% (valor esperado)','contacts':k,'observed_acceptances_in_selected':k*float(yt.mean()),'precision_selected':float(yt.mean()),'proxy_profit':k*(benefits*float(yt.mean())-cost),'benefit_assumption':benefits,'cost_assumption':cost})
    policy=pd.DataFrame(policy); policy.to_csv(tables/'business_scenarios.csv',index=False)
    sensitivity=[]
    for benefit in [25,50,100,150,200]:
        for contact_cost in [2,5,10,20]:
            sensitivity.append({'benefit':benefit,'contact_cost':contact_cost,'top_10pct_proxy_profit':proxy_profit(yt,selected,benefit,contact_cost),'random_10pct_expected_proxy_profit':k*(benefit*yt.mean()-contact_cost),'all_proxy_profit':proxy_profit(yt,np.ones(len(yt),bool),benefit,contact_cost)})
    pd.DataFrame(sensitivity).to_csv(tables/'cost_benefit_sensitivity.csv',index=False)
    pd.DataFrame({'source_row':test,'y_true':yt,'probability':p,'contact_top10':selected.astype(int),'contact_threshold':(p>=best_threshold).astype(int)}).to_csv(tables/'test_predictions.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(14,4)); fpr,tpr,_=roc_curve(yt,p); axes[0].plot(fpr,tpr,label=f'AUC {roc_auc_score(yt,p):.3f}',color=COLORS[0]); axes[0].plot([0,1],[0,1],'--',color='gray'); axes[0].set(xlabel='Falsos positivos',ylabel='Verdaderos positivos',title='ROC: prueba cronológica'); axes[0].legend()
    prec,rec,_=precision_recall_curve(yt,p); axes[1].plot(rec,prec,color=COLORS[0],label=f'AP {average_precision_score(yt,p):.3f}'); axes[1].axhline(yt.mean(),ls='--',color=COLORS[1],label='Prevalencia'); axes[1].set(xlabel='Recall',ylabel='Precision',title='Precision–recall'); axes[1].legend()
    for name,prob,color in [('Sin calibrar',uncal,COLORS[1]),('Sigmoide',p,COLORS[0])]:
        observed,mean=calibration_curve(yt,prob,n_bins=8,strategy='quantile'); axes[2].plot(mean,observed,'o-',label=name,color=color)
    axes[2].plot([0,1],[0,1],'--',color='gray'); axes[2].set(xlabel='Probabilidad media',ylabel='Tasa observada',title='Calibración fuera de tiempo'); axes[2].legend(); savefig(fig,figures/'03_model_evaluation.png')
    sorted_y=yt[np.argsort(-p,kind='stable')]; fractions=np.arange(1,len(yt)+1)/len(yt); capture=np.cumsum(sorted_y)/yt.sum()
    fig,axes=plt.subplots(1,2,figsize=(11,4)); axes[0].plot(fractions,capture,color=COLORS[0],label='Modelo'); axes[0].plot([0,1],[0,1],'--',color=COLORS[1],label='Aleatorio esperado'); axes[0].axvline(.1,ls=':',color='gray'); axes[0].set(xlabel='Fracción de contactos',ylabel='Fracción de suscripciones observadas',title='Curva de ganancias (asociación)'); axes[0].legend()
    axes[1].barh(policy.policy,policy.proxy_profit,color=COLORS[0]); from matplotlib.ticker import MaxNLocator, FuncFormatter
    axes[1].xaxis.set_major_locator(MaxNLocator(5)); axes[1].xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value/1000:.0f} mil')); axes[1].set_xlabel('Unidades monetarias hipotéticas'); axes[1].set_title('Contabilidad de escenarios: beneficio 100, coste 5'); savefig(fig,figures/'04_policy_scenarios.png')
    fig,ax=plt.subplots(figsize=(5,4)); ConfusionMatrixDisplay(confusion_matrix(yt,p>=best_threshold),display_labels=['No','Sí']).plot(ax=ax,colorbar=False,cmap='Blues'); ax.set_title(f'Umbral decidido en calibración: {best_threshold:.3f}'); savefig(fig,figures/'05_confusion_matrix.png')
    importance=permutation_importance(calibrated,Xt[RAW_FEATURES],yt,n_repeats=5,random_state=SEED,scoring='average_precision',n_jobs=1)
    imp=pd.DataFrame({'feature':RAW_FEATURES,'mean_ap_drop':importance.importances_mean,'std_ap_drop':importance.importances_std}).sort_values('mean_ap_drop',ascending=False); imp.to_csv(tables/'permutation_importance.csv',index=False)
    fig,ax=plt.subplots(figsize=(8,4.5)); ax.barh(imp.feature[::-1],imp.mean_ap_drop[::-1],xerr=imp.std_ap_drop[::-1],color=COLORS[0]); ax.set(xlabel='Caída de average precision al permutar',title='Importancia predictiva: no causal'); savefig(fig,figures/'06_permutation_importance.png')
    fig,axes=plt.subplots(1,2,figsize=(10,4)); PartialDependenceDisplay.from_estimator(calibrated,Xt.astype({'age':float,'previous':float}),features=['age','previous'],kind='average',grid_resolution=25,method='brute',ax=axes); axes[0].set_title('Edad'); axes[1].set_title('Contactos previos'); fig.suptitle('Dependencia parcial: asociación y combinaciones sintéticas'); savefig(fig,figures/'07_partial_dependence.png')
    audit=pd.DataFrame({'age_group':pd.cut(Xt.age,[0,29,44,59,120],labels=['<30','30–44','45–59','60+']),'y':yt,'p':p,'selected':selected})
    audit_rows=[]
    for group,sub in audit.groupby('age_group',observed=True):
        audit_rows.append({'age_group':str(group),'n':len(sub),'observed_prevalence':sub.y.mean(),'selection_rate':sub.selected.mean(),'precision_selected':sub.loc[sub.selected,'y'].mean() if sub.selected.any() else None,'recall_selected':sub.loc[sub.selected,'y'].sum()/max(sub.y.sum(),1),'brier':((sub.y-sub.p)**2).mean()})
    pd.DataFrame(audit_rows).to_csv(tables/'age_group_audit.csv',index=False)
    metadata={'project':'Pre-contact bank marketing prioritization','selected_model':winner,'seed':SEED,'features_used':RAW_FEATURES,'excluded_features':FORBIDDEN,'split_protocol':'60% fit / 10% selection / 10% sigmoid calibration and policy / 20% chronological test; original source row order','selected_threshold':best_threshold,'threshold_selection':'maximum historical accounting proxy on calibration segment; benefit=100 cost=5','test':evaluate(yt,p,best_threshold),'test_uncalibrated_brier':float(evaluate(yt,uncal,.5)['brier']),'test_oracle_prevalence_brier':float(yt.mean()*(1-yt.mean())),'calibration_prior_brier_on_test':float(np.mean((yt-yc.mean())**2)),'dummy_ranking_note':'For constant dummy scores, top-10% statistics report random selection expectations, avoiding source-order tie artefacts.','data':quality,'versions':{'python':platform.python_version(),'scikit_learn':sklearn.__version__,'pandas':pd.__version__,'numpy':np.__version__,'scipy':scipy.__version__},'bootstrap_caveat':'Row bootstrap assumes independent rows; repeated clients/date dependence unknown. Conditional/descriptive intervals, not rigorous deployment assurance.','calibration_note':'Sigmoid method selected in advance; no switching based on test results. Calibration and threshold use dedicated pretest block.','production_note':'Retrospective benchmark. Requires recent prospectively logged client IDs, consent, publication timing, A/B test and monitoring before any deployment.'}
    write_json(root/'reports/results.json',metadata)
    joblib.dump(calibrated,models/'bank_marketing.joblib')
    print(json.dumps(metadata['test'],indent=2),flush=True)
    print('Completed evidence: reports/ and local models/',flush=True)

if __name__=='__main__': main()
