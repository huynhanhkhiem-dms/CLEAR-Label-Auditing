"""Reproducible offline validation on real features with *synthetically injected* label noise.

NOT a PMLB or CIFAR-10N replication; NOT naturally occurring annotation errors.
Uses only verified labels in certification; true injected errors remain evaluation-only.
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn import __version__ as sklearn_version
from clear_certification import fixed_rank_strata, stratified_draw, simultaneous_certificates, largest_certified

SEED=20261008
NOISE=(0.20, 0.35)
N_REPEATS=30
Q=0.10
DELTA=0.05
BUDGET_FRACTION=0.30
CUT_FRACTIONS=(0.10,0.20,0.30,0.40,0.50)
STRAT_WEIGHTS={
    'uniform_srs':None,
    'balanced_strata':(0.20,0.16,0.14,0.12,0.11,0.10,0.09,0.08),
    'rank_heavy':(0.35,0.25,0.12,0.09,0.07,0.05,0.04,0.03),
}

def allocate(weights, group_sizes, budget):
    sizes=np.asarray(group_sizes,dtype=int)
    if budget>sizes.sum():raise ValueError('impossible budget')
    if weights is None:return [budget]
    w=np.asarray(weights,dtype=float)
    raw=np.asarray(w/w.sum()*budget)
    counts=np.minimum(np.floor(raw).astype(int),sizes)
    while counts.sum()<budget:
        valid=np.flatnonzero(counts<sizes)
        # deterministic, score-only allocation; no access to labels
        priority=raw[valid]-counts[valid]
        k=int(valid[np.argmax(priority)])
        counts[k]+=1
    return counts.tolist()

def noisy_copy(y, frac, rng):
    y=np.asarray(y,dtype=int)
    classes=np.unique(y)
    num=len(classes)
    positions=rng.choice(len(y),size=round(frac*len(y)),replace=False)
    err=np.zeros(len(y),dtype=int)
    err[positions]=1
    yc=y.copy()
    for i in positions:
        other=classes[classes!=y[i]]
        yc[i]=rng.choice(other)
    assert np.all((yc!=y)==(err==1))
    return yc,err

def scores_out_of_fold(X, y_observed):
    clf=RandomForestClassifier(n_estimators=60,min_samples_leaf=1,max_features='sqrt',
                               random_state=SEED,n_jobs=2)
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
    prob=cross_val_predict(clf,X,y_observed,cv=cv,method='predict_proba',n_jobs=1)
    return 1.-prob[np.arange(len(y_observed)),y_observed]

def evaluate_dataset(name,X,y,noise,repeats,out):
    rng=np.random.default_rng(SEED+int(1000*noise)+len(y))
    y_obs,error=noisy_copy(y,noise,rng)
    score=scores_out_of_fold(X,y_obs)
    auc=roc_auc_score(error,score)
    N=len(y)
    budget=round(N*BUDGET_FRACTION)
    _,strata=fixed_rank_strata(score,8)
    sizes=[int(np.sum(strata==h)) for h in range(8)]
    cuts=sorted(set(max(1,round(N*f)) for f in CUT_FRACTIONS))
    for rep in range(repeats):
        for di, (method,weights) in enumerate(STRAT_WEIGHTS.items()):
            trial_rng=np.random.default_rng(SEED+100000*int(noise*100)+100*rep+di+13*N)
            if weights is None:
                st=np.zeros(N,dtype=int)
                allocation=[budget]
            else:
                st=strata
                allocation=allocate(weights,sizes,budget)
            verified=stratified_draw(st,allocation,trial_rng)
            certs=simultaneous_certificates(score,verified,error[verified],st,cuts,delta=DELTA)
            chosen=largest_certified(certs,q=Q)
            selected=(chosen.selected_unverified if chosen is not None
                      else np.empty(0,dtype=int))
            coverage_violation=any(int(np.sum(error[c.selected_unverified]==0))>c.upper_false
                                   for c in certs)
            false=int(np.sum(error[selected]==0))
            out.append(dict(dataset=name,n_records=N,n_classes=len(np.unique(y)),
               noise_fraction=noise,actual_errors=int(error.sum()),score_auc=round(float(auc),6),
               replicate=rep,design=method,budget=budget,selected_n=int(len(selected)),
               returned_n=int(len(selected)),n_verified_errors=int(error[verified].sum()),
               total_unique_errors=int(error[verified].sum()+error[selected].sum()),
               certified=int(len(selected)>0),
               false_n=false,actual_fdp=float(false/max(len(selected),1)),
               bound_violated=int(coverage_violation),
               selected_cut=int(chosen.cut) if chosen else 0))
    return {"name":name,"n":N,"n_classes":int(len(np.unique(y))),
            "noise_rate":noise,"errors":int(error.sum()),"score_auc":auc}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--repeats',type=int,default=N_REPEATS)
    parser.add_argument('--outdir',type=str,default='real_data_validation')
    args=parser.parse_args()
    dest=Path(args.outdir);dest.mkdir(parents=True,exist_ok=True)
    datasets={'iris':load_iris,'wine':load_wine,'breast_cancer':load_breast_cancer,'digits':load_digits}
    out=[];meta=[]
    for name,loader in datasets.items():
        bunch=loader()
        X=np.asarray(bunch.data)
        y=np.asarray(bunch.target)
        for noise in NOISE:
            a=evaluate_dataset(name,X,y,noise,args.repeats,out)
            meta.append(a)
            print('Finished',name,noise,'N=',a['n'],'score_AUC=',round(a['score_auc'],3),flush=True)
    with (dest/'replicates.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(out[0]));writer.writeheader();writer.writerows(out)
    summary=[]
    for name in datasets:
        for noise in NOISE:
            for design in STRAT_WEIGHTS:
                group=[r for r in out if r['dataset']==name and r['noise_fraction']==noise and r['design']==design]
                summary.append(dict(dataset=name,noise=noise,design=design,
                   n_reps=len(group),verified_budget=group[0]['budget'],
                   suspicion_auc=group[0]['score_auc'],
                   cert_rate=float(np.mean([r['certified'] for r in group])),
                   mean_returned=float(np.mean([r['returned_n'] for r in group])),
                   mean_verified_errors=float(np.mean([r['n_verified_errors'] for r in group])),
                   mean_total_found=float(np.mean([r['total_unique_errors'] for r in group])),
                   maximum_observed_fdp=float(max(r['actual_fdp'] for r in group)),
                   simultaneous_bound_failures=int(sum(r['bound_violated'] for r in group))))
    with (dest/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    (dest/'provenance.json').write_text(json.dumps(dict(
       experiment_type='Real observed feature datasets; synthetic replacement-label noise',
       seed=SEED,sklearn_version=sklearn_version,numpy_version=np.__version__,
       budget_fraction=BUDGET_FRACTION,repeats=args.repeats,candidate_cut_fractions=CUT_FRACTIONS,
       delta=DELTA,certified_fdp_q=Q,noise_rates=NOISE,
       results_qualifications='Not naturally mislabeled PMLB or CIFAR-10N; OOF training uses only noisy labels. Test truth is used only to evaluate risk.',
       datasets=meta,allocations=STRAT_WEIGHTS),indent=2))
    print('SUMMARY')
    for s in summary:print(s)
if __name__=='__main__':main()
