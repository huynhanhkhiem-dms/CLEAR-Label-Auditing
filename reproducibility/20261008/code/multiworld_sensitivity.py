"""Predeclared robustness: independently injected label-error *worlds*.
Each world is a different frozen finite library; draws within a world are not
independent datasets. This is a sensitivity audit, not natural label-noise data.
"""
from __future__ import annotations
import csv, json, argparse
from pathlib import Path
import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.metrics import roc_auc_score
from real_features_noise_audit import (noisy_copy, scores_out_of_fold,
          allocate, BUDGET_FRACTION, CUT_FRACTIONS, STRAT_WEIGHTS)
from clear_certification import (fixed_rank_strata, stratified_draw,
    simultaneous_certificates, largest_certified)

DATASETS={'iris':load_iris,'wine':load_wine,'breast_cancer':load_breast_cancer,'digits':load_digits}
SEED=20261008

def experiment(worlds=4,reps=12,out_dir='multiworld', dataset_names=('iris','wine','breast_cancer','digits')):
    out=[]; metadata=[]
    for dataset in dataset_names:
        loader=DATASETS[dataset]
        bunch=loader(); X=np.asarray(bunch.data); y=np.asarray(bunch.target)
        N=len(y); budget=round(BUDGET_FRACTION*N)
        cuts=sorted(set(max(1,round(N*f)) for f in CUT_FRACTIONS))
        for noise in (.20,.35):
            for world in range(worlds):
                rng=np.random.default_rng(SEED + 65437*world + 100000*int(noise*100)+N)
                y_obs,err=noisy_copy(y,noise,rng)
                score=scores_out_of_fold(X,y_obs)
                auc=float(roc_auc_score(err,score))
                _,strata=fixed_rank_strata(score,8)
                sz=[int(np.sum(strata==h)) for h in range(8)]
                for rep in range(reps):
                    for di,(design,weights) in enumerate(STRAT_WEIGHTS.items()):
                        audit_rng=np.random.default_rng(SEED+world*190001+rep*4003+di*71+N+round(noise*100)*100000)
                        if weights is None:
                            st=np.zeros(N,dtype=int); allocation=[budget]
                        else:
                            st=strata; allocation=allocate(weights,sz,budget)
                        v=stratified_draw(st,allocation,audit_rng)
                        certificates=simultaneous_certificates(score,v,err[v],st,cuts,.05)
                        selected=largest_certified(certificates,.10)
                        chosen=selected.selected_unverified if selected else np.array([],dtype=int)
                        violations=any(int(np.sum(err[c.selected_unverified]==0))>c.upper_false for c in certificates)
                        out.append(dict(dataset=dataset,noise=noise,world=world,rep=rep,
                            design=design,N=N,budget=budget,auc=auc,
                            n_returned=len(chosen),nonempty=int(len(chosen)>0),
                            verified_errors=int(np.sum(err[v])),
                            total_errors_found=int(np.sum(err[v])+np.sum(err[chosen])),
                            actual_fdp=float(np.sum(err[chosen]==0)/len(chosen)) if len(chosen) else 0.,
                            simultaneous_bound_violation=int(violations)))
                print(dataset,noise,world,'auc',round(auc,3),flush=True)
    dest=Path(out_dir);dest.mkdir(exist_ok=True,parents=True)
    with open(dest/'audit_draws.csv','w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(out[0]));wr.writeheader();wr.writerows(out)
    per_world=[]
    for dataset in dataset_names:
        for noise in (.2,.35):
            for world in range(worlds):
                for design in STRAT_WEIGHTS:
                    arr=[r for r in out if r['dataset']==dataset and r['noise']==noise and r['world']==world and r['design']==design]
                    per_world.append(dict(dataset=dataset,noise=noise,world=world,design=design,
                        score_auc=arr[0]['auc'],nonempty_rate=float(np.mean([r['nonempty'] for r in arr])),
                        mean_returned=float(np.mean([r['n_returned'] for r in arr])),
                        mean_verified_errors=float(np.mean([r['verified_errors'] for r in arr])),
                        mean_total_errors_found=float(np.mean([r['total_errors_found'] for r in arr])),
                        bound_violations=sum(r['simultaneous_bound_violation'] for r in arr)))
    with open(dest/'world_level.csv','w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(per_world[0]));wr.writeheader();wr.writerows(per_world)
    summary=[]
    for dataset in dataset_names:
        for noise in (.20,.35):
            for design in STRAT_WEIGHTS:
                arr=[r for r in per_world if r['dataset']==dataset and r['noise']==noise and r['design']==design]
                summary.append(dict(dataset=dataset,noise=noise,design=design,n_worlds=worlds,
                    mean_across_world_nonempty_rate=float(np.mean([r['nonempty_rate'] for r in arr])),
                    min_world_nonempty_rate=float(np.min([r['nonempty_rate'] for r in arr])),
                    max_world_nonempty_rate=float(np.max([r['nonempty_rate'] for r in arr])),
                    mean_across_world_returned=float(np.mean([r['mean_returned'] for r in arr])),
                    mean_across_world_verified_errors=float(np.mean([r['mean_verified_errors'] for r in arr])),
                    mean_across_world_total_errors_found=float(np.mean([r['mean_total_errors_found'] for r in arr])),
                    total_bound_violations=int(sum(r['bound_violations'] for r in arr))))
    with open(dest/'summary.csv','w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(summary[0]));wr.writeheader();wr.writerows(summary)
    (dest/'provenance.json').write_text(json.dumps(dict(seed=SEED,worlds=worlds,repetitions_per_world=reps,
         note='Worlds are independent injected corruption assignments on shared original feature matrices; audit draws within world condition on one library; outcomes are not independent natural-noise datasets.',
         noise_levels=[.20,.35],fdr_target=.10,noncoverage_delta=.05,budget_fraction=.30,
         rank_cut_fractions=list(CUT_FRACTIONS),designs=STRAT_WEIGHTS),indent=2))
    return summary
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--worlds',type=int,default=4)
    ap.add_argument('--reps',type=int,default=12);ap.add_argument('--out_dir',default='multiworld')
    ap.add_argument('--datasets',default='iris,wine,breast_cancer,digits')
    a=ap.parse_args();print(json.dumps(experiment(a.worlds,a.reps,a.out_dir,a.datasets.split(',')),indent=2))
