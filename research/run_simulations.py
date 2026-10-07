"""Reproducible pilot (not a benchmark-data experiment).
Run: python research/run_simulations.py
From repository root with numpy and scipy installed.
"""
import csv
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from clear_certification import (fixed_rank_strata, stratified_draw,
    simultaneous_certificates, largest_certified)

ROOT = Path(__file__).resolve().parent
N=4000
BUDGET=400
CUTS=[250,500,750,1000,1250]
DELTA=.05
Q=.10
REPEATS=90
SEED=20261008
STRAT_ALLOCATION=[220,80,20,16,16,16,16,16]
MODELS={"strong":(.995,.92,.08),"moderate":(.97,.72,.12),
        "weak":(.85,.50,.25),"uninformative":(.20,.20,.20)}
score=np.linspace(1.0,0.0,N)
_,strata=fixed_rank_strata(score,8)
rng=np.random.default_rng(SEED)
rows=[]
for name,pr in MODELS.items():
    rates=np.select([np.arange(N)<500,np.arange(N)<1000],
                    [pr[0],pr[1]],default=pr[2])
    truth=(rng.random(N)<rates).astype(int)
    for rep in range(REPEATS):
        for design in ("uniform","score_stratified"):
            if design=="uniform":
                st=np.zeros(N,dtype=int)
                ver=stratified_draw(st,[BUDGET],rng)
            else:
                st=strata
                ver=stratified_draw(st,STRAT_ALLOCATION,rng)
            cs=simultaneous_certificates(score,ver,truth[ver],st,CUTS,DELTA)
            c=largest_certified(cs,Q)
            selected=c.selected_unverified if c else np.array([],dtype=int)
            false=int(np.sum(truth[selected]==0))
            found=int(np.sum(truth[ver]==1)+np.sum(truth[selected]==1))
            failure=any(np.sum(truth[z.selected_unverified]==0)>z.upper_false
                        for z in cs)
            rows.append(dict(scenario=name,rep=rep,design=design,
                n_verified=BUDGET,selected_size=len(selected),
                total_found=found,certified=int(len(selected)>0),
                actual_fdp=false/len(selected) if len(selected)>0 else 0.,
                simultaneous_bound_failure=int(failure),
                selected_cut=c.cut if c else 0))
with (ROOT/"pilot_repetitions.csv").open("w",newline="") as f:
    writer=csv.DictWriter(f,fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
summary=[]
for name in MODELS:
    for design in ("uniform","score_stratified"):
        a=[r for r in rows if r["scenario"]==name and r["design"]==design]
        summary.append(dict(scenario=name,design=design,
            certified_rate=sum(r["certified"] for r in a)/len(a),
            mean_selected=sum(r["selected_size"] for r in a)/len(a),
            mean_total_found=sum(r["total_found"] for r in a)/len(a),
            bound_coverage_failures=sum(r["simultaneous_bound_failure"] for r in a)))
(ROOT/"pilot_summary.json").write_text(json.dumps(dict(seed=SEED,N=N,
    budget=BUDGET,cuts=CUTS,delta=DELTA,q=Q,repeats=REPEATS,
    allocation=STRAT_ALLOCATION,truth_model=MODELS,summary=summary),indent=2))
print(json.dumps(summary,indent=2))
