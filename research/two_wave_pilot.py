"""Negative-control pilot for outcome-adaptive two-wave stratification.
Conditional on pilot adjudication outcomes, only second-wave SRS samples are
used for the finite-population FDP certificate.
Run: python research/two_wave_pilot.py
"""
from pathlib import Path
import numpy as np, json, csv
from clear_certification import fixed_rank_strata, stratified_draw, simultaneous_certificates, largest_certified

N=4000; B0=80; B1=320; CUTS=[250,500,750,1000,1250]; DELTA=.05; Q=.1
N_REP=90; SEED=20261009
MODELS={'strong':(.995,.92,.08),'moderate':(.97,.72,.12),
        'weak':(.85,.50,.25),'uninformative':(.20,.20,.20)}
score=np.linspace(1.0,0.0,N)
_,first_strata=fixed_rank_strata(score,8)
PILOT_ALLOCATION=[40,15,5,4,4,4,4,4]
SECOND_L=[125,65,40,18,18,18,18,18]
SECOND_H=[205,55,20,8,8,8,8,8]
assert sum(PILOT_ALLOCATION)==B0 and sum(SECOND_L)==B1 and sum(SECOND_H)==B1
rows=[]
rng=np.random.default_rng(SEED)
for name,pr in MODELS.items():
    rates=np.select([np.arange(N)<500,np.arange(N)<1000],[pr[0],pr[1]],default=pr[2])
    truth=(rng.random(N)<rates).astype(int)
    for rep in range(N_REP):
        pilot=stratified_draw(first_strata,PILOT_ALLOCATION,rng)
        top_pilot=pilot[first_strata[pilot]==0]
        no_observed_correct=bool(np.all(truth[top_pilot]==1))
        left=np.setdiff1d(np.arange(N),pilot)
        scores_left=score[left]
        _,st=fixed_rank_strata(scores_left,8)
        for design in ('pilot_uniform','pilot_fixed','pilot_adaptive'):
            if design=='pilot_uniform':
                stratum=np.zeros(len(left),dtype=int)
                allocation=[B1]
            else:
                stratum=st
                allocation=SECOND_L if design=='pilot_fixed' or no_observed_correct else SECOND_H
            second_local=stratified_draw(stratum,allocation,rng)
            cs=simultaneous_certificates(scores_left,second_local,truth[left[second_local]],stratum,CUTS,DELTA)
            cert=largest_certified(cs,Q)
            picked=left[cert.selected_unverified] if cert else np.array([],dtype=int)
            verified=np.concatenate([pilot,left[second_local]])
            false_selected=int(np.sum(truth[picked]==0))
            bound_fail=any(int(np.sum(truth[left[z.selected_unverified]]==0))>z.upper_false for z in cs)
            rows.append(dict(scenario=name,rep=rep,design=design,
                adaptive_low_allocation=int(no_observed_correct),
                certified=int(len(picked)>0),selected_size=len(picked),
                total_found=int(np.sum(truth[verified])+np.sum(truth[picked])),
                actual_fdp=float(false_selected/len(picked)) if len(picked) else 0.,
                bound_failure=int(bound_fail)))
summary=[]
for name in MODELS:
    for design in ('pilot_uniform','pilot_fixed','pilot_adaptive'):
        rec=[r for r in rows if r['scenario']==name and r['design']==design]
        summary.append(dict(scenario=name,design=design,n_reps=len(rec),
            nonempty_cert_rate=round(float(np.mean([r['certified'] for r in rec])),4),
            mean_selected=round(float(np.mean([r['selected_size'] for r in rec])),2),
            mean_total_found=round(float(np.mean([r['total_found'] for r in rec])),2),
            failure_count=sum(r['bound_failure'] for r in rec)))
base=Path(__file__).parent
with (base/'two_wave_repetitions.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
(base/'two_wave_summary.json').write_text(json.dumps(dict(seed=SEED,
    pilot_allocation=PILOT_ALLOCATION,low=SECOND_L,high=SECOND_H,summary=summary),indent=2))
print(json.dumps(summary,indent=2))
