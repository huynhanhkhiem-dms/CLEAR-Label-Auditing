"""Independent prospective certificate stress test on the archived CIFAR-10N human labels.

Not a rerun of PMLB or of archived CIFAR CLEAR FDR comparisons. Does not
train/fine-tune model, rather consumes frozen OOF scores from original Drive run.
Any result files here are new experiment outputs.
"""
from __future__ import annotations
import argparse, json, os, csv, time, hashlib
from pathlib import Path
import numpy as np
import torch
from torch.serialization import safe_globals
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from clear_certification import fixed_rank_strata, stratified_draw, simultaneous_certificates, largest_certified
from prefix_conditional import prefix_conditional_certificates

DATA=Path('/mnt/data')
LABELSETS=['aggre_label','worse_label','random_label1','random_label2','random_label3']
CUTS=[500,1000,2500,5000,10000,20000]
DELTA=0.05
Q=.10

def safe_load_labels(path):
    # Legacy UCSC-REAL checkpoint includes simple NumPy arrays, not torch tensors.
    # Explicit allowlist keeps weights_only mode instead of unsafe arbitrary pickle.
    reconstruct=np._core.multiarray._reconstruct
    with safe_globals([(reconstruct,'numpy.core.multiarray._reconstruct'),
                       (np.ndarray,'numpy.ndarray'), (np.dtype,'numpy.dtype'),
                       (type(np.dtype('int64')),'numpy.dtypes.Int64DType')]):
        return torch.load(path,map_location='cpu',weights_only=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',type=Path,default=Path('/mnt/data'))
    p.add_argument('--labels-file',default='CIFAR-10_human.pt')
    p.add_argument('--repeats',type=int,default=20)
    p.add_argument('--budgets',type=int,nargs='+',default=[200,2000])
    p.add_argument('--seed',type=int,default=20261008)
    p.add_argument('--output',default='/mnt/data/CLEAR_DRIVE_REAL_AUDIT')
    args=p.parse_args()
    start=time.time()
    data=args.data_dir.resolve()
    labels_path=Path(args.labels_file)
    if not labels_path.is_absolute(): labels_path=data/labels_path
    if hashlib.sha256(labels_path.read_bytes()).hexdigest()!='873e69c39cb9b5e97fb6bae2d60bb59b38a5cbc31d2b868f7903dcb2b9dd2310':
        raise ValueError('Official CIFAR-10N annotation file SHA-256 mismatch')
    labels=safe_load_labels(labels_path)
    truth=np.asarray(labels['clean_label'],dtype=np.int64)
    configs=[]
    for name in LABELSETS:
        y=np.asarray(labels[name],dtype=np.int64)
        score_path=data/'frozen_oof'/(name+'_oof_probs.npy')
        if not score_path.exists(): score_path=data/(name+'_oof_probs.npy.bin')
        pr=np.load(score_path,allow_pickle=False)
        assert pr.shape==(50000,10)
        assert np.allclose(pr.sum(axis=1),1,atol=1e-5)
        errs=(y!=truth).astype(np.int8)
        score=1-pr[np.arange(len(y)),y]
        order,strata=fixed_rank_strata(score,4)
        configs.append((name,errs,score,strata))
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for name,errs,score,strata in configs:
        for budget in args.budgets:
            for rep in range(args.repeats):
                # Independent counter-based seed per scenario, avoids RNG coupling in saved runs.
                for design in ['uniform','balanced','rank_focused']:
                    rng=np.random.default_rng(args.seed+100000*LABELSETS.index(name)+100*budget+3*rep+['uniform','balanced','rank_focused'].index(design))
                    if design=='uniform':
                        st=np.zeros(len(errs),dtype=np.int8);alloc=[budget]
                    else:
                        st=strata
                        alloc=[budget//4]*4 if design=='balanced' else [int(.60*budget),int(.20*budget),int(.12*budget),0]
                        alloc[-1]=budget-sum(alloc[:-1])
                    verify=stratified_draw(st,alloc,rng)
                    observed=errs[verify]
                    for method in ['full_stratum','prefix_conditional']:
                        func=simultaneous_certificates if method=='full_stratum' else prefix_conditional_certificates
                        cs=func(score,verify,observed,st,CUTS,delta=DELTA)
                        cert=largest_certified(cs,q=Q)
                        selected=cert.selected_unverified if cert else np.empty(0,dtype=int)
                        bad=int(np.sum(errs[selected]==0))
                        violated=any(np.sum(errs[z.selected_unverified]==0)>z.upper_false for z in cs)
                        rows.append(dict(label_set=name,noise_rate=float(errs.mean()),budget=budget,
                            replicate=rep,design=design,certificate=method,
                            certified=int(len(selected)>0),returned=int(len(selected)),
                            upper_fdp=float(cert.upper_fdp) if cert else 0,
                            actual_fdp=float(bad/len(selected)) if len(selected)>0 else 0,
                            false_selected=bad,verified_errors=int(observed.sum()),
                            discovered_errors=int(np.sum(errs[selected]))+int(observed.sum()),
                            bound_failure=int(violated),selected_cut=int(cert.cut) if cert else 0))
            print('PROGRESS',name,'budget',budget,'elapsed',round(time.time()-start,1),flush=True)
    output=out/'cifar10n_real_noise_certificate_replicates.csv'
    with output.open('w',newline='') as f:
        dw=csv.DictWriter(f,fieldnames=rows[0].keys());dw.writeheader();dw.writerows(rows)
    summaries=[]
    for name in LABELSETS:
        for budget in args.budgets:
            for design in ['uniform','balanced','rank_focused']:
                for method in ['full_stratum','prefix_conditional']:
                    ss=[r for r in rows if (r['label_set'],r['budget'],r['design'],r['certificate'])==(name,budget,design,method)]
                    summaries.append(dict(label_set=name,budget=budget,design=design,certificate=method,n=len(ss),
                        noise_rate=ss[0]['noise_rate'],nonempty_rate=np.mean([r['certified'] for r in ss]),
                        mean_returned=np.mean([r['returned'] for r in ss]),
                        mean_verified_errors=np.mean([r['verified_errors'] for r in ss]),
                        mean_discovered_errors=np.mean([r['discovered_errors'] for r in ss]),
                        mean_actual_fdp=np.mean([r['actual_fdp'] for r in ss]),
                        bound_failure_draws=sum(r['bound_failure'] for r in ss)))
    with (out/'cifar10n_real_noise_certificate_summary.csv').open('w',newline='') as f:
        dw=csv.DictWriter(f,fieldnames=summaries[0].keys());dw.writeheader();dw.writerows(summaries)
    with (out/'cifar10n_real_noise_certificate_provenance.json').open('w') as f:
        json.dump(dict(seed=args.seed,repeats=args.repeats,budgets=args.budgets,
            label_sets=LABELSETS,N=50000,
            source_sha256=hashlib.sha256(labels_path.read_bytes()).hexdigest(),
            random_noise_generated=False,original_OOF_predictions_reused=True,
            frozen_score='1 - oof_probability_assigned_to_recorded_human_label',
            cuts=CUTS,delta=DELTA,q=Q,strata=4,
            sampling_designs={'uniform':'SRS without replacement, one stratum',
            'balanced':'4 frozen rank strata, 25% budget each',
            'rank_focused':'4 frozen rank strata, 60/20/12/8 percent budget'},
            caveats=['Use of fixed released CIFAR clean_label as reference, not fresh expert adjudication',
            'Not a new optimal policy; no validity claim beyond established hypergeometric inversion',
            'Repeated audit draws share each fixed human-noise population',
            'The source 2026-08-19 scores were constructed by models trained using noisy labels before verification',
            'Not all possible datasets/noise mechanisms are represented'],
            minutes=(time.time()-start)/60),f,indent=2)
    print('FINAL',len(rows),'rows',len(summaries),'summary groups', 'seconds',round(time.time()-start,2),flush=True)
    for x in summaries:
        if x['budget']==max(args.budgets) and x['certificate']=='prefix_conditional':
            print(x['label_set'],x['design'],'return',round(x['mean_returned'],2),'cert',round(x['nonempty_rate'],2),'errors',round(x['mean_verified_errors'],1),'fails',x['bound_failure_draws'],flush=True)
if __name__=='__main__': main()
