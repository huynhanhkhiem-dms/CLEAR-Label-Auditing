"""CIFAR-10N human-noise verification design replication (research application).
Requires official CIFAR-10_human.pt and frozen five score matrices from ESM_1.zip.
This is a repeat verification-sampling experiment, not new annotation collection.
"""
from pathlib import Path
import hashlib, json
from functools import lru_cache
import codecs
import numpy as np, pandas as pd, torch
from numpy.core.multiarray import _reconstruct
import clear_certification as cc
import prefix_conditional as pc

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "inputs"
OUT = ROOT / "results" / "human_noise_extension"
OUT.mkdir(parents=True, exist_ok=True)
HUMAN_SHA = "873e69c39cb9b5e97fb6bae2d60bb59b38a5cbc31d2b868f7903dcb2b9dd2310"
LABELS = ["aggre_label","worse_label","random_label1","random_label2","random_label3"]
CUTS = [250,500,1000,2500,5000,10000]
BUDGETS = [200,1000]
ALLOCATION = {200:[125,35,20,20],1000:[625,175,100,100]}
SEED = 210426
REPEATS = 100
DELTA = 0.05
Q = [0.10,0.20]
original_hg = cc.hg_upper_total
cached_hg = lru_cache(maxsize=250000)(original_hg)
cc.hg_upper_total = cached_hg
pc.hg_upper_total = cached_hg

source = INPUT / "CIFAR-10_human.pt"
assert hashlib.sha256(source.read_bytes()).hexdigest() == HUMAN_SHA
with torch.serialization.safe_globals([
    (_reconstruct, "numpy.core.multiarray._reconstruct"),
    np.ndarray, (np.dtype, "numpy.dtype"), codecs.encode,
    np.dtypes.Int64DType, np.dtypes.Int32DType, np.dtypes.Float64DType
]):
    labels = torch.load(source, map_location="cpu", weights_only=True)
truth = np.asarray(labels["clean_label"], dtype=int)
rng = np.random.default_rng(SEED)
rows = []
provenance = []
for label_set in LABELS:
    noisy = np.asarray(labels[label_set], dtype=int)
    error = (noisy != truth).astype(np.int8)
    score_path = INPUT / (label_set + "_oof_probs.npy")
    probability = np.load(score_path)
    assert probability.shape == (50000,10)
    assert np.all(np.isfinite(probability))
    assert np.allclose(probability.sum(axis=1), 1.0, atol=1e-5)
    score = 1.0 - probability[np.arange(50000),noisy]
    _, frozen_strata = cc.fixed_rank_strata(score, 4)
    ranking = np.argsort(-score, kind="stable")
    provenance.append(dict(label_set=label_set,noise_rate=float(error.mean()),
        top_1000_error_rate=float(error[ranking[:1000]].mean()),
        top_5000_error_rate=float(error[ranking[:5000]].mean()),
        score_sha256=hashlib.sha256(score_path.read_bytes()).hexdigest()))
    for budget in BUDGETS:
        for rep in range(REPEATS):
            for design in ["uniform","score_stratified"]:
                strata = np.zeros(50000,dtype=int) if design=="uniform" else frozen_strata
                allocation = [budget] if design=="uniform" else ALLOCATION[budget]
                verified = cc.stratified_draw(strata, allocation, rng)
                checked_error = error[verified]   # do not pass other labels to inferential code
                for method, infer in [
                    ("full_stratum",cc.simultaneous_certificates),
                    ("prefix_conditional",pc.prefix_conditional_certificates)
                ]:
                    family = infer(score,verified,checked_error,strata,CUTS,DELTA)
                    cover_fail = int(any(
                        np.sum(error[c.selected_unverified] == 0) > c.upper_false
                        for c in family))
                    for q in Q:
                        cert = cc.largest_certified(family,q)
                        returned = np.array([],dtype=int) if cert is None else cert.selected_unverified
                        rows.append(dict(label_set=label_set,noise_rate=float(error.mean()),
                            budget=budget,rep=rep,design=design,certificate_method=method,
                            q=q,delta=DELTA,returned_size=len(returned),
                            selected_cut=cert.cut if cert else 0,
                            certificate_issued=int(len(returned)>0),
                            upper_fdp=cert.upper_fdp if cert else 0.0,
                            realized_fdp=float(np.mean(error[returned]==0)) if len(returned) else 0.0,
                            verified_errors_found=int(error[verified].sum()),
                            selected_errors_discovered=int(error[returned].sum()),
                            simultaneous_bound_failure=cover_fail))
    print("finished",label_set,flush=True)
frame = pd.DataFrame(rows)
frame.to_csv(OUT/"cifar_real_design_audit_replicates.csv",index=False)
summ = frame.groupby(["label_set","budget","design","certificate_method","q"],
        as_index=False).agg(runs=("rep","size"),
        certificate_rate=("certificate_issued","mean"),
        mean_returned=("returned_size","mean"),
        mean_verified_errors=("verified_errors_found","mean"),
        mean_selected_errors=("selected_errors_discovered","mean"),
        mean_realized_fdp=("realized_fdp","mean"),
        mean_claimed_upper_fdp=("upper_fdp","mean"),
        joint_coverage_failures=("simultaneous_bound_failure","sum"))
summ.to_csv(OUT/"cifar_real_design_audit_summary.csv",index=False)
(OUT/"cifar_real_design_audit_provenance.json").write_text(json.dumps({
    "description":"Exploratory, real human-noise but additional policy-and-confidence experiment; not historical CLEAR FDR run",
    "run_date":"2026-10-08","random_seed":SEED,"repeats":REPEATS,
    "budgets":BUDGETS,"q":Q,"delta":DELTA,"cuts":CUTS,
    "rank_strata":4,"allocations":ALLOCATION,
    "verified_budget_per_method_equal":True,
    "human_label_sha256":HUMAN_SHA,"label_sets":provenance,
    "scores_are_postprocessed_oof_predictions":True,
    "multiple_q_values_derived_from_same_frozen_candidates":True,
    "full_dataset_labels_are_only_used_for_evaluation":True,
    "warning":"Exploratory; no independently optimized allocation. Statistical coverage permits a nonzero failure rate."
},indent=2))
print("FINISHED",len(frame),"rows",len(summ),"groups")
