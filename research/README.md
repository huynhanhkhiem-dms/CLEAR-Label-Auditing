# CLEAR research extension — finite-list certification (INTERNAL REVIEW)

This is an isolated, **unmerged** research branch based on the frozen CLEAR submission snapshot `6beb703d2e29baa9dc6955c10801033e001ca554`. The original `clear_core.py`, PMLB tables, CIFAR-10N tables and original manuscript data are unchanged.

## Scope and statistical target

The original CLEAR method controls *expected* false discovery proportion using existing finite-library conformal inference. The extension instead constructs **high-probability bounds for the FDP of a realized list of unverified records**. These are different estimands and the proofs cannot be interchanged.

The extension freezes the score order, rank strata, random verification allocation and a finite family of candidate top-k lists before adjudication. Within each stratum, it samples uniformly without replacement; for each predeclared top-k prefix it inverts the exact hypergeometric lower tail to upper-bound the number of **correct** labels among candidates. Bonferroni's union bound controls all precommitted prefixes jointly, so a list may be selected from the family after inspection of the verification results. Verified labels are removed from the returned list. The inference module accepts only observed verification labels, not withheld truth.

These mathematical ingredients are established in the literature. Closest work includes Wendell and Schmee (1996), Anthony and Salehzadeh Nobari (2026), and Song, Jin and Candès (2026). **No claim of a new concentration theorem or Q1-grade novelty is made.**

## Run

From repo root:

```bash
python -m pip install numpy scipy
python -m unittest discover -s research -p 'test_certification.py' -v
python research/run_simulations.py
```

The pilot writes `research/pilot_repetitions.csv` and `research/pilot_summary.json`. Both files are synthetic experiment artifacts; they are **not** the historical PMLB or CIFAR-10N experiment.

N=4,000; 400 verified labels; 90 audit draws per scenario and design; q=.10; delta=.05; five frozen top-k cutoffs (250, 500, 750, 1,000, 1,250). Score-stratified allocation across eight equal-size strata is (220, 80, 20, 16, 16, 16, 16, 16). Four fixed truth populations are generated once, then reused across audit draws. Uniform SRS receives the same verification budget.

## Interpretation

The pilot is conditional on four synthetic finite repositories. Strong-score settings may favor uniform sampling on **number of unverified records returned**, because verifying more high-ranked labels reduces the remaining candidate pool. In the moderate-score setting, score stratification can improve the nonempty certificate rate and total discovered errors. Weak-score and uninformative scenarios abstain. Neither a simulation frequency nor the exact-inversion theorem establishes new methodology beyond prior literature.

## Outstanding before any submission

1. Independent novelty audit of PAC Labels, simultaneous FDP bounds, finite-population quality audits and 2026 label-auditing research.
2. Re-run this extension on multiple **real** label-noise benchmark datasets with fixed score specifications, verified data provenance and honest performance comparisons.
3. Compare more than these two naive allocations, including optimal budget policy and out-of-distribution cases. Avoid tuning allocations on evaluation labels.
4. Formalize a *genuinely new* allocation/decision scholarly object, if possible; otherwise treat this as an empirical extension or stop the Q1 methodological pathway.
5. Human author verification of all quantitative results, manuscript and journal-specific disclosure policy.

The branch is **not** submission-ready and should not be merged into the archival submission snapshot before the next independent QA.
