Online Resource 1 — Reproducibility materials
Title: Label Quality Audits in Fixed Repositories: Random Verification, Error-Rate Control, and Certification Trade-offs
Target journal: International Journal of Data Science and Analytics
Author: Huynh Anh Khiem
Affiliation: Faculty of Information Technology, Ton Duc Thang University, Ho Chi Minh City, Vietnam
Email: huynhanhkhiem@tdtu.edu.vn
ORCID: 0009-0007-7210-174X

CONTENTS
The code directory contains finite-population certification routines, two complementary valid bound formulations, offline experiments and nine unit tests. The results directory contains the archived CSV summaries, per-draw results, and configurations for the NEW synthetic-label corruption extensions. Historical PMLB/CIFAR-10N benchmark outputs are NOT recreated in this package; the frozen original repository linked in the manuscript separately contains the historical materials.

ENVIRONMENT
Python 3.11 or newer; see requirements.txt for packages pinned in the verified offline environment. scikit-learn packaged datasets are used; no private data or survey is required. All injected errors are synthetic. Do not interpret these as naturally occurring human noise.

RUN UNIT TESTS (from this extracted directory)
python -m pip install -r requirements.txt
python -m unittest discover -s code -p "test_*.py" -v

RERUN FIXED-LIBRARY ANALYSIS (may take significant time)
python code/real_features_noise_audit.py --repeats 30 --outdir reproduced_main
python code/real_features_noise_audit_compare.py --repeats 30 --outdir reproduced_comparison
python code/multiworld_sensitivity.py --datasets digits --worlds 3 --reps 12 --out_dir reproduced_multiworld

The experiment generation seeds, data inclusion, class replacements, nominal q/delta and candidate list definitions are recorded in the code and provenance files. Audit randomizations within one generated corrupted library are not independent datasets. Violations of finite-sample upper bounds are reported without filtering: three among 216 audit draws in the Digits multiworld sensitivity.

NOTE ON CLAIMS
Finite-population bounds are constructed from established hypergeometric inversion and a union bound. The package is not evidence of a new FDR theorem. The manuscript distinguishes the original expected-FDR selection and the high-probability realized-list certificate.
