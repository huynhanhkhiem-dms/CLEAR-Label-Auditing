"""Prefix-conditional finite-population certificate under stratified SRS.

For each precommitted list prefix and fixed stratum, condition on the number
of SRS-verified records falling in that prefix. Those verified prefix records
are a simple random sample of the prefix-by-stratum subpopulation.
This is a (known) exact-hypergeometric inference construction, not a claimed
new statistical theorem. Adaptively picking among *predeclared* lists is
covered by finite-family Bonferroni allocation.
"""
import numpy as np
from clear_certification import Certificate, hg_upper_total

def prefix_conditional_certificates(scores, verified, observed_error, strata, cuts,
                                     delta=0.05):
    scores=np.asarray(scores,dtype=float)
    strata=np.asarray(strata,dtype=int)
    verified=np.asarray(verified,dtype=int)
    observed_error=np.asarray(observed_error,dtype=int)
    N=len(scores)
    if scores.ndim!=1 or len(strata)!=N or not np.isfinite(scores).all():
        raise ValueError('bad scores/strata')
    if len(verified)!=len(observed_error) or len(np.unique(verified))!=len(verified):
        raise ValueError('invalid verification indices')
    if np.any((verified<0)|(verified>=N)) or np.any(~np.isin(observed_error,[0,1])):
        raise ValueError('invalid indices or adjudications')
    if not 0<delta<1 or len(cuts)==0 or len(set(cuts))!=len(cuts):
        raise ValueError('invalid confidence or cutoffs')
    if any(int(c)!=c or not 1<=c<=N for c in cuts):
        raise ValueError('invalid cutoffs')
    order=np.argsort(-scores,kind='stable')
    rank=np.empty(N,dtype=int)
    rank[order]=np.arange(N)
    unique=np.unique(strata)
    alpha=delta/(len(unique)*len(cuts))
    verified_mask=np.zeros(N,dtype=bool)
    verified_mask[verified]=True
    correct_verified=np.zeros(N,dtype=bool)
    correct_verified[verified]=observed_error==0
    certificates=[]
    for cut in cuts:
        prefix=rank<cut
        unseen=prefix & (~verified_mask)
        false_upper=0
        verified_false=0
        for h in unique:
            g=(strata==h)&prefix
            n_prefix=int(g.sum())
            m=int(np.sum(g&verified_mask))
            x=int(np.sum(g&correct_verified))
            verified_false+=x
            if n_prefix == 0:
                continue
            U=hg_upper_total(n_prefix,m,x,alpha)
            false_upper+=min(n_prefix-m,max(0,U-x))
        size=int(unseen.sum())
        certificates.append(Certificate(int(cut),size,int(false_upper),
            float(false_upper/size) if size else 0.,int(verified_false),np.flatnonzero(unseen)))
    return certificates
