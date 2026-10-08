import unittest
import numpy as np
from clear_certification import fixed_rank_strata, stratified_draw, simultaneous_certificates
from prefix_conditional import prefix_conditional_certificates

class TestPrefixConditional(unittest.TestCase):
    def test_no_leakage(self):
        rng=np.random.default_rng(111)
        scores=rng.random(120)
        _,strata=fixed_rank_strata(scores,6)
        verified=stratified_draw(strata,[7,7,7,7,7,7],rng)
        errors=rng.integers(0,2,size=120)
        a=prefix_conditional_certificates(scores,verified,errors[verified],strata,[10,20,40,80])
        errors[np.setdiff1d(np.arange(120),verified)]=1-errors[np.setdiff1d(np.arange(120),verified)]
        b=prefix_conditional_certificates(scores,verified,errors[verified],strata,[10,20,40,80])
        self.assertEqual([(c.upper_false,c.upper_fdp) for c in a],
                         [(c.upper_false,c.upper_fdp) for c in b])

    def test_census_no_false_unverified(self):
        scores=np.arange(32,dtype=float)
        _,strata=fixed_rank_strata(scores,4)
        ver=np.arange(32)
        errors=np.ones(32,dtype=int)
        out=prefix_conditional_certificates(scores,ver,errors[ver],strata,[4,8,16,32])
        self.assertTrue(all(c.upper_false==0 and c.returned_size==0 for c in out))

    def test_empirical_simultaneous_coverage(self):
        rng=np.random.default_rng(222)
        N=160
        score=rng.random(N)
        _,strata=fixed_rank_strata(score,4)
        errors=np.asarray([rng.random()<p for p in np.linspace(.9,.1,N)],dtype=int)
        violations=0
        for _ in range(400):
            ver=stratified_draw(strata,[12,12,12,12],rng)
            certs=prefix_conditional_certificates(score,ver,errors[ver],strata,[20,40,80,120],delta=.05)
            violations+=int(any(int(np.sum(errors[c.selected_unverified]==0))>c.upper_false for c in certs))
        self.assertLessEqual(violations,40) # smoke test only; not the coverage proof

    def test_agrees_at_stratum_aligned_prefix(self):
        rng=np.random.default_rng(223)
        N=160
        score=np.arange(N,dtype=float)
        _,st=fixed_rank_strata(score,4)
        ver=stratified_draw(st,[20,20,20,20],rng)
        error=rng.integers(0,2,size=N)
        a=simultaneous_certificates(score,ver,error[ver],st,[40,80,120,160])
        b=prefix_conditional_certificates(score,ver,error[ver],st,[40,80,120,160])
        self.assertEqual([z.upper_false for z in a],[z.upper_false for z in b])

if __name__=='__main__':unittest.main()
