"""Independent tests for research/clear_certification.py."""
import unittest
import numpy as np
from scipy.stats import hypergeom
from clear_certification import (hg_upper_total, fixed_rank_strata,
    stratified_draw, simultaneous_certificates)

class TestFiniteListAudit(unittest.TestCase):
    def test_exact_inversion(self):
        for N in range(2, 26):
            for n in range(N+1):
                for x in range(n+1):
                    for alpha in (.01, .05, .20):
                        observed = hg_upper_total(N,n,x,alpha)
                        expected = max(K for K in range(x,N-(n-x)+1)
                                       if hypergeom.cdf(x,N,K,n)>=alpha)
                        self.assertEqual(observed, expected)
    def test_census(self):
        self.assertEqual(hg_upper_total(200,200,19,.05),19)
    def test_empty_remainder(self):
        scores=np.arange(40,dtype=float)[::-1]
        _,strata=fixed_rank_strata(scores,4)
        ver=np.arange(40)
        c=simultaneous_certificates(scores,ver,np.zeros(40,dtype=int),
                                    strata,[5,10,20,40])
        self.assertTrue(all(z.upper_false==0 for z in c))
    def test_coverage_on_fixed_library(self):
        rng=np.random.default_rng(448)
        n=160
        scores=np.arange(n,dtype=float)[::-1]
        _,strata=fixed_rank_strata(scores,4)
        truth=(rng.random(n)<np.linspace(.99,.1,n)).astype(int)
        for _ in range(70):
            ver=stratified_draw(strata,[20,10,10,10],rng)
            cs=simultaneous_certificates(scores,ver,truth[ver],
                    strata,[20,40,60,80,120],delta=.05)
            for c in cs:
                self.assertLessEqual(int(np.sum(truth[c.selected_unverified]==0)),
                                     c.upper_false)
    def test_does_not_see_unverified_truth(self):
        rng=np.random.default_rng(55)
        score=rng.random(100)
        _,st=fixed_rank_strata(score,5)
        ver=stratified_draw(st,[8]*5,rng)
        y=rng.integers(0,2,size=100)
        a=simultaneous_certificates(score,ver,y[ver],st,[10,30,50])
        other=y.copy()
        unseen=np.setdiff1d(np.arange(100),ver)
        other[unseen]=1-y[unseen]
        b=simultaneous_certificates(score,ver,other[ver],st,[10,30,50])
        self.assertEqual([(z.cut,z.upper_false,z.upper_fdp) for z in a],
                         [(z.cut,z.upper_false,z.upper_fdp) for z in b])

if __name__=="__main__":
    unittest.main()
