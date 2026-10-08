import unittest
import numpy as np
from scipy.stats import hypergeom
from clear_certification import (hg_upper_total, fixed_rank_strata,
    stratified_draw, simultaneous_certificates, largest_certified)

class TestCertifiedAudit(unittest.TestCase):
    def test_hypergeom_inversion_bruteforce(self):
        for N in range(2, 26):
            for n in range(0, N+1):
                for x in range(n+1):
                    for alpha in (.01, .05, .2):
                        b=hg_upper_total(N,n,x,alpha)
                        expected=max(K for K in range(x,N-(n-x)+1)
                                     if hypergeom.cdf(x,N,K,n)>=alpha)
                        self.assertEqual(b,expected,(N,n,x,alpha))

    def test_census_exact(self):
        self.assertEqual(hg_upper_total(200,200,19,.05),19)

    def test_empty(self):
        x=np.arange(40,dtype=float)[::-1]
        _,s=fixed_rank_strata(x,4)
        v=np.arange(40)
        true=np.zeros(40,dtype=int)
        r=simultaneous_certificates(x,v,true[v],s,[5,10,20,40])
        self.assertTrue(all(a.upper_false==0 for a in r))

    def test_bound_covers_actual_unverified_false_count(self):
        rng=np.random.default_rng(448)
        n=160
        s=np.arange(n,dtype=float)[::-1]
        _,groups=fixed_rank_strata(s,4)
        truth=(rng.random(n)< np.linspace(.99,.1,n)).astype(int)
        for _ in range(70):
            ver=stratified_draw(groups,[20,10,10,10],rng)
            cs=simultaneous_certificates(s,ver,truth[ver],groups,[20,40,60,80,120],delta=.05)
            for c in cs:
                actual=int(np.sum(truth[c.selected_unverified]==0))
                # This seed/test is not a proof, but validates implementation.
                self.assertLessEqual(actual,c.upper_false)

    def test_no_peeking_in_unverified_outcomes(self):
        rng=np.random.default_rng(55)
        score=rng.random(100)
        _,st=fixed_rank_strata(score,5)
        ver=stratified_draw(st,[8]*5,rng)
        y=rng.integers(0,2,size=100)
        a=simultaneous_certificates(score,ver,y[ver],st,[10,30,50])
        other=y.copy(); other[np.setdiff1d(np.arange(100),ver)]=1-y[np.setdiff1d(np.arange(100),ver)]
        b=simultaneous_certificates(score,ver,other[ver],st,[10,30,50])
        self.assertEqual([(z.cut,z.upper_false,z.upper_fdp) for z in a],
                         [(z.cut,z.upper_false,z.upper_fdp) for z in b])

if __name__=='__main__':unittest.main()
