# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Euclidean manifold."""

from manifolds.base import Manifold


class Euclidean(Manifold):
    """
    Euclidean Manifold class.
    """

    def __init__(self,c=None):
        super(Euclidean, self).__init__()
        self.c = None # c is the curvature which is None for Euclidean
        self.name = 'Euclidean'

    def normalize(self, p):
        dim = p.size(-1)
        p.view(-1, dim).renorm_(2, 0, 1.)
        return p

    def sqdist(self, p1, p2):
        return (p1 - p2).pow(2).sum(dim=-1)

    def egrad2rgrad(self, p, dp):
        return dp

    def proj(self, p):
        return p

    def proj_tan(self, u, p):
        return u

    def proj_tan0(self, u):
        return u

    def expmap(self, u, p):
        return p + u

    def logmap(self, p1, p2):
        return p2 - p1

    def expmap0(self, u):
        return u

    def logmap0(self, p):
        return p

    def mobius_add(self, x, y, c, dim=-1):
        return x + y

    def mobius_matvec(self, m, x):
        mx = x @ m.transpose(-1, -2)
        return mx

    def init_weights(self, w, c, irange=1e-5):
        w.data.uniform_(-irange, irange)
        return w

    def inner(self, p, c, u, v=None, keepdim=False):
        if v is None:
            v = u
        return (u * v).sum(dim=-1, keepdim=keepdim)

    def ptransp(self, x, y, v):
        return v

    def ptransp0(self, x, v):
        return x + v
