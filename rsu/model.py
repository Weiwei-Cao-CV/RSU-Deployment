"""Lifetime-constrained RSU deployment model (re-implementation).

Network: undirected graph of discrete nodes (1 km spacing, R = 500 m).
For each link and evaluation period the traffic state is the two-way density k (veh/km) and mean speed v (m/s).
Propagation follows the information-traffic coupled model: each propagation cycle consists of a chain of
immediate V2V hops between CAVs followed by one ferry (store-carry-forward) phase.
"""
import numpy as np
from scipy.sparse.csgraph import shortest_path, minimum_spanning_tree

P = dict(R=500.0, delta=0.05, s0=2.0, a=4.5, TH=1.2, TC=0.6, dv_ratio=0.3, pmode=(0.25, 0.5, 0.25),
         alpha=0.1, beta=6.0, hop='renewal')


def link_time(k, v, theta, length, p=P):
    """mean propagation time (s) over one link of `length` m; k veh/km (two-way), v m/s"""
    R, delta = p['R'], p['delta']
    v = max(v, 0.5)
    kc = max(theta * k, 1e-6) / 1000.0                     # CAVs per m (both directions relay)
    dH = p['s0'] + v * p['TH'] + v * v / (2 * p['a'])
    dC = p['s0'] + v * p['TC'] + v * v / (2 * p['a'])
    d0 = (1 - theta) * dH + theta * dC                     # equivalent minimum spacing
    d0 = min(d0, 0.9 / kc)
    lam = kc / (1 - kc * d0)                              # shifted-exponential parameter, E[d] = 1/kc
    if R <= d0:
        q, dbar = 0.0, 0.0
    else:
        q = 1 - np.exp(-lam * (R - d0))
        # E[d | d0 <= d < R]
        x = R - d0
        dbar = d0 + (1 / lam - x * np.exp(-lam * x) / (1 - np.exp(-lam * x)))
    m = max(1, int(round(kc * length)))                    # maximum hops on the link (m + 1 CAVs)
    h = np.arange(1, m + 1)
    if p.get('hop', 'renewal') == 'renewal':
        # validated default: hop chain not truncated at the link end; hops per cycle ~ geometric (mean q/(1-q)),
        # a ferry phase only after a break (see validate_propagation.py)
        hbar = min(q, 1 - 1e-9) / (1 - min(q, 1 - 1e-9))
    elif p.get('hop') == 'chain_end':
        # earlier implementation: chain always starts at the first of the m + 1 CAVs
        ph = q ** np.arange(m + 1) * (1 - q); ph[-1] = q ** m
        hbar = float((np.arange(m + 1) * ph).sum())
    else:  # 'paper'
        # original Eq. (6): starting CAV uniform over the m + 1 positions, hops truncated by the remaining CAVs
        hbar = float((h * ((m - h) * q ** h * (1 - q) + q ** h) / (m + 1)).sum())
    EX, tX = hbar * dbar, hbar * delta
    # ferry: residual gap is exponential with mean 1/lam; three closing modes
    G = 1 / lam
    dv = p['dv_ratio'] * v
    closing = np.array([dv, 2 * v, dv])                    # same-dir same-lane, via opposite lane, reverse same-lane
    tY = float(np.dot(p['pmode'], G / closing))         # weighted by mode probabilities
    EY = v * tY
    u = (EX + EY) / (tX + tY)
    return length / u


def reach_time(G_links, states, theta, n):
    """all-pairs shortest propagation times. G_links: list of (i, j, length_m); states: (k, v) per link"""
    T = np.full((n, n), np.inf)
    for (i, j, L), (k, v) in zip(G_links, states):
        t = link_time(k, v, theta, L)
        T[i, j] = T[j, i] = min(T[i, j], t)
    np.fill_diagonal(T, 0.0)
    return shortest_path(T, method='D', directed=False)


def node_weights(G_links, states, theta, n, R=None):
    """CAVs within R of each node (half of each adjacent 1 km link)"""
    R = P['R'] if R is None else R
    w = np.zeros(n)
    for (i, j, L), (k, v) in zip(G_links, states):
        share = min(R, L / 2) / 1000.0
        w[i] += theta * k * share
        w[j] += theta * k * share
    return w


class Instance:
    """one traffic snapshot (period) of a network"""
    def __init__(self, G_links, states, theta, tau, n, Droad, cand=None):
        self.n, self.theta, self.tau = n, theta, tau
        self.T = reach_time(G_links, states, theta, n)
        self.A = (self.T <= tau)                            # A[i, j]: RSU at i reaches j
        self.w = node_weights(G_links, states, theta, n)
        self.D = Droad                                      # road distance between nodes, km
        self.cand = np.arange(n) if cand is None else np.asarray(cand)
        dens = np.zeros(n); cnt = np.zeros(n)
        for (i, j, L), (k, v) in zip(G_links, states):
            dens[i] += k; dens[j] += k; cnt[i] += 1; cnt[j] += 1
        self.density = dens / np.maximum(cnt, 1)

    def cable(self, sel):
        sel = list(sel)
        if len(sel) < 2:
            return 0.0
        sub = self.D[np.ix_(sel, sel)]
        if not np.isfinite(sub).all():
            return np.inf
        return float(minimum_spanning_tree(sub).sum())

    def coverage(self, sel):
        if len(sel) == 0:
            return np.zeros(self.n, bool)
        return self.A[list(sel)].any(0)

    def evaluate(self, sel, p=P):
        y = self.coverage(sel)
        B = float(self.w[y].sum())
        L = self.cable(sel)
        K = len(sel)
        return dict(U=B - p['alpha'] * L - p['beta'] * K, B=B, L=L, K=K,
                    node_cov=float(y.mean()), cav_cov=B / max(self.w.sum(), 1e-9))
