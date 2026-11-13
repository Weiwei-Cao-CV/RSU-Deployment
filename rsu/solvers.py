"""Baseline solvers for a fixed number of RSUs K."""
import numpy as np, itertools, time

def greedy(inst, K):
    sel = []
    for _ in range(K):
        best, bu = None, -np.inf
        for c in inst.cand:
            if c in sel: continue
            u = inst.evaluate(sel + [c])['U']
            if u > bu: best, bu = c, u
        sel.append(best)
    return sorted(sel)

def swap_ls(inst, sel):
    """1-swap local search (first improvement) from a starting layout"""
    sel = list(sel); u = inst.evaluate(sel)['U']; improved = True
    while improved:
        improved = False
        for i in range(len(sel)):
            for c in inst.cand:
                if c in sel: continue
                t = sel[:i] + [c] + sel[i + 1:]
                ut = inst.evaluate(t)['U']
                if ut > u + 1e-9:
                    sel, u, improved = t, ut, True
                    break
            if improved: break
    return sorted(sel)

def greedy_free_k(inst, Kmax=None):
    """number of RSUs chosen by the model: greedy order, keep the prefix with the largest U"""
    sel, us = [], []
    for _ in range(min(Kmax or len(inst.cand), len(inst.cand))):
        c = max((c for c in inst.cand if c not in sel), key=lambda c: inst.evaluate(sel + [c])['U'])
        sel.append(c); us.append(inst.evaluate(sel)['U'])
    return sorted(sel[:int(np.argmax(us)) + 1])

def drsu(inst, K, min_sep=1.0):
    """density-based baseline (Barrachina et al.): highest-density candidates first"""
    order = sorted(inst.cand, key=lambda c: -inst.density[c])
    sel = []
    for c in order:
        if all(inst.D[c, s] >= min_sep for s in sel):
            sel.append(c)
        if len(sel) == K: break
    return sorted(sel)

def ga(inst, K, pop=150, gens=150, pc=0.9, pm=0.1, seed=0, history=False):
    """history=True -> (layout, best U per generation); history='mean' -> also the population mean per generation.
    Generation 0 is the random initial population."""
    rng = np.random.default_rng(seed)
    cand = np.array(inst.cand)
    def fit(ind): return inst.evaluate(list(ind))['U']
    P = [sorted(rng.choice(cand, K, replace=False)) for _ in range(pop)]
    F = np.array([fit(i) for i in P]); hist = [F.max()]; hmean = [F.mean()]
    for g in range(gens):
        new = [P[int(np.argmax(F))]]                      # elitism
        while len(new) < pop:
            a = P[max(rng.integers(pop, size=3), key=lambda i: F[i])]
            b = P[max(rng.integers(pop, size=3), key=lambda i: F[i])]
            child = list(a)
            if rng.random() < pc:                          # uniform crossover on the union, repaired to K
                pool = list(set(a) | set(b))
                child = list(rng.choice(pool, K, replace=False))
            if rng.random() < pm:                          # swap mutation
                out = rng.integers(K); free = np.setdiff1d(cand, child)
                child[out] = rng.choice(free)
            new.append(sorted(child))
        P = new; F = np.array([fit(i) for i in P]); hist.append(F.max()); hmean.append(F.mean())
    best = P[int(np.argmax(F))]
    if history == 'mean':
        return sorted(best), hist, hmean
    return (sorted(best), hist) if history else sorted(best)

def enumerate_best(inst, K, subset):
    best, bu = None, -np.inf
    for comb in itertools.combinations(subset, K):
        u = inst.evaluate(list(comb))['U']
        if u > bu: best, bu = list(comb), u
    return best, bu
