"""Experiments on the regular network: GA convergence against enumeration, layouts at different CAV penetration
rates and information lifetimes, and the sweep over the number of RSUs.
Traffic states are the mean link density and speed over 900 to 3,600 s.
usage: python run_regular.py --seed 42"""
import argparse, os, sys, time, itertools, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rsu'))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import model, data_regular, solvers

THETAS = (0.05, 0.15, 0.30, 0.45)
TAUS = (5, 10, 20, 30)
GA = dict(pop=150, gens=200)


def instance(theta, tau, seed, cand=None):
    model.P['R'] = float(os.environ.get('RSU_R', model.P['R']))
    xy, names, links, D = data_regular.build_graph()
    st = data_regular.load_states(f'regular/th{int(round(theta * 100)):02d}_s{seed}.edge.xml', links, xy)[0]
    inst = model.Instance([(i, j, 1000.0) for i, j in links], st, theta, tau, len(xy), D, cand)
    inst.names, inst.links, inst.xy, inst.state = names, links, np.array(xy, float), st
    return inst


def describe(inst, sel):
    e = inst.evaluate(sel)
    return dict(U=e['U'], B=e['B'], cable_cost=model.P['alpha'] * e['L'], device_cost=model.P['beta'] * e['K'],
                cable_km=e['L'], K=e['K'], reach_nodes=int(inst.coverage(sel).sum()), CAV_coverage=e['cav_cov'],
                nodes=' '.join(inst.names[i] for i in sorted(sel)))


def job_convergence(args):                     # Fig. 6: GA vs enumeration on a local candidate subset
    theta, seed, nsub, K = args
    full = instance(theta, 10, seed)
    sub = sorted(np.argsort(-full.w)[:nsub])    # the nsub nodes with the largest CAV demand
    inst = instance(theta, 10, seed, cand=sub)
    best, bu = solvers.enumerate_best(inst, K, sub)
    sel, hist, hmean = solvers.ga(inst, K, history='mean', **GA)
    return dict(theta=theta, enum_U=bu, enum_nodes=' '.join(inst.names[i] for i in sorted(best)),
                ga_U=hist[-1], ga_first_gen_at_best=int(np.argmax(np.isclose(hist, bu))) if np.isclose(hist[-1], bu) else None,
                hist=hist, hist_mean=hmean)


def job_fixed(args):                           # one GA run for (theta, tau, K), plus greedy+LS as a check
    theta, tau, K, seed = args
    inst = instance(theta, tau, seed)
    t = time.time(); sel = solvers.ga(inst, K, **GA); sec = time.time() - t
    ref = solvers.swap_ls(inst, solvers.greedy(inst, K))
    r = dict(theta=theta, tau=tau, K_fixed=K, ga_seconds=sec, **describe(inst, sel))
    r['U_greedy_ls'] = inst.evaluate(ref)['U']
    return r


def fig8(seed, theta=0.30):
    inst = instance(theta, 10, seed)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.6))
    k = np.array([s[0] for s in inst.state]); v = np.array([s[1] for s in inst.state]) * 3.6
    for ax, val, lab, cmap in ((axs[0], k, 'Average density (veh/km)', 'PiYG_r'), (axs[1], v, 'Mean speed (km/h)', 'RdYlGn')):
        norm = matplotlib.colors.Normalize(val.min(), val.max()); cm = plt.get_cmap(cmap)
        for (i, j), x in zip(inst.links, val):
            ax.plot(*inst.xy[[i, j]].T / 1000, color=cm(norm(x)), lw=6, solid_capstyle='butt')
        ax.set_aspect('equal'); ax.axis('off')
        plt.colorbar(matplotlib.cm.ScalarMappable(norm, cm), ax=ax, shrink=0.7, label=lab)
    plt.tight_layout(); plt.savefig(f'results/regular/fig8_traffic_s{seed}.png', dpi=300); plt.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--workers', type=int, default=18)
    ap.add_argument('--R', type=float, default=500.0, help='communication range (m)')
    a = ap.parse_args()
    model.P['R'] = a.R; os.environ['RSU_R'] = str(a.R)
    os.makedirs('results/regular', exist_ok=True)
    out = f'results/regular'; tag = f's{a.seed}' + ('' if a.R == 500 else f'_R{a.R:g}')
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 11})
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        conv = list(ex.map(job_convergence, [(th, a.seed, 20, 5) for th in THETAS]))
        pen = list(ex.map(job_fixed, [(th, 10, 5, a.seed) for th in THETAS]))                       # Tables 3-4
        life = list(ex.map(job_fixed, [(0.30, tau, 5, a.seed) for tau in TAUS]))                    # Table 5
        sweep = list(ex.map(job_fixed, [(0.30, tau, K, a.seed) for tau in TAUS for K in range(1, 25)]))  # Fig. 7
    print(f'optimisation done in {time.time()-t0:.0f}s', flush=True)

    # Fig. 6
    fig, axs = plt.subplots(2, 2, figsize=(9, 6.5));
    for ax, c in zip(axs.flat, conv):
        ax.plot(range(len(c['hist'])), c['hist'], color='#1f4fd8', label='Comprehensive benefit (GA)')
        ax.axhline(c['enum_U'], ls='--', color='#d62728', label='Optimum by enumeration')
        ax.set_title(f"CAV penetration {c['theta']:.0%}"); ax.set_xlabel('Generation'); ax.set_ylabel('Comprehensive benefit')
    axs[0, 0].legend(frameon=False, fontsize=9); plt.tight_layout(); plt.savefig(f'{out}/fig6_ga_convergence_{tag}.png', dpi=300); plt.close()
    pd.DataFrame([{k: v for k, v in c.items() if not k.startswith('hist')} for c in conv]).to_csv(f'{out}/fig6_enumeration_{tag}.csv', index=False)

    pd.DataFrame(pen).to_csv(f'{out}/table3_4_penetration_{tag}.csv', index=False)
    pd.DataFrame(life).to_csv(f'{out}/table5_lifetime_{tag}.csv', index=False)
    sw = pd.DataFrame(sweep); sw.to_csv(f'{out}/fig7_rsu_sweep_{tag}.csv', index=False)

    # Fig. 7
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    for tau, mk, c in zip(TAUS, 'sodD', ('#d62728', '#3fa9f5', '#2ca02c', '#111111')):
        d = sw[sw.tau == tau]
        axs[0].plot(d.K_fixed, d.U, marker=mk, color=c, ms=4, label=f'τ = {tau} s')
        k = d.loc[d.U.idxmax()]; axs[0].plot(k.K_fixed, k.U, marker='*', color=c, ms=14, mec='k')
    axs[0].set_xlabel('Number of RSUs'); axs[0].set_ylabel('Comprehensive benefit'); axs[0].legend(frameon=False)
    d = sw[sw.tau == 10]
    axs[1].plot(d.K_fixed, d.B, marker='o', ms=4, label='Coverage benefit'); axs[1].plot(d.K_fixed, d.U, marker='s', ms=4, label='Comprehensive benefit')
    axs[1].set_xlabel('Number of RSUs'); axs[1].set_ylabel('Benefit'); axs[1].legend(frameon=False)
    plt.tight_layout(); plt.savefig(f'{out}/fig7_rsu_lifetime_{tag}.png', dpi=300); plt.close()
    fig8(a.seed) if a.R == 500 else None

    pd.set_option('display.width', 220); pd.set_option('display.max_colwidth', 60)
    print('\nFig 6 (enumeration vs GA, 20-node subset, K=5, tau=10):')
    print(pd.DataFrame([{k: v for k, v in c.items() if not k.startswith('hist')} for c in conv]).round(2).to_string(index=False))
    cols = ['theta', 'tau', 'U', 'B', 'cable_cost', 'device_cost', 'reach_nodes', 'CAV_coverage', 'U_greedy_ls', 'nodes']
    print('\nTables 3-4 (K=5, tau=10):'); print(pd.DataFrame(pen)[cols].round(2).to_string(index=False))
    print('\nTable 5 (theta=0.3, K=5):'); print(pd.DataFrame(life)[cols].round(2).to_string(index=False))
    print('\nFig 7 optimum number of RSUs (theta=0.3):')
    print(sw.loc[sw.groupby('tau').U.idxmax(), ['tau', 'K_fixed', 'U', 'B', 'reach_nodes', 'nodes']].round(2).to_string(index=False))
    print('max |GA - greedy+LS| over all runs:', (sw.U - sw.U_greedy_ls).abs().max().round(3))
