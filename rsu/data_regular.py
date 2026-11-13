"""Load SUMO edgeData of the regular network into undirected link states."""
import xml.etree.ElementTree as ET, numpy as np, re, collections
from scipy.sparse.csgraph import shortest_path

def build_graph():
    nodes = []
    for X in (5000, 10000):
        nodes += [(X, 15000 - 1000 * k) for k in range(16)]
    for Y in (10000, 5000):
        nodes += [(1000 * k, Y) for k in range(16)]
    labels = collections.OrderedDict()
    for lab, xy in enumerate(nodes, 1):
        labels.setdefault(xy, []).append(lab)
    xy = list(labels)
    idx = {p: i for i, p in enumerate(xy)}
    links = set()
    roads = [nodes[16 * r:16 * r + 16] for r in range(4)]
    for r in roads:
        for a, b in zip(r[:-1], r[1:]):
            links.add(tuple(sorted((idx[a], idx[b]))))
    links = sorted(links)
    n = len(xy)
    W = np.full((n, n), np.inf)
    for i, j in links:
        W[i, j] = W[j, i] = 1.0
    D = shortest_path(W, directed=False)
    names = ['/'.join(map(str, labels[p])) for p in xy]
    return xy, names, links, D

def load_states(edge_xml, links, xy, t0=900, t1=3600, period=None):
    """returns list of periods; each a list of (k_two_way, v) per link. period=None -> one mean state"""
    idx = {f"n{x}_{y}": i for i, (x, y) in enumerate(xy)}
    root = ET.parse(edge_xml).getroot()
    rows = []
    for iv in root.findall('interval'):
        b = float(iv.get('begin'))
        if b < t0 or b >= t1:
            continue
        st = collections.defaultdict(lambda: [0.0, 0.0])   # link -> [density sum, density*speed]
        for e in iv.findall('edge'):
            m = re.match(r'(n\d+_\d+)__(n\d+_\d+)$', e.get('id'))
            if not m:
                continue
            key = tuple(sorted((idx[m.group(1)], idx[m.group(2)])))
            k = float(e.get('density', 0) or 0); v = float(e.get('speed', 0) or 0)
            if not e.get('speed'):
                v = 16.67
            st[key][0] += k; st[key][1] += k * v
        rows.append((b, {l: (s[0], s[1] / s[0] if s[0] > 0 else 16.67) for l, s in st.items()}))
    def agg(rs):
        out = []
        for l in links:
            ks = np.array([r[l][0] if l in r else 0.0 for _, r in rs])
            vs = np.array([r[l][1] if l in r else 16.67 for _, r in rs])
            k = ks.mean(); v = (ks * vs).sum() / ks.sum() if ks.sum() > 0 else 16.67
            out.append((k, v))
        return out
    if period is None:
        return [agg(rows)]
    groups = collections.OrderedDict()
    for b, r in rows:
        groups.setdefault(int((b - t0) // period), []).append((b, r))
    return [agg(g) for g in groups.values()]
