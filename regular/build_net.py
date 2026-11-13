"""Regular test network: 2 vertical + 2 horizontal roads, 15 km each, crossing at 5 km and 10 km.
Nodes every 1 km (60 unique locations, 64 labels). Two lanes per direction, 60 km/h, signalized crossings."""
import subprocess, os
import sumolib
nodes, edges = {}, []
def nid(x, y): return f"n{x}_{y}"
roads = []
for X in (5000, 10000):            # vertical roads, label 1/17 at the top (y = 15000)
    roads.append([(X, 15000 - 1000 * k) for k in range(16)])
for Y in (10000, 5000):            # horizontal roads, label 33/49 at the left
    roads.append([(1000 * k, Y) for k in range(16)])
for r in roads:
    for (x, y) in r:
        nodes[nid(x, y)] = (x, y)
    for a, b in zip(r[:-1], r[1:]):
        edges.append((nid(*a), nid(*b)))
cross = {nid(5000, 10000), nid(10000, 10000), nid(5000, 5000), nid(10000, 5000)}
with open('reg.nod.xml', 'w') as f:
    f.write('<nodes>\n')
    for n, (x, y) in nodes.items():
        t = 'traffic_light' if n in cross else 'priority'
        f.write(f'  <node id="{n}" x="{x}" y="{y}" type="{t}"/>\n')
    f.write('</nodes>\n')
with open('reg.edg.xml', 'w') as f:
    f.write('<edges>\n')
    for a, b in edges:
        for u, v in ((a, b), (b, a)):
            f.write(f'  <edge id="{u}__{v}" from="{u}" to="{v}" numLanes="2" speed="16.67"/>\n')
    f.write('</edges>\n')
subprocess.run([sumolib.checkBinary('netconvert'), '-n', 'reg.nod.xml', '-e', 'reg.edg.xml', '-o', 'reg.net.xml',
                '--no-turnarounds', 'true', '--tls.cycle.time', '90', '--no-warnings', 'true'], check=True)
print(len(nodes), 'nodes', len(edges) * 2, 'directed edges')
