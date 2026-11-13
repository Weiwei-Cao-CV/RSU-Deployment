"""Mixed CAV/HDV traffic on the regular network; edge density/speed every 60 s."""
import subprocess, os, sys
try:
    import sumo
    SUMO = sumo.SUMO_HOME
except ImportError:
    SUMO = os.environ['SUMO_HOME']
import sumolib
SUMO_BIN = sumolib.checkBinary('sumo')
def nid(x, y): return f"n{x}_{y}"
ends = {  # end node -> first interior node
    'V1N': ((5000, 15000), (5000, 14000)), 'V1S': ((5000, 0), (5000, 1000)),
    'V2N': ((10000, 15000), (10000, 14000)), 'V2S': ((10000, 0), (10000, 1000)),
    'H1W': ((0, 10000), (1000, 10000)), 'H1E': ((15000, 10000), (14000, 10000)),
    'H2W': ((0, 5000), (1000, 5000)), 'H2E': ((15000, 5000), (14000, 5000))}
rate = {'V1N': 1100, 'V1S': 500, 'V2N': 800, 'V2S': 900, 'H1W': 1200, 'H1E': 600, 'H2W': 700, 'H2E': 1000}
tfac = [0.7, 1.0, 1.25, 0.9, 1.15, 0.8]          # 600 s blocks: time-varying demand
VTYPES = '''  <vType id="HDV" vClass="passenger" carFollowModel="Krauss" accel="2.6" decel="4.5" sigma="0.5" tau="1.2" minGap="2.0" speedFactor="normc(1,0.12,0.6,1.3)" color="0.6,0.6,0.6"/>
  <vType id="CAV" vClass="passenger" carFollowModel="IDM" accel="2.6" decel="4.5" sigma="0" tau="0.6" minGap="2.0" speedFactor="normc(1,0.05,0.8,1.2)" color="0,0.6,0.3"/>
'''
def write_routes(theta, fn):
    with open(fn, 'w') as f:
        f.write('<routes>\n' + VTYPES)
        f.write(f'  <vTypeDistribution id="mix"><vType id="CAVd" refId="CAV" probability="{theta}"/>'
                f'<vType id="HDVd" refId="HDV" probability="{1-theta}"/></vTypeDistribution>\n')
        flows = []
        for o, (a, b) in ends.items():
            frm = f"{nid(*a)}__{nid(*b)}"
            dests = [d for d in ends if d != o]
            for d in dests:
                da, db = ends[d]
                to = f"{nid(*db)}__{nid(*da)}"
                for blk, fac in enumerate(tfac):
                    vph = rate[o] * fac / len(dests)
                    flows.append((blk * 600, f'  <flow id="{o}_{d}_{blk}" type="mix" begin="{blk*600}" end="{blk*600+600}" '
                                 f'from="{frm}" to="{to}" vehsPerHour="{vph:.1f}" departLane="best" departSpeed="max"/>\n'))
        for _, s in sorted(flows, key=lambda t: t[0]):
            f.write(s)
        f.write('</routes>\n')
def run(theta, seed=42):
    tag = f"th{int(theta*100):02d}_s{seed}"
    write_routes(theta, f'{tag}.rou.xml')
    with open(f'{tag}.add.xml', 'w') as f:
        f.write(f'<additional><edgeData id="ed" file="{tag}.edge.xml" period="60" begin="0" end="3600"/></additional>\n')
    subprocess.run([SUMO_BIN, '-n', 'reg.net.xml', '-r', f'{tag}.rou.xml', '-a', f'{tag}.add.xml',
                    '--begin', '0', '--end', '3600', '--step-length', '1', '--seed', str(seed),
                    '--no-step-log', 'true', '--no-warnings', 'true', '--time-to-teleport', '300',
                    '--fcd-output', f'{tag}.fcd.xml', '--fcd-output.attributes', 'x,y,speed,type',
                    '--device.fcd.period', '60'], check=True)
    return tag
if __name__ == '__main__':
    seed = int(os.environ.get('SEED', 42))
    for th in [float(a) for a in sys.argv[1:]] or [0.05, 0.15, 0.30, 0.45]:
        print(run(th, seed), flush=True)
