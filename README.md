# Cost-Aware RSU Deployment with Mixed Traffic and Information Lifetime

Code for the paper *Cost-Aware Roadside Unit Deployment Considering Mixed Traffic and Information Lifetime Constraints* (submitted to *Sustainability*).

The repository contains the information propagation model, the RSU deployment model, the solvers, and the regular test network used in the paper.

## Contents

| Path | Description |
|---|---|
| `rsu/model.py` | Propagation model (immediate V2V forwarding and store-carry-forward), reachability matrix within the information lifetime, node demand weights, MST cable length, and comprehensive benefit |
| `rsu/solvers.py` | Genetic algorithm, greedy selection, greedy selection with swap local search, and exhaustive enumeration |
| `rsu/data_regular.py` | Discrete graph of the regular network and loading of SUMO edge data |
| `regular/build_net.py` | Builds the regular network (two vertical and two horizontal 15 km roads, nodes every 1 km) |
| `regular/run_sim.py` | Simulates mixed CAV/HDV traffic in SUMO and writes link density and speed every 60 s |
| `run_regular.py` | Runs the experiments on the regular network |

## Requirements

Python 3.10 or later and SUMO 1.20 or later.

```bash
pip install -r requirements.txt
```

## Usage

1. Simulate traffic on the regular network (CAV penetration 5%, 15%, 30% and 45%):

   ```bash
   cd regular
   SEED=42 python run_sim.py 0.05 0.15 0.30 0.45
   cd ..
   ```

   The network files `reg.*.xml` are included; `python regular/build_net.py` rebuilds them.

2. Optimize and evaluate the layouts:

   ```bash
   python run_regular.py --seed 42
   ```

   Results are written to `results/regular/`. The paper uses seeds 42, 1, 2, 3, 4 and 5.

The main parameters (communication range `R`, cable cost `alpha`, RSU cost `beta`, reaction times, etc.) are set in `P` at the top of `rsu/model.py`.

## Data

The traffic demand of the Tianjin case is not included in this repository.

## License

MIT License. See `LICENSE`.
