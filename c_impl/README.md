# RDIM Model Simulation (C Implementation)

This directory hosts a standalone, high-performance C translation of `RDIM_MODEL_SIMULATIONS.py`. It mirrors the original stochastic lattice model while removing the Python dependencies and leveraging tight memory layouts plus optional OpenMP parallelism for speed.

## Building

The project uses a tiny Makefile:

```bash
cd c_impl
make            # builds with clang/gcc, single-threaded
make OPENMP=1   # builds with -fopenmp if your toolchain supports it
```

The resulting binary is `rdim_model_simulations`.

## Running

Key runtime flags:

- `--replicas N` – number of replica sweeps (default 1)
- `--tol VALUE` – convergence tolerance (default `1e-6`)
- `--snap STRIDE` – write GFP/RFP snapshots every STRIDE steps (default 1000)
- `--max-steps N` – safety cutoff (defaults to `final_time / dt`)
- `--final-time MIN` – nominal stop time in minutes (default 720)
- `--output DIR` – directory that will contain simulation folders (default `c_rdim_outputs`)
- `--quiet` – suppress per-snapshot logging

Example run (single replica, OpenMP with 8 threads):

```bash
cd c_impl
make OPENMP=1
OMP_NUM_THREADS=8 ./rdim_model_simulations --replicas 2 --output ../c_runs --tol 5e-7
```

Outputs land in timestamped folders: `simulation_<date>_rep<r>_dm<...>_de<...>_job<...>`. Each folder contains:

- `simulation_parameters.txt` – constants, grid specs, diffusion coefficients
- `gfp_rfp_<k>.bin` – binary snapshots storing grid size followed by GFP then RFP doubles (little endian)

## Data format helper

To load a snapshot in Python:

```python
import numpy as np
with open('gfp_rfp_0.bin', 'rb') as fh:
    n = np.fromfile(fh, np.uint64, 1)[0]
    gfp = np.fromfile(fh, np.float64, n).reshape(256, 256)
    rfp = np.fromfile(fh, np.float64, n).reshape(256, 256)
```

## Notes

- The solver keeps the lattice size (`N=256`), biochemical constants, and convergence heuristic from the Python reference.
- OpenMP parallelizes the diffusion stencil, promoter update, and field updates. You can disable it by omitting `OPENMP=1` at build time or by exporting `OMP_NUM_THREADS=1`.
- Random initial conditions use a thread-local xorshift128+ generator so replicas remain reproducible per job even under parallel scheduling.
- The code guards against negative concentrations (like the NumPy `clip`) and writes binary snapshots instead of `.npz` files for minimal IO overhead.

## Turning snapshots into PNGs

Use `bin_to_png.py` once the simulation finishes to convert every `gfp_rfp_<k>.bin` into an 8-bit RGB image (red = RFP, green = GFP, channels independently normalized per file):

```bash
cd c_impl
python bin_to_png.py path/to/simulation_*/ --output png_exports
```

You can target individual `.bin` files or entire folders, and add `--suffix _norm` (or similar) to tag the PNG names. The script depends on `numpy` and `Pillow` (install with `pip install numpy pillow`).

## Live viewer (SDL2 + ImGui)

You can watch a single simulation evolve in real time using SDL2 plus Dear ImGui. This mode is optional and off by default.

1. Install SDL2 development headers (e.g., `brew install sdl2` on macOS, `sudo apt install libsdl2-dev` on Debian/Ubuntu).
2. Fetch Dear ImGui into `third_party/imgui` (or point `IMGUI_DIR` elsewhere):
    ```bash
    cd c_impl
    mkdir -p third_party
    git clone https://github.com/ocornut/imgui.git third_party/imgui
    ```
3. Build with GUI support enabled:
    ```bash
    cd c_impl
    make GUI=1 IMGUI_DIR=third_party/imgui
    ```
    If `sdl2-config` is not on your path, set `SDL2_CFLAGS`/`SDL2_LIBS` manually, e.g. `make GUI=1 SDL2_CFLAGS="-I/usr/local/include/SDL2" SDL2_LIBS="-L/usr/local/lib -lSDL2"`.

Run the simulator with the viewer via `--gui`. GUI mode executes exactly one diffusion/depth combination (selected by indices) to keep the window manageable:

```bash
./rdim_model_simulations --gui --viz-stride 250 --dm-index 3 --de-index 1
```

- `--viz-stride` controls how many solver steps elapse between pushed frames (can also be changed live from the ImGui slider).
- `--dm-index` selects which entry from `DM_COEFFS`/`C6_CONCS` to simulate (0-based).
- `--de-index` selects the entry from `DEXT_COEFFS`.
- GUI runs currently force `--replicas 1` and bypass the batch loops/parallel worker pool.

Inside the viewer you’ll see the current GFP/RFP texture plus live stats (step, simulated minutes, convergence, channel means) and a slider to adjust the stride.

Additional viewer controls:

- **Zoom** slider: scales the displayed lattice anywhere from 1× to 4× with GPU-side interpolation (no extra solver cost).
- **Magnetization plot**: running trace of `(red - green) / (red + green)` per frame, treating red cells as +1 and green cells as -1 to mirror the RDIM magnetization definition.
    - Built-in zero line plus ±1/0 labels keep the scale readable while you scrub.
    - The viewer preallocates the full magnetization history so you can use the *Timeline* slider to inspect any portion of the run, optionally locking the view to the newest samples or widening the window for long-term trends.
