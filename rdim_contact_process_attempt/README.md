# RDIM Contact Process Attempt

This prototype couples the full RDIM biochemical dynamics with a lattice-based cellular growth and division layer on the same 256×256 grid. Cells occupy lattice sites (0 = empty, 1 = occupied). Biochemical fields (GFP/RFP, LuxI/LacI, etc.) update exactly as in the main C implementation, while an additional step applies exponential volume growth and volume-triggered division into neighboring empty sites (no death events).

## Building

```bash
cd rdim_contact_process_attempt
make
```

This produces the `rdim_contact_process` binary.

### Optional GUI build

The prototype can stream frames to an SDL2 + Dear ImGui viewer (blue channel = occupied cells). Install SDL2 (e.g., `brew install sdl2` on macOS) and build with the GUI toggle:

```bash
cd rdim_contact_process_attempt
make GUI=1
```

Launch the viewer with `--gui`; pick a diffusion pair via `--dm-index/--de-index` (defaults to index 0 if omitted) and optionally adjust the refresh stride:

```bash
./rdim_contact_process --gui --viz-stride 250
```

The ImGui control panel mirrors the magnetization timeline from the main solver, adds live occupancy statistics, and exposes a stride slider plus 1×–4× zoom.

## Running

```bash
./rdim_contact_process \
	--final-time 720 \
	--snap 1000 \
	--init-occupancy 0.5
```

Key CLI flags (mirroring the main RDIM solver plus the new growth controls):

- `--birth-rate` – legacy contact-process parameter λ (currently unused; colonization is driven purely by volume-based division).
- `--init-occupancy` – fraction of lattice sites seeded with cells at `t=0`.
- `--seed-center` – ignore the random occupancy draw and start with a single occupied cell at the lattice center (random AHL background is still sampled).
- `--gui`, `--viz-stride` – enable the live viewer and pick the update cadence (requires `make GUI=1`).
- `--dm-index`, `--de-index` – pick membrane permeability and external diffusion coefficients from the legacy grids (otherwise all combinations are simulated).
- `--snap`, `--final-time`, `--max-steps`, `--replicas`, `--quiet` – same semantics as the base solver.

### Volume growth and division model

- Each occupied pixel has an associated volume `V` (initially 1.0 for new cells, 0.0 for empty sites).
- Volume grows exponentially according to

	$$ V(t + \Delta t) = V(t) \exp(g \Delta t) $$

	with growth rate `g` (per minute). In the current implementation we use `g \approx \ln(2)/30 \approx 0.023\,\text{min}^{-1}` by default, which corresponds to a division time of roughly 30 minutes when `V` grows from 1 to 2.
- When `V` exceeds a critical division volume `V_\text{div}` (currently `V_div = 2.0`), the cell attempts to divide:
	- A random ordering of the 8 neighboring pixels is sampled.
	- The first empty neighbor in this ordering becomes the daughter site; if no neighbors are empty, division is skipped for that timestep.
	- Mother and daughter volumes are both set to `V_\text{div}/2` (i.e. 1.0), representing symmetric division.
	- All intracellular concentrations (LuxI/LacI, GFP/RFP, etc.) are copied from the mother to the daughter. Since we work in concentration units and split volume symmetrically, this corresponds to a deterministic, symmetric partition of molecular content that preserves concentration.
	- External fields (C6e/C12e) are also copied for convenience, but diffusion and reactions quickly smooth these profiles.

Internally, volumes are dimensionless and measured in units of a “typical cell volume”. If you want to map to a physical scale, you can interpret `V = 1` as roughly one *E. coli*-sized cell (∼$10^{-15}$ L), so `V_\text{div} = 2` corresponds to doubling in size before division.

Each successful run produces a timestamped folder containing:

- `simulation_parameters.txt` with biochemical, diffusion, and contact-process settings.
- `gfp_rfp_k.bin` snapshots (compatible with `c_impl/bin_to_png.py`).
- `cells_k.bin` occupancy bitmaps (uint8) aligned with the GFP/RFP grids for downstream analysis.

This folder intentionally isolates the experimental growth logic so we can iterate without disturbing the production solver while still reusing its constants and IO utilities.
