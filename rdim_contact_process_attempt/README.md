# RDIM Contact Process Attempt

This prototype couples the full RDIM biochemical dynamics with a lattice-based cellular growth and division layer on the same 256×256 grid. Cells occupy lattice sites (0 = empty, 1 = occupied). Biochemical fields (GFP/RFP, LuxI/LacI, etc.) update exactly as in the main C implementation, while an additional step applies exponential volume growth and volume-triggered division into neighboring empty sites (no death events).

## Building

```bash
cd rdim_contact_process_attempt
make
```

This produces the `rdim_contact_process` binary.

### Optional GUI build

The prototype can stream frames to an SDL2 + Dear ImGui viewer (R/G channels show reporters; occupancy is shown in a separate pane). Reporter channels are normalized per-frame over occupied cells only, and only the dominant channel per pixel is shown (toggle-like red-or-green). Install SDL2 (e.g., `brew install sdl2` on macOS) and build with the GUI toggle:

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
- `--seed-count` – ignore the random occupancy draw and start with exactly N occupied cells arranged from the center outward (center cell first, then nearest neighbors).
- `--gui`, `--viz-stride` – enable the live viewer and pick the update cadence (requires `make GUI=1`).
- `--dm-index`, `--de-index` – pick membrane permeability and external diffusion coefficients from the legacy grids (otherwise all combinations are simulated).
- `--c6-conc`, `--c12-conc` – override the initial extracellular C6/C12 concentrations (defaults come from the legacy tables: C6 depends on `--dm-index`, C12 defaults to 0).
- Division behavior:
	- `--division-mode`: `B` (default) retries a random ordering of Moore neighbors; `D` picks one random Moore neighbor (no retries); `E` picks one random site within a Chebyshev radius.
	- `--division-radius`: layers for Mode `E` (default 1).
	- `--move-prob`: for Modes `D/E`, probability to attempt a movement into a random target right after a successful division (default 1.0).
- Nutrient-limited growth (Petri dish model):
	- `--nutrient-init` sets the initial nutrient concentration everywhere (default 1.0). A diffusing nutrient field is updated each step with diffusion (`--nutrient-diffusion`, default 500/60 per minute), decay (`--nutrient-decay`, default 0), and uptake proportional to local cell length and saturating with `n/(kappa+n)` (`--nutrient-uptake`, default 2/60 per minute).
	- Growth law uses `dl/dt = g * l * n/(kappa+n) * (1 - xi*T)` with toxin term disabled by default (`xi=0`). `--kappa` sets the half-saturation (default 0.333).
	- Division length is drawn per cell from a truncated normal (`--division-mean`, `--division-sd`, `--division-min`, `--division-max`), and mother/daughter lengths are reset to half of their newly sampled targets on division.
- `--snap`, `--final-time`, `--max-steps`, `--replicas`, `--quiet` – same semantics as the base solver.

Extracellular flux scaling: membrane exchange into the external grids is scaled by the global occupancy fraction (total occupied sites / total sites) to mimic dilution in a finite chamber; when few cells are present, their export has a smaller impact on the bulk field.

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
	- Extracellular fields (C6e/C12e) are **not** copied on division; they remain on their own diffusion grids and are updated only by diffusion, decay, and membrane exchange.

Internally, volumes are dimensionless and measured in units of a “typical cell volume”. If you want to map to a physical scale, you can interpret `V = 1` as roughly one *E. coli*-sized cell (∼$10^{-15}$ L), so `V_\text{div} = 2` corresponds to doubling in size before division.

Each successful run produces a timestamped folder containing:

- `simulation_parameters.txt` with biochemical, diffusion, and contact-process settings.
- `gfp_rfp_k.bin` snapshots (compatible with `c_impl/bin_to_png.py`).
- `cells_k.bin` occupancy bitmaps (uint8) aligned with the GFP/RFP grids for downstream analysis.

This folder intentionally isolates the experimental growth logic so we can iterate without disturbing the production solver while still reusing its constants and IO utilities.
