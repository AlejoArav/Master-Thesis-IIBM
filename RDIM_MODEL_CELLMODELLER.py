import random
from CellModeller.Regulation.ModuleRegulator import ModuleRegulator
from CellModeller.Biophysics.BacterialModels.CLBacterium import CLBacterium
import numpy
import math

from CellModeller.Integration.CLCrankNicIntegrator import CLCrankNicIntegrator
from CellModeller.Integration.CLEulerSigIntegrator import CLEulerSigIntegrator
from CellModeller.Signalling.GridDiffusion import GridDiffusion

# Define the maximum number of cells
max_cells = 256**2

# Specify parameter for solving diffusion dynamics #Add
grid_dim = (256, 256, 3)  # dimension of diffusion space, unit = number of grid
grid_size = (1, 1, 1)  # grid size
grid_orig = (-128, -128, -1)  # Where to place the diffusion space onto simulation space

n_signals = 2
n_species = 8

# C6e_init = 0.001615
C6e_init = 0.0101
C12e_init = 0.0


def setup(sim):
    # Set biophysics, signalling, and regulation models
    biophys = CLBacterium(sim, jitter_z=False, gamma=1e2)
    sig = GridDiffusion(
        sim,
        n_signals,
        grid_dim,
        grid_size,
        grid_orig,
        [10.0, 10.0],
        initLevels=[C6e_init, C12e_init],
        # initLevels=[0.00355, 1e-6],
    )
    # Here we set up the numerical integration:
    # Crank-Nicholson method:
    integ = CLCrankNicIntegrator(
        sim, n_signals, n_species, max_cells, sig, boundcond="wrap"
    )

    # integ = CLCrankNicIntegrator(
    #    sim, n_signals, n_species, max_cells, sig, boundcond="reflect"
    # )
    # Q: Is the constant keyword for the boundary conditions equivalent to periodic boundary conditions?

    # Alternative is to use the simple forward Euler method:
    # integ = CLEulerSigIntegrator(
    #    sim, n_signals, n_species, max_cells, sig, boundcond="constant"
    # )

    # use this file for reg too
    regul = ModuleRegulator(sim, sim.moduleName)
    # Only biophys and regulation
    sim.init(biophys, regul, sig, integ)

    # Add 4 inital cells at the center of the grid
    sim.addCell(cellType=0, pos=(-5, 0, 0))
    sim.addCell(cellType=0, pos=(5, 0, 0))
    sim.addCell(cellType=0, pos=(0, -5, 0))
    sim.addCell(cellType=0, pos=(0, 5, 0))
    # Add to the corners also
    sim.addCell(cellType=0, pos=(-2.5, -2.5, 0))
    sim.addCell(cellType=0, pos=(-2.5, 2.5, 0))
    sim.addCell(cellType=0, pos=(2.5, -2.5, 0))
    sim.addCell(cellType=0, pos=(2.5, 2.5, 0))
    # And the middle
    sim.addCell(cellType=0, pos=(0, 0, 0))

    if sim.is_gui:
        # Add some objects to draw the models
        from CellModeller.GUI import Renderers

        therenderer = Renderers.GLBacteriumRenderer(sim)
        sim.addRenderer(therenderer)
        # For signal rendering
        sigrend = Renderers.GLGridRenderer(sig, integ)  # Add
        sim.addRenderer(sigrend)  # Add

    sim.pickleSteps = 100


def init(cell):
    # Specify mean and distribution of initial cell size
    cell.targetVol = 1.5 + random.uniform(0.25, 0.25)
    # Specify growth rate of cells
    cell.growthRate = 0.02 * 60
    # Specify initial concentration of chemical species
    # There are 14 species
    cell.species[:] = [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    # Specify initial concentration of signaling molecules
    cell.signals[:] = [
        0.0,
        0.0,
    ]


# 1e-1, # k_tx # p[0]
# 5e0, # k_tl # p[1]
# 5e-1, # d_m # p[2]
# 5e-2, # d_p # p[3]
# 0.01, # KLUXI # p[4]
# 0.01, # KLASI # p[5]
# 0.01, # membrane transport rate of AHL # p[6]
# 0.01, # d_HSLi # p[7]
# 0.001, # d_HSLe # p[8]
# 1/0.10, # K_GR # p[9]
# 1/0.10, # K_GS # p[10]
# 1/0.001, # K_TET # p[11]
# 1/0.100, # K_LAC # p[12]
# 2.0, # n_R # p[13]
# 2.0, # n_S # p[14]
# 5e-5,  # beta_0 # p[15]
# 2.5e-4,  # beta_1 # p[16]
# 1/0.1, # K_1R # p[17]
# 1/0.1, # K_1S # p[18]
# 2.0, # n_REP1 # p[19]
# 4.0, # n_REP2 # p[20]
# 1/0.02, # K_2R # p[21]
# 1/0.02, # K_2S # p[22]
# ((1e-1*5e0)/(5e-1*5e-2))**2 # c # p[23]


params_prefix = """
    const float k_tx = 6.0f;
    const float k_tr = 300.0f;
    const float d_mrna = 30.0f;
    const float d_p = 3.0f;
    const float K_1R = 10.0f;
    const float K_1S = 10.0f;
    const float K_2R = 50.0f;
    const float K_2S = 50.0f;
    const float K_GR = 10.0f;
    const float K_GS = 10.0f;
    const float k_LUXI = 2.4f;
    const float k_LASI = 2.4f;
    const float d_HSLi = 0.6f;
    const float d_HSLe = 0.06f;
    const float K_TET = 1000.0f;
    const float K_LAC = 100.0f;
    const float n_R = 2.0f;
    const float n_S = 2.0f;
    const float n_TET = 2.0f;
    const float n_LAC = 4.0f;
    const float KmHSL = 10.0f;
    const float b_0 = 0.06f;
    const float b_1 = 0.30f;
    const float Dm_HSL = 2.0f;

    float C6i = species[0];
    float C12i = species[1];
    float C6e = signals[0];
    float C12e = signals[1];
    float TET = species[2];
    float LAC = species[3];
    float LUXI = species[4];
    float LASI = species[5];
    float RFP = species[6];
    float GFP = species[7];
    """


def specRateCL():
    global params_prefix

    return (
        params_prefix
        + """
    if (cellType==0){
        rates[0] = k_LUXI*LUXI/(KmHSL + LUXI) + Dm_HSL*(C6e - C6i)*area/gridVolume - d_HSLi * C6i;
        rates[1] = k_LASI*LASI/(KmHSL + LASI) + Dm_HSL*(C12e - C12i)*area/gridVolume - d_HSLi * C12i;
        rates[2] = ((K_GS * K_2S * pow(k_tx*k_tr/(d_mrna*d_p), 2) * pow((K_1S * C12i)/(1 + K_1S * C12i), n_S))/(1 + (K_GS * K_2S * pow(k_tx*k_tr/(d_mrna*d_p), 2) * pow((K_1S * C12i)/(1 + K_1S * C12i), n_S)) + pow((K_LAC * LAC), n_LAC)) * k_tx + b_1)*(k_tr/d_mrna) - 5 * d_p * TET;
        rates[3] = ((K_GR * K_2R * pow(k_tx*k_tr/(d_mrna*d_p), 2) * pow((K_1R * C6i)/(1 + K_1R * C6i), n_R))/(1 + (K_GR * K_2R * pow(k_tx*k_tr/(d_mrna*d_p), 2) * pow((K_1R * C6i)/(1 + K_1R * C6i), n_R)) + pow((K_TET * TET), n_TET)) * k_tx + b_0)*(k_tr/d_mrna) - 5 * d_p * LAC;
        rates[4] = ((K_GR * K_2R * pow(k_tx*k_tr/(d_mrna*d_p), 2) * pow((K_1R * C6i)/(1 + K_1R * C6i), n_R))/(1 + (K_GR * K_2R * pow(k_tx*k_tr/(d_mrna*d_p), 2) * pow((K_1R * C6i)/(1 + K_1R * C6i), n_R)) + pow((K_TET * TET), n_TET)) * k_tx + b_0)*(k_tr/d_mrna) - d_p * LUXI;
        rates[5] = ((K_GS * K_2S * pow(k_tx*k_tr/(d_mrna*d_p), 2) * pow((K_1S * C12i)/(1 + K_1S * C12i), n_S))/(1 + (K_GS * K_2S * pow(k_tx*k_tr/(d_mrna*d_p), 2) * pow((K_1S * C12i)/(1 + K_1S * C12i), n_S)) + pow((K_LAC * LAC), n_LAC)) * k_tx + b_1)*(k_tr/d_mrna) - d_p * LASI;
        rates[6] = ((K_GR * K_2R * pow(k_tx*k_tr/(d_mrna*d_p), 2) * pow((K_1R * C6i)/(1 + K_1R * C6i), n_R))/(1 + (K_GR * K_2R * pow(k_tx*k_tr/(d_mrna*d_p), 2) * pow((K_1R * C6i)/(1 + K_1R * C6i), n_R)) + pow((K_TET * TET), n_TET)) * k_tx + b_0)*(k_tr/d_mrna) - d_p * RFP;
        rates[7] = ((K_GS * K_2S * pow(k_tx*k_tr/(d_mrna*d_p), 2) * pow((K_1S * C12i)/(1 + K_1S * C12i), n_S))/(1 + (K_GS * K_2S * pow(k_tx*k_tr/(d_mrna*d_p), 2) * pow((K_1S * C12i)/(1 + K_1S * C12i), n_S)) + pow((K_LAC * LAC), n_LAC)) * k_tx + b_1)*(k_tr/d_mrna) - d_p * GFP;
    }
    """
    )


def sigRateCL():
    global params_prefix

    return (
        params_prefix
        + """    
    rates[0] = Dm_HSL*(C6i - C6e)*area/gridVolume - d_HSLe * C6e;
    rates[1] = Dm_HSL*(C12i - C12e)*area/gridVolume - d_HSLe * C12e;
    """
    )


def update(cells):
    # Iterate through each cell and flag cells that reach target size for division
    for id, cell in cells.items():
        # Check whether the RFP is greater than GFP
        if cell.species[6] >= cell.species[7]:
            cell.color = [255, 0, 0]
        elif cell.species[6] < cell.species[7]:
            cell.color = [0, 255, 0]
        else:
            cell.color = [0, 0, 255]

        # print(cell.signals[:])
        if cell.volume > cell.targetVol:
            cell.divideFlag = True


def divide(parent, d1, d2):
    # Specify target cell size that triggers cell division
    d1.targetVol = 1.5 + random.uniform(0.25, 0.25)
    d2.targetVol = 1.5 + random.uniform(0.25, 0.25)
