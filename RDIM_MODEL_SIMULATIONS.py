# Import the necessary packages
import numpy as np  # For numerical operations
import os  # For manipulating files and folders
import matplotlib.pyplot as plt  # For plotting
from pylab import *  # For the random number calculations
from numba import jit, prange  # For speeding up the simulation
import multiprocessing as mp  # For multiprocessing
import datetime  # For getting the current date and time

# Set the default parameters for matplotlib
plt.rcParams.update(plt.rcParamsDefault)
# Don't display plots automatically, this is done in order to save RAM
plt.ioff()

# ----------------------------------------------------------------------------------------------------------------------
# Parameter Definitions
# ----------------------------------------------------------------------------------------------------------------------
# Transcription and translation parameters
k_tx = 1e-1  # uM/min
k_tr = 5e0  # min-1
d_mrna = 5e-1  # min-1
d_p = 5e-2  # min-1

# LuxR / LasR and C6i / C12i parameters
K_1R = 1 / 0.1  # uM^-1
K_1S = 1 / 0.1  # uM^-1
K_2R = 1 / 0.02  # uM^-1
K_2S = 1 / 0.02  # uM^-1
K_GR = 1 / 0.10  # uM^-1
K_GS = 1 / 0.10  # uM^-1
KM_ENZ = 10.0  # uM

# LuxI and LasI parameters
k_LUXI = 0.04  # min-1
k_LASI = 0.04  # min-1

# HSL parameters
d_HSLi = 0.01  # min-1
d_HSLe = 0.001  # min-1

# Promoter parameters
K_TET = 1 / 0.001  # uM^-1
K_LAC = 1 / 0.100  # uM^-1
n_R = 2.0
n_S = 2.0
n_REP1 = 2.0
n_REP2 = 4.0
b_0 = 5e-5  # No units
b_1 = 2.5e-4  # No units

# Calculate the c constant
c = ((k_tx * k_tr) / (d_mrna * d_p)) ** 2
# Calculate the alpha constant
alpha = (k_tx * k_tr) / (d_mrna * d_p)

QSS_txtl = (k_tx * k_tr) / d_mrna

# Grid specs (300x300)
domain_size = 256  # In meters (3e-4 is 300 micrometers)
N = 256  # N squares by N squares
Dh = (
    domain_size / N
)  # Spatial resolution (The length of one of the sides of the square)
dh2_inv = 1.0 / (Dh**2)  # Inverse of Dh^2
numberOfSquares = N * N  # Number of square (each one is Dh by Dh)

V_cell = 1.0  # In cubic micrometers and considering a squished cell (1x1x1 um)
V_ext = domain_size**2  # Volume of ONE of the squares of the lattice (cubic meters)
N_cells = (
    N * N
)  # cells per square, assuming a cell can occupy a space defined by 1x1 um

# Timestep duration
Dt = 0.001  # minutes


# ----------------------------------------------------------------------------------------------------------------------
# Auxiliary functions definitions
# ----------------------------------------------------------------------------------------------------------------------
@jit(nopython=True, fastmath=True, parallel=True)
def laplacian_PBC(field, dh2_inv, out):
    """
    Compute the laplacian of an array using a 9 point stencil.
    """
    # Note the new range: 0, ..., N-1
    for i in prange(N):
        for j in prange(N):
            out[i, j] = (
                0.5 * field[(i + 1) % N, j]
                + 0.5 * field[(i - 1) % N, j]
                + 0.5 * field[i, (j + 1) % N]
                + 0.5 * field[i, (j - 1) % N]
                + 0.25 * field[(i + 1) % N, (j + 1) % N]
                + 0.25 * field[(i + 1) % N, (j - 1) % N]
                + 0.25 * field[(i - 1) % N, (j + 1) % N]
                + 0.25 * field[(i - 1) % N, (j - 1) % N]
                - 3 * field[i, j]
            ) * dh2_inv

    return out


@jit(nopython=True, fastmath=True, parallel=True)
def promoterFunction(HSL, REP, K_GENE_HSL, K_DIMER, K_TF_HSL, K_REP, n_HSL, n_REP, C):
    """
    This function returns the value of the promoter function
    Inputs:
        - HSL1: Concentration of HSL1
        - REP: Concentration of the repressor
        - b_0: Basal transcription rate for the activated promoter
        - K_A: Association constant of the activator
        - K_1A: Association constant of the activator to HSL
        - K_REP: Association constant of the repressor to the promoter
        - n_A: Hill coefficient of the activator
        - c: Tx/Tl constant (assuming QSS)
    Returns:
        - The value of the promoter function
    """
    A = K_GENE_HSL * K_DIMER * C * ((K_TF_HSL * HSL) / (1 + K_TF_HSL * HSL)) ** n_HSL
    D = 1 / (1 + A + (K_REP * REP) ** n_REP)

    return A * D


# ----------------------------------------------------------------------------------------------------------------------
# Main functions (initialize and update)
# ----------------------------------------------------------------------------------------------------------------------
def initialize(final_time, samples, C6_conc, C12_conc):
    """
    This function initializes the system, and returns the initial conditions
    Inputs:
        - final_time: Final time of the simulation (hours)
        - samples: Number of samples to be taken
        - C6_conc: Factor by which the concentration of C6 is increased
        - C12_conc: Factor by which the concentration of C12 is increased
    Returns:
        - All the initial conditions for the system
    """

    # INITIALIZE THE SYSTEM
    # ----------------------
    # First we set the numpy random seed
    np.random.seed()

    # Enzymes
    # -------
    LUXI = np.zeros([N, N], dtype=np.float64)
    LASI = np.zeros([N, N], dtype=np.float64)

    # C6 and C12 EXTRACELLULAR concentrations, add some uniform noise to the initial conditions
    # We begin with 0.01 nanomolar (1E-5 micromolar; 1E-11 molar) concentrations of both C6 and C12
    C6i = np.zeros([N, N], dtype=np.float64)
    C12i = np.zeros([N, N], dtype=np.float64)
    # Generate a random gaussian distribution of size N x N centered at the center of the C6_e and C12_e
    C6e = np.random.normal(C6_conc, C6_conc * 0.25, size=[N, N])
    C12e = np.random.normal(C12_conc, C12_conc * 0.25, size=[N, N])

    # TetR and LacI repressor
    # -----------------------
    TETR = np.zeros([N, N], dtype=np.float64)
    LACI = np.zeros([N, N], dtype=np.float64)

    # RFP and GFP concentrations
    RFP = np.zeros([N, N], dtype=np.float64)
    GFP = np.zeros([N, N], dtype=np.float64)

    # Promoters
    # ---------
    pLT = np.zeros([N, N], dtype=np.float64)
    pLL = np.zeros([N, N], dtype=np.float64)

    # --------------------------------------------------------------------------------
    # Create the next arrays (for the next time step)
    # --------------------------------------------------------------------------------

    # Enzymes
    # -------
    LUXI_next = np.zeros([N, N], dtype=np.float64)
    LASI_next = np.zeros([N, N], dtype=np.float64)

    # C6 and C12 EXTRACELLULAR concentrations
    C6i_next = np.zeros([N, N], dtype=np.float64)
    C12i_next = np.zeros([N, N], dtype=np.float64)
    C6e_next = np.zeros([N, N], dtype=np.float64)
    C12e_next = np.zeros([N, N], dtype=np.float64)

    # TetR and LacI repressor
    # -----------------------
    TETR_next = np.zeros([N, N], dtype=np.float64)
    LACI_next = np.zeros([N, N], dtype=np.float64)

    # RFP and GFP concentrations
    RFP_next = np.zeros([N, N], dtype=np.float64)
    GFP_next = np.zeros([N, N], dtype=np.float64)

    # --------------------------------------------------------------------------------
    # Arrays used for storing variables
    # --------------------------------------------------------------------------------
    # For the laplacian of C6
    lap_C6 = np.zeros([N, N], dtype=np.float64)
    # For the laplacian of C12
    lap_C12 = np.zeros([N, N], dtype=np.float64)

    # Timesteps
    T = final_time  # minutes
    n_steps = int(T / Dt)

    # Samples
    sample_times = [int(j) for j in np.linspace(0, n_steps, samples)]

    # Return everything
    return (
        LUXI,
        LASI,
        C6i,
        C12i,
        C6e,
        C12e,
        TETR,
        LACI,
        RFP,
        GFP,
        LUXI_next,
        LASI_next,
        C6i_next,
        C12i_next,
        C6e_next,
        C12e_next,
        TETR_next,
        LACI_next,
        RFP_next,
        GFP_next,
        lap_C6,
        lap_C12,
        pLT,
        pLL,
        sample_times,
        n_steps,
    )


# Update function
@jit(nopython=True, fastmath=True)
def update_vars(
    LUXI,
    LASI,
    C6i,
    C12i,
    C6e,
    C12e,
    TETR,
    LACI,
    RFP,
    GFP,
    LUXI_next,
    LASI_next,
    C6i_next,
    C12i_next,
    C6e_next,
    C12e_next,
    TETR_next,
    LACI_next,
    RFP_next,
    GFP_next,
    lap_C6,
    lap_C12,
    pLT,
    pLL,
    Dm_HSL,
    De_HSL,
):
    # Compute the laplacian of C6 and C12 in the extracellular space
    lap_C6 = laplacian_PBC(C6e, dh2_inv, out=lap_C6)
    laplacian_PBC(C12e, dh2_inv, out=lap_C12)

    # Calculate the numerator and denominator of the phiProm_LT
    pLT = promoterFunction(C6i, TETR, K_GR, K_2R, K_1R, K_TET, n_R, n_REP1, c)
    pLL = promoterFunction(C12i, LACI, K_GS, K_2S, K_1R, K_LAC, n_S, n_REP2, c)

    # Update the variables (only the inner grid)

    # C6i HSL
    C6i_next = (
        C6i
        + (((k_LUXI * LUXI) / (KM_ENZ + LUXI)) + Dm_HSL * (C6e - C6i) - d_HSLi * C6i)
        * Dt
    )
    # C12i HSL
    C12i_next = (
        C12i
        + (((k_LASI * LASI) / (KM_ENZ + LASI)) + Dm_HSL * (C12e - C12i) - d_HSLi * C12i)
        * Dt
    )
    # C6e HSL
    C6e_next = C6e + (Dm_HSL * (C6i - C6e) + De_HSL * lap_C6 - d_HSLe * C6e) * Dt
    # C12e HSL
    C12e_next = C12e + (Dm_HSL * (C12i - C12e) + De_HSL * lap_C12 - d_HSLe * C12e) * Dt
    # TetR repressor
    TETR_next = TETR + ((pLL * k_tx + b_1) * (k_tr / d_mrna) - d_p * TETR) * Dt
    # LacI repressor
    LACI_next = LACI + ((pLT * k_tx + b_0) * (k_tr / d_mrna) - d_p * LACI) * Dt
    # LuxI
    LUXI_next = LUXI + ((pLT * k_tx + b_0) * (k_tr / d_mrna) - d_p * LUXI) * Dt
    # LasI
    LASI_next = LASI + ((pLL * k_tx + b_1) * (k_tr / d_mrna) - d_p * LASI) * Dt
    # RFP
    RFP_next = RFP + ((pLT * k_tx + b_0) * (k_tr / d_mrna) - d_p * RFP) * Dt
    # GFP
    GFP_next = GFP + ((pLL * k_tx + b_1) * (k_tr / d_mrna) - d_p * GFP) * Dt

    # All arrays must be positive (use numpy clip)
    np.clip(C6i_next, 0, None, out=C6i_next)
    np.clip(C12i_next, 0, None, out=C12i_next)
    np.clip(C6e_next, 0, None, out=C6e_next)
    np.clip(C12e_next, 0, None, out=C12e_next)
    np.clip(TETR_next, 0, None, out=TETR_next)
    np.clip(LACI_next, 0, None, out=LACI_next)
    np.clip(LUXI_next, 0, None, out=LUXI_next)
    np.clip(LASI_next, 0, None, out=LASI_next)
    np.clip(RFP_next, 0, None, out=RFP_next)
    np.clip(GFP_next, 0, None, out=GFP_next)

    # Update the arrays and next arrays
    C6i, C6i_next = C6i_next, C6i
    C12i, C12i_next = C12i_next, C12i
    C6e, C6e_next = C6e_next, C6e
    C12e, C12e_next = C12e_next, C12e
    TETR, TETR_next = TETR_next, TETR
    LACI, LACI_next = LACI_next, LACI
    LUXI, LUXI_next = LUXI_next, LUXI
    LASI, LASI_next = LASI_next, LASI
    RFP, RFP_next = RFP_next, RFP
    GFP, GFP_next = GFP_next, GFP

    # Return everything
    return (
        LUXI,
        LASI,
        C6i,
        C12i,
        C6e,
        C12e,
        TETR,
        LACI,
        RFP,
        GFP,
        LUXI_next,
        LASI_next,
        C6i_next,
        C12i_next,
        C6e_next,
        C12e_next,
        TETR_next,
        LACI_next,
        RFP_next,
        GFP_next,
        lap_C6,
        lap_C12,
        pLT,
        pLL,
    )


def ferromagenetic(
    final_time,
    samples,
    working_directory,
    Dext_c6,
    Dext_c12,
    Dm_c6,
    Dm_c12,
    C6_conc,
    C12_conc,
):
    # Initialize the system variables
    # initialize(final_time, samples, C6_conc, C12_conc)
    (
        LUXI,
        LASI,
        C6i,
        C12i,
        C6e,
        C12e,
        TETR,
        LACI,
        RFP,
        GFP,
        LUXI_next,
        LASI_next,
        C6i_next,
        C12i_next,
        C6e_next,
        C12e_next,
        TETR_next,
        LACI_next,
        RFP_next,
        GFP_next,
        lap_C6,
        lap_C12,
        pLT,
        pLL,
        sample_times,
        n_steps,
    ) = initialize(final_time, samples, C6_conc, C12_conc)

    Dm_HSL = Dm_c6
    De_HSL = Dext_c6

    # Change to the working directory
    os.chdir(working_directory)

    # Create a new folder inside the current folder for the simulation, using the date
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    os.mkdir("simulation_" + date + "_DM_" + str(Dm_HSL) + "_DE_" + str(De_HSL))
    os.chdir("simulation_" + date + "_DM_" + str(Dm_HSL) + "_DE_" + str(De_HSL))

    # Save a txt file containing the phi parameters used, the grid specs and the initial C6_e and C12_e values
    # The current date and time is saved in the filename
    with open("simulation_parameters_" + date + ".txt", "w") as f:
        # Write the promoter parameters used
        f.write("K_1R = " + str(K_1R) + "\n")
        f.write("K_1S = " + str(K_1S) + "\n")
        # Write K_GR AND K_GS AND K_TET AND K_LAC
        f.write("K_GR = " + str(K_GR) + "\n")
        f.write("K_GS = " + str(K_GS) + "\n")
        f.write("K_TET = " + str(K_TET) + "\n")
        f.write("K_LAC = " + str(K_LAC) + "\n")
        # Write the Hill coefficients used
        f.write("n_R = " + str(n_R) + "\n")
        f.write("n_S = " + str(n_S) + "\n")

        # Write the grid specs
        f.write("Domain size (micrometers): %s\n" % domain_size)
        f.write("Grid points (number of points): %s\n" % N)
        f.write("Grid spacing (micrometers): %s\n" % Dh)

        # Write the initial C6_e and C12_e values along with the diffusion coefficients used
        f.write("Dm_c6 = " + str(Dm_HSL) + "\n")
        f.write("Dm_c6 = " + str(Dm_HSL) + "\n")
        f.write("Dext_c6 = " + str(De_HSL) + "\n")
        f.write("Dext_c12 = " + str(De_HSL) + "\n")
        f.write("C6e mean initial value (uM): %s\n" % np.mean(C6e))
        f.write("C12e mean initial value (uM): %s\n" % np.mean(C12e))

    # Close the file
    f.close()

    # Create a figure object
    fig = plt.figure(figsize=(8, 5), dpi=100)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    # and a counter
    counter = 0

    convergence = 1

    # Loop over each step (use tqdm for progress bar)
    while convergence > 1e-6:
        # Update the system
        (
            LUXI,
            LASI,
            C6i,
            C12i,
            C6e,
            C12e,
            TETR,
            LACI,
            RFP,
            GFP,
            LUXI_next,
            LASI_next,
            C6i_next,
            C12i_next,
            C6e_next,
            C12e_next,
            TETR_next,
            LACI_next,
            RFP_next,
            GFP_next,
            lap_C6,
            lap_C12,
            pLT,
            pLL,
        ) = update_vars(
            LUXI,
            LASI,
            C6i,
            C12i,
            C6e,
            C12e,
            TETR,
            LACI,
            RFP,
            GFP,
            LUXI_next,
            LASI_next,
            C6i_next,
            C12i_next,
            C6e_next,
            C12e_next,
            TETR_next,
            LACI_next,
            RFP_next,
            GFP_next,
            lap_C6,
            lap_C12,
            pLT,
            pLL,
            Dm_HSL,
            De_HSL,
        )

        # If the counter is a multiple of 1000
        if counter % 1000 == 0:  # and counter != 0:
            convergence = np.abs(np.mean(GFP_next) - np.mean(GFP))
            # print("Convergence: %s" % convergence)
            # Save the current GFP and RFP, C6_i and C6_e arrays in one numpy file
            np.savez(
                "gfp_rfp_%d.npz" % int(counter / 1000), gfp=GFP, rfp=RFP
            )  # C6_i=C6i, C12_i=C12i)
            # Print the current state of C6e and C12e
            # print("C6e: %s" % np.mean(C6e))
            # print("C12e: %s" % np.mean(C12e))

            gfp_img = ax1.imshow(GFP, cmap="inferno")
            # Show the colorbar for the current axis
            cb1 = plt.colorbar(gfp_img, ax=ax1)

            rfp_img = ax2.imshow(RFP, cmap="inferno")
            # Show the colorbar for the current axis
            cb2 = plt.colorbar(rfp_img, ax=ax2)

            # Save the figure
            fig.savefig("gfp_rfp_%d.png" % int(counter / 1000))

            # Clear both axes and the colorbars for the next iteration
            cb1.remove()
            cb2.remove()
            ax1.clear()
            ax2.clear()

        counter += 1

    # When the difference between the RFP at the current step and the previous step is less than 1e-3
    # we stop the simulation
    # while np.abs(RFP_next - RFP) > 1e-3:
    #     # Update the system

    # Print that the run has finished and change back to the working directory
    print("Solution has converged, run finished")
    os.chdir(working_directory)

    return None


# Main function
if __name__ == "__main__":
    # Create a list containing the C6_concs for the ferromagnetic simulations

    C6_concs = np.array(
        [
            0.01148090,
            0.00237850,
            0.00124500,
            0.00056350,
            0.00033850,
            0.00026450,
            0.00022800,
            0.00015900,
            0.00014450,
            0.00013450,
        ]
    )

    C12_concs = np.zeros_like(C6_concs)

    # Create a list containing the different external diffusion coefficients for C6 and C12
    # C6_Dm_coeffs = [0.001, 0.0025, 0.005, 0.0075, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
    # C12_Dm_coeffs = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
    C6_Dm_coeffs = C12_Dm_coeffs = [
        0.01,
        0.05,
        0.1,
        0.25,
        0.5,
        0.75,
        1.0,
        3.0,
        5.0,
        10.0,
    ]
    # For higher Dexts: [7.0, 8.0, 9.0, 10.0, 15.0, 20.0, 25.0, 30.0]
    # For lower Dexts: [0.01, 0.1, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    # De_vals = [0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
    C6_Dext_coeffs = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 25.0, 50.0, 75.0]
    C12_Dext_coeffs = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 25.0, 50.0, 75.0]

    # Define the working directory
    working_directory = os.getcwd()

    # Define the final time and samples to take
    # Final time is 24 hours (Since time is in minutes, we multiply by 60)
    final_time = 12 * 60
    # Samples are every 0.1 minute
    samples = final_time * 10

    # Set the number of cores to use
    # nb_cores = mp.cpu_count()
    nb_cores = len(C6_concs)

    # Ask the user the number of replicas to run
    n_replicas = int(input("How many replicas do you want to run? "))

    # Loop over the replicas
    for replica in range(n_replicas):
        for Dext_c6, Dext_c12 in zip(C6_Dext_coeffs, C12_Dext_coeffs):
            # Print that we are currently analyzing the current membrane diffusion coefficients
            print(
                "Currently simulating C6_De_coeff = %s and C12_De_coeff = %s"
                % (Dext_c6, Dext_c12)
            )

            # Run the simulation with the C6_conc, C12_conc and all the external diffusion coefficients
            with mp.Pool(processes=nb_cores) as pool:
                pool.starmap(
                    ferromagenetic,
                    [
                        (
                            final_time,
                            samples,
                            working_directory,
                            Dext_c6,
                            Dext_c12,
                            Dm_c6,
                            Dm_c12,
                            C6_conc,
                            C12_conc,
                        )
                        for Dm_c6, Dm_c12, C6_conc, C12_conc in zip(
                            C6_Dm_coeffs, C12_Dm_coeffs, C6_concs, C12_concs
                        )
                    ],
                )

            # Print that the current C6_Dm_coeff and C12_Dm_coeff have finished
            print(
                "Finished simulation with C6_De_coeff = %s and C12_De_coeff = %s"
                % (Dext_c6, Dext_c12)
            )
