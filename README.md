# Master-Thesis-IIBM
Codes used for the my master's thesis in biological and medical engineering.

## ```PROMOTER_ACTIVITY_FUNCTIONS.ipynb``` (Python)

Contains code used for the simulation of the response curves for the three different activity functions tested in the thesis:
- First function corresponds to a thermodynamic ensemble model.
- Second function corresponds to a Hill-response model.
- Third function corresponds to a simplfied Hill-type function obtained from elementary reactions.
Figures from chapter 5.1 were generated using this code

## ```ODE_SYSTEM_SIMULATIONS.ipynb``` (Python)

Contains the code used for the numerical integration of the ODE system. Figures from chapter 5.2 and 5.3 were generated using this code.

## ```ODE_SYSTEM_JULIA_ANALYSIS.ipynb``` (Julia)

Contains the Julia code used for the bifurcation analysis of the ODE system and for the calculation of the steady states. Figures from chapter 5.4 were generated using this code.

## ```RDIM_MODEL_SIMULATIONS.py``` (Python)

Contains the code used for the simulation of the square grid of cells for the RDIM system. Figures from chapter 6 onward were generated using this code.

## ```RDIM_MODEL_CELLMODELLER.py``` (Python/OpenCL)

Contains the code used for the CellModeller simulations of the RDIM system. Figures from chapter 6.4 were generated using this code.

## ```RDIM_SIMULATION_PROCESSING.ipynb``` (Python)

Contains the code used for processing the output data from the ```RDIM_MODEL_SIMULATIONS```. Necessary for obtaining the final figures of each simulation, where the concentrations of the reporter proteins are converted to 8 bit values.

## ```RDIM_MAGNETIZATION_GRAPHS.ipynb``` (Python)

Contains the code for calculating the magnetization of a given RDIM lattice. Figures from chapter 8 were generated using this code.

## ```PHASE_DIAGRAM_RDIM_MODEL.ipynb``` (Python)

Contains the code for the generation of the phase diagram in chapter 9.

## ```POWERLAW_FITS.m``` (Matlab)

Contains the matlab code used for the MLE fits to the powerlaw. Data used for chapter 7 were generated using this code.

## ```POWERLAW_ANALYSIS.ipynb``` (Python)

Contains the code used for generating the powerlaw graphs from the data obtained in ```POWERLAW_FITS.m```. Figures from chapter 7 were generated using this code.
