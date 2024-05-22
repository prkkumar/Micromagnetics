# MagneX
MagneX is a massively parallel, 3D micromagnetics solver for modeling magnetic materials.
MagneX solves the Landau-Lifshitz-Gilbert (LLG) equations, including exchange, anisotropy, demagnetization, and Dzyaloshinskii-Moriya interaction (DMI) coupling.
The algorithm is implemented using Exascale Computing Project software framework, AMReX, which provides effective scalability on manycore and GPU-based supercomputing architectures.

# Installation
## Download AMReX Repository
``` git clone git@github.com:AMReX-Codes/amrex.git ```
## Download MagneX Repository
``` git@github.com:AMReX-Microelectronics/MagneX.git ```
## Build
Make sure that the AMReX and MagneX are cloned in the same location in their filesystem. Navogate to the Exec folder of MagneX and execute
```make -j 4```

# Running MagneX
Example input scripts are located in `Exec/standard_problem_inputs/` directory. 
## Simple Testcase
You can run the following to simulate muMAG Standard Problem 4 dynamics:
## For pure MPI build (but with a single MPI rank)
```./main3d.gnu.MPI.ex standard_problem_inputs/inputs_std4```
# Visualization and Data Analysis
Refer to the following link for several visualization tools that can be used for AMReX plotfiles. 

[Visualization](https://amrex-codes.github.io/amrex/docs_html/Visualization_Chapter.html)

### Data Analysis in Python using yt 
You can extract the data in numpy array format using yt (you can refer to this for installation and usage of [yt](https://yt-project.org/). After you have installed yt, you can do something as follows, for example, to get variable 'Pz' (z-component of polarization)
```
import yt
ds = yt.load('./plt00001000/') # for data at time step 1000
ad0 = ds.covering_grid(level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions)
P_array = ad0['Pz'].to_ndarray()
```
# Publications
1. Z. Yao, P. Kumar, J. C. LePelch, and A. Nonaka, MagneX: An Exascale-Enabled Micromagnetics Solver for Spintronic Systems, in preparation.
