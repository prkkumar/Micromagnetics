#include "MagneX.H"

#include <AMReX_ParmParse.H>

// total number of grid cells in the simulation domain in each direction
AMREX_GPU_MANAGED amrex::GpuArray<int, 3> MagneX::n_cell;

// maximum grid size in each direction (for spatial domain decomposition)
int MagneX::max_grid_size_x;
int MagneX::max_grid_size_y;
int MagneX::max_grid_size_z;

// physical lo/hi coordiates
AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, 3> MagneX::prob_lo;
AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, 3> MagneX::prob_hi;

// total steps in simulation
int MagneX::nsteps;

// maximum simulation time (will override nsteps if reached)
amrex::Real MagneX::stop_time;

// time integration options
// 1 = first order forward Euler
// 2 = iterative predictor-corrector
// 3 = iterative direct solver
// 4 = AMReX and SUNDIALS integrators
int MagneX::TimeIntegratorOption;

// for TimeIntegrationOption 2 and 3 options only
// tolerance threshold (L_inf change between iterations) required for completing time step
// special cases:
// for TimeIntegrationOption=2, iterative_tolerance=0. means force 2 iterations
// for TimeIntegrationOption=3, iterative_tolerance=0. means force 1 iteration
amrex::Real MagneX::iterative_tolerance;

// time step
amrex::Real MagneX::dt;

// how often to write a plotfile
int MagneX::plot_int;

// whether or not to include specific variables in plotfile
int MagneX::plot_Ms;
int MagneX::plot_H_bias;
int MagneX::plot_exchange;
int MagneX::plot_DMI;
int MagneX::plot_anisotropy;
int MagneX::plot_demag;

// how often to write a checkpoint
int MagneX::chk_int;

// step to restart from
int MagneX::restart;

// what type of extra diagnostics?
// 4 = standard problem 4
// 3 = standard problem 3
// 2 = standard problem 2
int MagneX::diag_type;

// flag for enabling Hbias sweeps for hysteresis calculations
int MagneX::Hbias_sweep;

// tolerance threshold for equilibrium in M in Hbias sweeping
amrex::Real MagneX::equilibrium_tolerance;

// increment size for Hbias  
AMREX_GPU_MANAGED amrex::Real MagneX::increment_size;

// number of increments on Hbias before reversing sign of increment
int MagneX::nsteps_hysteresis;

// permeability
AMREX_GPU_MANAGED amrex::Real MagneX::mu0;

// whether to call the parser each time step, or only at initialization
int MagneX::timedependent_Hbias;
int MagneX::timedependent_alpha;

// 0 = disable precession; 1 = enable precession
AMREX_GPU_MANAGED int MagneX::precession;

// 0 = no demagnetization; 1 = use demanetization
AMREX_GPU_MANAGED int MagneX::demag_coupling;

// 0 = unsaturated; 1 = saturated
AMREX_GPU_MANAGED int MagneX::M_normalization;

// turn on exchange
AMREX_GPU_MANAGED int MagneX::exchange_coupling;

// turn on DMI
AMREX_GPU_MANAGED int MagneX::DMI_coupling;

// turn on anisotropy
AMREX_GPU_MANAGED int MagneX::anisotropy_coupling;
AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, 3> MagneX::anisotropy_axis;

// Choose an FFT solver; 0 = FFTW, 1 = heFFTe
AMREX_GPU_MANAGED int MagneX::FFT_solver;

void InitializeMagneXNamespace() {

    BL_PROFILE_VAR("InitializeMagneXNamespace()",InitializeMagneXNameSpace);

    // ParmParse is way of reading inputs from the inputs file
    // pp.get means we require the inputs file to have it
    // pp.query means we optionally need the inputs file to have it - but we must supply a default here
    ParmParse pp;

    // temporary vectors for reading in parameters from inputs files
    // these get copied into the GpuArray versions of these parameters
    amrex::Vector<int> temp_int(AMREX_SPACEDIM);
    amrex::Vector<amrex::Real> temp(AMREX_SPACEDIM);

    pp.getarr("n_cell",temp_int);
    for (int i=0; i<AMREX_SPACEDIM; ++i) {
        n_cell[i] = temp_int[i];
    }

    pp.get("max_grid_size_x",max_grid_size_x);
    pp.get("max_grid_size_y",max_grid_size_y);
    pp.get("max_grid_size_z",max_grid_size_z);

    pp.getarr("prob_lo",temp);
    for (int i=0; i<AMREX_SPACEDIM; ++i) {
        prob_lo[i] = temp[i];
    }
    pp.getarr("prob_hi",temp);
    for (int i=0; i<AMREX_SPACEDIM; ++i) {
        prob_hi[i] = temp[i];
    }

    nsteps = 1000000000;
    pp.query("nsteps",nsteps);

    stop_time = 1.e9;
    pp.query("stop_time",stop_time);

    nsteps_hysteresis = 1000000000;
    pp.query("nsteps_hysteresis",nsteps_hysteresis);

    equilibrium_tolerance = 1.e-6;
    pp.query("equilibrium_tolerance",equilibrium_tolerance);

    increment_size = 1.e-5;
    pp.query("increment_size",increment_size);

    pp.get("TimeIntegratorOption",TimeIntegratorOption);

    iterative_tolerance = 1.e-9;
    pp.query("iterative_tolerance",iterative_tolerance);

    pp.get("dt",dt);

    plot_int = -1;
    pp.query("plot_int",plot_int);

    plot_Ms = 1;
    pp.query("plot_Ms",plot_Ms);
    plot_H_bias = 1;
    pp.query("plot_H_bias",plot_H_bias);
    plot_exchange = 1;
    pp.query("plot_exchange",plot_exchange);
    plot_DMI = 1;
    pp.query("plot_DMI",plot_DMI);
    plot_anisotropy = 1;
    pp.query("plot_anisotropy",plot_anisotropy);
    plot_demag = 1;
    pp.query("plot_demag",plot_demag);

    chk_int= -1;
    pp.query("chk_int",chk_int);

    restart = -1;
    pp.query("restart",restart);

    diag_type = -1;
    pp.query("diag_type",diag_type);

    Hbias_sweep = 0;
    pp.query("Hbias_sweep", Hbias_sweep);
	
    pp.get("mu0",mu0);

    pp.get("timedependent_Hbias",timedependent_Hbias);
    pp.get("timedependent_alpha",timedependent_alpha);

    precession = 1;
    pp.query("precession",precession);
   
    pp.get("demag_coupling",demag_coupling);

    if (demag_coupling == 1) {
        pp.get("FFT_solver",FFT_solver);
    }
        
    pp.get("M_normalization", M_normalization);
    pp.get("exchange_coupling", exchange_coupling);
    pp.get("DMI_coupling", DMI_coupling);

    pp.get("anisotropy_coupling", anisotropy_coupling);
    if (anisotropy_coupling) {
        pp.getarr("anisotropy_axis",temp);
        for (int i=0; i<AMREX_SPACEDIM; ++i) {
            anisotropy_axis[i] = temp[i];
        }
    }
}
