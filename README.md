# klft

A library for lattice field theory simulation accelerated using Kokkos

# Installation

use git to clone the repository

```bash
git clone https://github.com/aniketsen/klft.git /path/to/klft
cd /path/to/klft
```

setup `kokkos`, `KTune` and `yaml-cpp` 

```bash
git submodule update --init --recursive
```

build the library

```bash
mkdir /path/to/build
cd /path/to/build

cmake [Kokkos options] /path/to/klft

make -j<number of threads>
```

### Kokkos options

The most important Kokkos options are:

`-DKokkos_ENABLE_CUDA=ON` to enable CUDA support

`-DKokkos_ENABLE_OPENMP=ON` to enable OpenMP support

`-DKokkos_ARCH_<arch>=ON` to enable a specific architecture (e.g. `-DKokkos_ARCH_AMPERE80=ON` for NVIDIA A100 gpus)

see the [Kokkos documentation](https://kokkos.org/kokkos-core-wiki/get-started/configuration-guide.html#cmake-keywords) for more options



# Usage

## Metropolis

```bash
binaries/metropolis
  -f <file_name> --filename <file_name>
    Name of the input file.
    Default: input.yaml
  -o <file_name> --output <file_name>\n
    Path to the output folder.\n
    Hint: if the folder does not exist, it will be created.\n
    Default: .\n   
  -h, --help
     Prints this message.
     Hint: use --kokkos-help to see command line options provided by Kokkos.
```

### Example input.yaml for Metropolis

```yaml
# input.yaml
MetropolisParams:    # parameters for the Metropolis algorithm
  Ndims: 4    # number of dimensions [2,3,4]
  Nd: 4       # number of link dimensions (this must be strictly same as Ndims)
  Nc: 1       # number of colors (defines SU(Nc)) [1,2,3]
  L0: 8       # lattice extent in 0 direction
  L1: 8       # lattice extent in 1 direction
  L2: 8       # lattice extent in 2 direction
  L3: 8       # lattice extent in 3 direction
  nHits: 10       # number of hits per sweep
  nSweep: 1000      # number of sweeps
  seed: 32091     # random seed
  beta: 2.0       # inverse coupling constant
  delta: 0.1      # step size for the Metropolis algorithm

GaugeObservableParams:
  measurement_interval: 10               # interval for measurements
  measure_plaquette: true                # measure the plaquette
  measure_wilson_loop_temporal: true    # measure the temporal Wilson loop
  measure_wilson_loop_mu_nu: true       # measure the spatial Wilson loop
  W_temp_L_T_pairs:      # pairs of (L, T) values for the temporal Wilson loop
    - [2, 2]
    - [3, 4]             # keep a non-decreasing order (as much as possible)
    - [4, 3]             # for maximum efficiency
    - [4, 4]
  W_mu_nu_pairs:      # pairs of (mu, nu) values for the planar Wilson loop
    - [0, 1]
    - [1, 2]
    - [3, 2]
  W_Lmu_Lnu_pairs:      # pairs of (Lmu, Lnu) values for the lengths of the 
    - [2, 2]            # planar Wilson loop in the mu and nu directions
    - [3, 3]            # again, keep a non-decreasing order (as much as possible)
    - [4, 3]
  plaquette_filename: "plaquette.out"  # filename to output the plaquette
  W_temp_filename: "W_temp.out"        # filename to output the temporal Wilson loop
  W_mu_nu_filename: "W_mu_nu.out"      # filename to output the planar Wilson loop
  write_to_file: true                  # write the measurements to file
```
### Example input.yaml for HMC

```yaml 
HMCParams:
  Ndims: 4    # number of dimensions [2,3,4]
  Nd: 4       # number of link dimensions (this must be strictly same as Ndims)
  Nc: 2       # number of colors (defines SU(Nc)) [1,2] (SU(3) is still WiP)
  L0: 8       # lattice extent in 0 direction
  L1: 8       # lattice extent in 1 direction
  L2: 8       # lattice extent in 2 direction
  L3: 8       # lattice extent in 3 direction
  seed: 1234  # random seed
  coldStart: false # start with GaugeFIeld set to 1
  rngDelta: 1 # step size for the Metropolis algorithm


# .
Integrator: # parameters to configure the Integrator, Level 0 is the innermost level of the Integrator, i.e that is executed most frequently
  tau: 1    # time for md trajectory
  nSteps: 1000 # Number of md trajectory 
  Monomials: # Monomial types 
    - Type: "Leapfrog"  # Integrator to be used for this Level [Leapfrog,OMF2]
      level: 0          # Level for this Monomial used for matching the specific Monomial (see below)
      steps: 100        # Integration steps for specific Monomial
    - Type: "Leapfrog"
      level: 1
      steps: 20


Gauge Monomial: # Monomial for Pure Gauge [Must be used] 
  level: 0      # Level to identifiy it with the Integrator 
  beta: 2.12    # inverse coupling constant

Fermion Monomial: # Monomial for Fermions (2 mass degenerate Flavours) [For now only in 4D]
  level: 1  # Level to identifiy it with the Integrator  
  fermion: "HWilson" # Typ of Fermion(operator) [Wilson]
  solver: "CG" # "Solver for Matrix Inversion" [CG,CGMultiP]
  RepDim: 4 # Spinor Representation
  kappa: 0.15  # hopping parameter
  tol: 1e-10 # Solver tolerance

GaugeObservableParams:
  measurement_interval: 5
  measure_plaquette: true
  measure_wilson_loop_temporal: false
  measure_wilson_loop_mu_nu: false
  flush: 5 # Number of md trajectories before saving to disk

  W_temp_L_T_pairs:
    - [2, 3]
    - [3, 4]

  W_mu_nu_pairs:
    - [0, 1]
    - [1, 2]

  W_Lmu_Lnu_pairs:
    - [2, 2]
    - [3, 3]

  plaquette_filename: "plaquette_output20.txt"
  W_temp_filename: "wilson_temp_output20.txt"
  W_mu_nu_filename: "wilson_mu_nu_output20.txt"

  write_to_file: true

SimulationLoggingParams: 
  log_interval: 1  # interval for logging
  log_delta_H: true # Log S_old - S_new
  log_acceptance: true # Log acceptance Rate
  flush: 5 # Number of md trajectories before saving to disk
  
  log_filename: "simulation_log.txt"
  write_to_file: true



```
### Tuning
Tuning is done via [KTune](link_to_ktune). 
To deactivate Tuning, set the enviroment variable  `KTUNE_DISABLE_TUNING` to `1`.
Seek the documentation for more in-depth information.


# Environment variables

### KLFT_VERBOSITY
Set the verbosity level of the library.
- 0: silent
- 1: summarize
- 2: verbose
- &gt;=3: debug
- Default: 0
