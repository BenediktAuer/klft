//******************************************************************************/
//
// This file is part of the Kokkos Lattice Field Theory (KLFT) library.
//
// KLFT is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// KLFT is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with KLFT.  If not, see <http://www.gnu.org/licenses/>.
//
//******************************************************************************/
#pragma once
#include "APE_smearing.hpp"
#include "FieldTypeHelper.hpp"
#include "GLOBAL.hpp"
#include "GammaMatrix.hpp"
#include "HMC_Params.hpp"
#include "Jacobi_smearing.hpp"

namespace klft {
struct FermionObservableParams {
  size_t measurement_interval;
  bool measure_pion_correlator;
  std::vector<std::vector<real_t>> pion_correlator;
  std::string pion_correlator_filename;
  std::vector<size_t> measurement_steps;
  real_t tol;
  real_t kappa;
  index_t n_sources;
  size_t RepDim;
  bool write_to_file;
  bool flushed;
  bool preconditioning;

  //
  size_t flush;  // interval to flush measurements to file, 0 to flush at the
  // end of the simulation
  APESmearingParams ape_smearing_params;
  bool do_ape_smearing;
  bool do_Jacobi_smearing;
  JacobiSmearingParams jacobi_smearing_params;
  void print() const {
    printf("FermionObservableParams:\n");
    printf("  measurement_interval: %zu\n", measurement_interval);
    printf("  measure_pion_correlator: %s\n",
           measure_pion_correlator ? "true" : "false");
    printf("  pion_correlator_filename: %s\n",
           pion_correlator_filename.c_str());
    printf("  tol: %e\n", tol);
    printf("  kappa: %f\n", kappa);
    printf("  RepDim: %zu\n", RepDim);
    printf("  write_to_file: %s\n", write_to_file ? "true" : "false");
    printf("  flush: %zu\n", flush);
    printf("  n_sources: %d\n", n_sources);
  }
};

// Parameters specific to the Dirac operator

struct diracParams {
  real_t kappa;
  real_t massShift;
  diracParams() = default;
  // kappa_tilde is the reduced kappa i.e kappa_tilde<kappa as in
  // arXiv:hep-lat/0107019v1
  diracParams(const real_t& _kappa) : kappa(_kappa), massShift(0.0) {};
  diracParams(const real_t& _kappa, const real_t& _massShift)
      : kappa(_kappa), massShift(_massShift) {};
};
auto getDiracParams(const FermionMonomial_Params& fparams) {
  diracParams dParams(fparams.kappa);
  return dParams;
}
auto getDiracParams(const Hasenbusch_Params fparams) {
  diracParams dParams(fparams.kappa, fparams.massShift);
  return dParams;
}

}  // namespace klft
