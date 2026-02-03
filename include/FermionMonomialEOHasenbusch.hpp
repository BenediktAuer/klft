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
#include "DiracOperator.hpp"
#include "GLOBAL.hpp"
#include "Monomial.hpp"
#include "Solver.hpp"
#include "SpinorFieldLinAlg.hpp"

#define SQRT2INV \
  0.707106781186547524400844362104849039284835937688474036588339868995366239231053519425193767163820786367506  // Oeis A010503
namespace klft {
template <class RNGType,
          typename DAdjFieldType,
          template <typename> class _Solver,
          class DiracOpT,
          class DiracOPNonShift>
class FermionMonomialEOHasenbusch
    : public Monomial<typename DiracOpT::DGaugeFieldType, DAdjFieldType> {
  using DSpinorFieldType = typename DiracOpT::DSpinorFieldType;
  using DGaugeFieldType = typename DiracOpT::DGaugeFieldType;
  static_assert(isDeviceFermionFieldType<DSpinorFieldType>::value);
  static_assert(isDeviceGaugeFieldType<DGaugeFieldType>::value);
  static_assert(isDeviceAdjFieldType<DAdjFieldType>::value);
  constexpr static size_t rank =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank;
  constexpr static size_t Nc = DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Nc;
  constexpr static size_t RepDim =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::RepDim;
  static_assert(rank == DeviceAdjFieldTypeTraits<DAdjFieldType>::Rank &&
                    rank ==
                        DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank &&
                    Nc == DeviceAdjFieldTypeTraits<DAdjFieldType>::Nc &&
                    Nc == DeviceFermionFieldTypeTraits<DSpinorFieldType>::Nc,
                "Rank and Nc must match between gauge, adjoint, and fermion "
                "field types.");
  static_assert(DeviceFermionFieldTypeTraits<DSpinorFieldType>::Layout ==
                    SpinorFieldLayout::Checkerboard,
                "When using Even/odd preconditioning "
                "the spinor field layout must be "
                "Checkerboard");
  using FermionField = typename DSpinorFieldType::type;
  using DiracOperator = DiracOpT;
  using Solver = _Solver<DiracOpT>;

 public:
  FermionField& phi;
  using DiracOpShifted = DiracOpT;
  using DiracOpNonShift = DiracOPNonShift;
  const diracParams params;

  FermionField x;

  FermionField x0;
  // auxillary fields
  FermionField xk;
  FermionField rk;
  FermionField apk;
  FermionField temp_D;
  typename DeviceScalarFieldType<rank>::type norm_per_site;
  typename DeviceFieldType<rank>::type dot_product_per_site;
  FermionField pk;

  const real_t tol;
  RNGType rng;
  FermionMonomialEOHasenbusch(FermionField& _phi,
                              const diracParams& params,
                              const real_t& tol_,
                              RNGType& RNG_,
                              unsigned int _time_scale)
      : Monomial<DGaugeFieldType, DAdjFieldType>(_time_scale),
        phi(_phi),
        params(params),

        rng(RNG_),
        tol(tol_) {
    this->x = FermionField(this->phi.dimensions, complex_t(0.0, 0.0));

    this->x0 = FermionField(this->phi.dimensions, complex_t(0.0, 0.0));
    // Auxillary
    this->xk = FermionField(phi.dimensions, complex_t(0.0, 0.0));
    this->rk = FermionField(phi.dimensions, complex_t(0.0, 0.0));
    this->apk = FermionField(phi.dimensions, complex_t(0.0, 0.0));
    this->temp_D = FermionField(phi.dimensions, complex_t(0.0, 0.0));
    this->pk = FermionField(phi.dimensions, complex_t(0.0, 0.0));
    this->norm_per_site =
        typename DeviceScalarFieldType<rank>::type(phi.dimensions, 0.0);
    this->dot_product_per_site = typename DeviceFieldType<rank>::type(
        phi.dimensions, complex_t(0.0, 0.0));
    Monomial<DGaugeFieldType, DAdjFieldType>::monomial_type =
        KLFT_MONOMIAL_FERMION;
    printf("Created Fermion HB Monomial EO\n");
  }

  void heatbath(HamiltonianField<DGaugeFieldType, DAdjFieldType> h) override {
    Kokkos::Profiling::pushRegion("FermionHeatbathEO");
    auto dims = phi.dimensions;
    DiracOpNonShift D_n(h.gauge_field, this->params);
    DiracOpShifted D_s(h.gauge_field, this->params);
    FermionField R(dims, rng, 0, SQRT2INV);
    D_n.template apply<Tags::TagG5Se>(R, this->temp_D, this->phi);
    Solver solver(this->phi, this->x, D_s, this->xk, this->rk, this->apk,
                  this->temp_D, this->pk, this->norm_per_site,
                  this->dot_product_per_site);
    solver.template solve<Tags::TagDdaggerD>(
        this->x0,
        this->tol);  // chi = S_e^-1 S_e^-1 R

    D_s.template apply<Tags::TagG5Se>(solver.x, this->x0, this->phi);

    Monomial<DGaugeFieldType, DAdjFieldType>::H_old =
        spinor_norm_sq<DSpinorFieldType>(R);
    Kokkos::Profiling::popRegion();
    // print_spinor__int(this->phi(0, 0, 0, 0), "HB Phi at hatbath");s
  }

  void accept(HamiltonianField<DGaugeFieldType, DAdjFieldType> h) override {
    Kokkos::Profiling::pushRegion("FermionAcceptEO");
    auto dims = phi.dimensions;

    FermionField y(dims, complex_t(0.0, 0.0));

    DiracOpNonShift D_n(h.gauge_field, this->params);
    DiracOpShifted D_s(h.gauge_field, this->params);
    D_s.template apply<Tags::TagG5Se>(this->phi, this->temp_D, y);
    _Solver<DiracOpNonShift> solver(
        y, this->x, D_n, this->xk, this->rk, this->apk, this->temp_D, this->pk,
        this->norm_per_site, this->dot_product_per_site);
    if (KLFT_VERBOSITY > 4) {
      printf("Solving inside Fermion Monomial accept:");
    }

    solver.template solve<Tags::TagDdaggerD>(this->x0, this->tol);

    Monomial<DGaugeFieldType, DAdjFieldType>::H_new =
        spinor_dot_product<DSpinorFieldType>(y, solver.x).real();
    Kokkos::Profiling::popRegion();
  }
  void print() override {
    printf("HB Fermion Monomial: %.20f\n", this->get_delta_H());
  }
};
}  // namespace klft
