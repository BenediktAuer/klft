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
          template <class DiracOpT> class _Solver,
          class DiracOpT>
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

  const diracParams params_light;
  const diracParams params_heavy;
    const real_t a ;
  const real_t b ;
  const real_t tol;
  RNGType rng;
  FermionMonomialEOHasenbusch(FermionField& _phi,const diracParams& params_light,
                              const diracParams& params_heavy,
                              const real_t& tol_,
                              RNGType& RNG_,
                              unsigned int _time_scale)
      : Monomial<DGaugeFieldType, DAdjFieldType>(_time_scale),
        phi(_phi),
        params_light(params_light),
        params_heavy(params_heavy),
        rng(RNG_),
        tol(tol_), a(params_heavy.kappa *params_heavy.kappa /(params_light.kappa*params_light.kappa)), b(1-a) {
    Monomial<DGaugeFieldType, DAdjFieldType>::monomial_type =
        KLFT_MONOMIAL_FERMION;
    printf("Created Fermion Monomial EO\n");
  }

  void heatbath(HamiltonianField<DGaugeFieldType, DAdjFieldType> h) override {
    Kokkos::Profiling::pushRegion("FermionHeatbathEO");
    auto dims = phi.dimensions;
    FermionField x(dims, complex_t(0.0, 0.0));
    FermionField x0(dims, complex_t(0.0, 0.0));

    FermionField R(dims, rng, 0, SQRT2INV);
    DiracOperator dirac_op_heavy(h.gauge_field,
                                 params_heavy);
    DiracOperator dirac_op_light(h.gauge_field,
                                 params_light);  // params.kappa = light kappa
    Solver solver(R, x, dirac_op_heavy);
    solver.template solve<Tags::TagDdaggerD>(
        x0,
        this->tol);  // chi = S_e^-1 S_e^-1 R

    dirac_op_heavy.template apply<Tags::TagG5Se>(solver.x, x0);
    dirac_op_light.template apply<Tags::TagG5Se>(x0, this->phi);

    Monomial<DGaugeFieldType, DAdjFieldType>::H_old =
        spinor_norm_sq<rank, Nc, RepDim>(R);
    Kokkos::Profiling::popRegion();
  }

  void accept(HamiltonianField<DGaugeFieldType, DAdjFieldType> h) override {
    Kokkos::Profiling::pushRegion("FermionAcceptEO");
    auto dims = phi.dimensions;

    FermionField x(dims, complex_t(0.0, 0.0));
    FermionField x0(dims, complex_t(0.0, 0.0));
    FermionField y(dims, complex_t(0.0, 0.0));
    DiracOperator dirac_op(h.gauge_field, this->params_light);
    Solver solver(this->phi, x, dirac_op);
    if (KLFT_VERBOSITY > 4) {
      printf("Solving inside Fermion Monomial accept:");
    }

    solver.template solve<Tags::TagDdaggerD>(
        x0,
        this->tol);  // chi = S_e^-1 S_e^-1 phi // with light one
     FermionField chi = solver.x;
    dirac_op.template apply<Tags::TagG5Se>(chi, y);  // y = S_e^-1 phi no gamma5 here
    axG5<DSpinorFieldType>(this->b, y, y);               // b* M^dagger^-1 phi // minus from commuting gamma 5
    axpy<DSpinorFieldType>(this->a, this->phi, y,
         chi);  // chi = a*phi + b* M^dagger^-1 phi
    Monomial<DGaugeFieldType, DAdjFieldType>::H_new =
        spinor_norm_sq<rank, Nc, RepDim>( chi)
            ;  // S_F = chi^dagger chi = phi^dagger S_e^-1 S_e^-1 phi
    Kokkos::Profiling::popRegion();
  }
  void print() override {
    printf("Fermion Monomial: %.20f\n", this->get_delta_H());
  }
};
}  // namespace klft
