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
#include "FieldTypeHelper.hpp"
#include "GDiracOperator.hpp"
#include "GLOBAL.hpp"
#include "SpinorFieldLinAlg.hpp"
namespace klft {

template <typename Derived, size_t rank, size_t Nc, size_t RepDim>
class Solver
    : public std::enable_shared_from_this<Solver<Derived, rank, Nc, RepDim>> {
 public:
  using SpinorFieldType =
      typename DeviceSpinorFieldType<rank, Nc, RepDim>::type;
  using GaugeFieldType = typename DeviceGaugeFieldType<rank, Nc>::type;
  const SpinorFieldType b;
  SpinorFieldType x;  // Solution to DiracOP*x=b
  DiracOperator<Derived, rank, Nc, RepDim> dirac_op;
  Solver(const SpinorFieldType& b,
         SpinorFieldType& x,
         const DiracOperator<Derived, rank, Nc, RepDim>& dirac_op)
      : b(b), x(x), dirac_op(dirac_op) {}

  virtual void solve(const SpinorFieldType& x0, const real_t& tol) = 0;
};
// Deduction guide for Solver
template <typename Operator, typename SpinorType>
Solver(const SpinorType&, SpinorType&, const Operator&)
    -> Solver<typename Operator::Derived,
              SpinorType::rank,
              SpinorType::Nc,
              SpinorType::RepDim>;

template <typename Derived, size_t rank, size_t Nc, size_t RepDim>
class CGSolver : public Solver<Derived, rank, Nc, RepDim> {
 public:
  using SpinorFieldType =
      typename DeviceSpinorFieldType<rank, Nc, RepDim>::type;
  using GaugeFieldType = typename DeviceGaugeFieldType<rank, Nc>::type;
  CGSolver() = delete;
  CGSolver(const SpinorFieldType& b,
           SpinorFieldType& x,
           DiracOperator<Derived, rank, Nc, RepDim>& dirac_op)
      : Solver<Derived, rank, Nc, RepDim>(b, x, dirac_op) {}

  void solve(const SpinorFieldType& x0, const real_t& tol) override {
    SpinorFieldType xk(this->dirac_op.params.dimensions, complex_t(0.0, 0.0));
    Kokkos::deep_copy(xk.field, x0.field);  // x_0

    SpinorFieldType rk = spinor_sub_mul<rank, Nc, RepDim>(
        this->b,
        this->dirac_op.applyDdagger(this->dirac_op.applyD(
            xk)),  // check if this is the right vector to apply dirac_op
        1.0);      // r_0
    SpinorFieldType pk(this->dirac_op.params.dimensions, complex_t(0.0, 0.0));
    Kokkos::deep_copy(pk.field, rk.field);  // p_0                        // d_0
    real_t rk_norm = spinor_norm<rank, Nc, RepDim>(rk);  //\delta_0
    int num_iter = 0;
    while (rk_norm > tol) {
      const SpinorFieldType apk =
          this->dirac_op.applyDdagger(this->dirac_op.applyD(pk));  // z = Ad_k
      const complex_t rkrk = spinor_dot_product<rank, Nc, RepDim>(rk, rk);
      const complex_t alpha = (rkrk / spinor_dot_product<rank, Nc, RepDim>(
                                          pk, apk));  // Always real
      xk = spinor_add_mul<rank, Nc, RepDim>(xk, pk, alpha);
      rk = spinor_sub_mul<rank, Nc, RepDim>(rk, apk, alpha);
      const complex_t beta =
          (spinor_dot_product<rank, Nc, RepDim>(rk, rk) / rkrk);
      pk = spinor_add_mul<rank, Nc, RepDim>(rk, pk, beta);
      // Check if swapping is needed of pk and rk, should be correct

      rk_norm = spinor_norm<rank, Nc, RepDim>(rk);
      num_iter++;
      if (KLFT_VERBOSITY > 1) {
        printf("CG Iteration %d: rk_norm = %.15f\n", num_iter, rk_norm);
        if (KLFT_VERBOSITY > 3) {
          printf("Norm of (b - A*x) %.15f\n",
                 spinor_norm<rank, Nc, RepDim>(spinor_sub_mul<rank, Nc, RepDim>(
                     this->b,
                     this->dirac_op.applyDdagger(this->dirac_op.applyD(xk)),
                     1.0)));
        }
      }
    }
    const real_t ex_res =
        spinor_norm<rank, Nc, RepDim>(spinor_sub_mul<rank, Nc, RepDim>(
            this->b, this->dirac_op.applyDdagger(this->dirac_op.applyD(xk)),
            1.0));

    if (Kokkos::abs(rk_norm - ex_res) < tol) {
      printf("Roundoff Error, relaunching CG solver with new initial guess\n");
      this->solve(xk, tol);
    } else {
      if (KLFT_VERBOSITY > 0) {
        printf("CG solver converged in %d iterations\n", num_iter);
      }
      this->x = xk;
    }
  }
};
}  // namespace klft
