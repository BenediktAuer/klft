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
#include "DiracOPTypeHelper.hpp"
#include "DiracOperator.hpp"
#include "FieldTypeHelper.hpp"
#include "GLOBAL.hpp"
#include "SpinorFieldLinAlg.hpp"

namespace klft {

template < template <typename> class _Solver,
          class DiracOpT>
class Solver {
  // using DSpinorFieldType =
  //     typename DiracOpFieldTypeTraits<DiracOperator>::DSpinorFieldType;
  // using DGaugeFieldType =
  //     typename DiracOpFieldTypeTraits<DiracOperator>::DGaugeFieldType;
  // template argument deduction and safety
  public:
  using DSpinorFieldType = typename DiracOpT::DSpinorFieldType;
  using DGaugeFieldType = typename DiracOpT::DGaugeFieldType;
  static_assert(isDeviceFermionFieldType<DSpinorFieldType>::value);
  static_assert(isDeviceGaugeFieldType<DGaugeFieldType>::value);
  constexpr static size_t rank =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank;
  constexpr static size_t Nc =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Nc;
  constexpr static size_t RepDim =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::RepDim;
  static_assert((rank == DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank) &&
                (Nc == DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Nc));
              

  // using DiracOp = DiracOpT<DSpinorFieldType, DGaugeFieldType, DiracOpT::HasMassshift>;
  using DiracOp = DiracOpT;
  using Derived = _Solver<DiracOp>;

  // using DiracOperator =
  //     DiracOperator<DerivedDiracOperator, DSpinorFieldType, DGaugeFieldType>;
  // using ConcreteSolver =
  //     _ConcreteSolver<DiracOperator, DSpinorFieldType, DGaugeFieldType>;

 public:
  using SpinorFieldType = typename DSpinorFieldType::type;
  using GaugeFieldType = typename DGaugeFieldType::type;
  SpinorFieldType b;
  SpinorFieldType x;  // Solution to DiracOP*x=b
  DiracOp dirac_op;
  // auxillary fields
  SpinorFieldType xk;
  SpinorFieldType rk;
  SpinorFieldType apk;
  SpinorFieldType temp_D;
  typename DeviceScalarFieldType<rank>::type norm_per_site;
  typename DeviceFieldType<rank>::type dot_product_per_site;
  SpinorFieldType pk;
  Solver(const SpinorFieldType& b, SpinorFieldType& x, const DiracOp& dirac_op)
      : b(b), x(x), dirac_op(dirac_op) {
    auto dims = this->x.dimensions;
    this->xk = SpinorFieldType(dims, complex_t(0.0, 0.0));
    this->rk = SpinorFieldType(dims, complex_t(0.0, 0.0));
    this->apk = SpinorFieldType(dims, complex_t(0.0, 0.0));
    this->temp_D = SpinorFieldType(dims, complex_t(0.0, 0.0));
    this->pk = SpinorFieldType(dims, complex_t(0.0, 0.0));
    this->norm_per_site = typename DeviceScalarFieldType<rank>::type(dims, 0.0);
    this->dot_product_per_site =
        typename DeviceFieldType<rank>::type(dims, complex_t(0.0, 0.0));
  }

  Solver(const SpinorFieldType& b,
         SpinorFieldType& x,
         const DiracOp& dirac_op,
         SpinorFieldType& xk,
         SpinorFieldType& rk,
         SpinorFieldType& apk,
         SpinorFieldType& temp_D,
         SpinorFieldType& pk,
         typename DeviceScalarFieldType<rank>::type& norm_per_site,
         typename DeviceFieldType<rank>::type(dot_product_per_site))
      : b(b),
        x(x),
        dirac_op(dirac_op),
        xk(xk),
        rk(rk),
        apk(apk),
        temp_D(temp_D),
        pk(pk),
        norm_per_site(norm_per_site),
        dot_product_per_site(dot_product_per_site) {}

  template <typename Tag>
  void solve(const SpinorFieldType& x0, const real_t& tol) {
    Kokkos::Profiling::pushRegion("Solver");
    static_cast<Derived*>(this)->template solve_int<Tag>(x0, tol);
    Kokkos::Profiling::popRegion();
  }

  /// @brief Constructs the b vector when using an Even/Odd Precondition Field,
  /// assumes that the field saved in b is the even part of the Vector b
  /// @param odd_b
  void construct_problem(const SpinorFieldType& odd_b) {
    auto out_even_from_odd_b = dirac_op.template apply<Tags::TagHeo>(odd_b);
    axpy<DSpinorFieldType>(this->dirac_op.params.kappa, out_even_from_odd_b,
                           this->b, this->b);
  }

  /// @brief Reconstructs the Odd part of the solution
  /// with zero odd part
  void reconstruct_solution_0(SpinorFieldType& out) {
    dirac_op.template apply<Tags::TagHoe>(this->x, out);
    ax<DSpinorFieldType>(this->dirac_op.params.kappa, out, out);
  }
  SpinorFieldType reconstruct_solution_0() {
    auto out = SpinorFieldType(x.dimensions, complex_t(0.0, 0.0));
    reconstruct_solution_0(out);
    return out;
  }
  /// @brief Reconstructs the Odd part of the solution
  /// @param out
  void reconstruct_solution(const SpinorFieldType& odd_b,
                            SpinorFieldType& out) {
    dirac_op.template apply<Tags::TagHoe>(this->x, out);
    axpy<DSpinorFieldType>(this->dirac_op.params.kappa, out, odd_b, out);
  }
  SpinorFieldType reconstruct_solution(const SpinorFieldType& odd_b) {
    auto out = SpinorFieldType(x.dimensions, complex_t(0.0, 0.0));
    reconstruct_solution(odd_b, out);
    return out;
  }
};
// // Deduction guide for Solver
// template <typename Operator, typename SpinorType>
// Solver(const SpinorType&, SpinorType&, const Operator&)
//     -> Solver<typename Operator::Derived,
//               SpinorType::rank,
//               SpinorType::Nc,
//               SpinorType::RepDim>;

template < class DiracOpT>
class CGSolver
    : public Solver<CGSolver,DiracOpT> {
  // using DSpinorFieldType =
  //     typename DiracOpFieldTypeTraits<DiracOperator>::DSpinorFieldType;
  // using DGaugeFieldType =
  //     typename DiracOpFieldTypeTraits<DiracOperator>::DGaugeFieldType;

 public:
 using Base = Solver<CGSolver, DiracOpT>;
   using Base::Base;
   using DSpinorFieldType = typename Base::DSpinorFieldType;
   using DGaugeFieldType = typename Base::DGaugeFieldType;
  using SpinorFieldType = typename Base::DSpinorFieldType::type;
  using GaugeFieldType = typename Base::DGaugeFieldType::type;
  constexpr static size_t rank =
      DeviceFermionFieldTypeTraits<typename Base::DSpinorFieldType>::Rank;
  constexpr static size_t Nc =
      DeviceFermionFieldTypeTraits<typename Base::DSpinorFieldType>::Nc;
  constexpr static size_t RepDim =
      DeviceFermionFieldTypeTraits<typename Base::DSpinorFieldType>::RepDim;
  static_assert((rank == DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank) &&
                (Nc == DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Nc));

  


  template <typename Tag>
  void solve_int(const SpinorFieldType& x0, const real_t& tol) {
    Kokkos::deep_copy(this->xk.field, x0.field);  // x_0

    axpy<DSpinorFieldType>(-1, this->dirac_op.template apply<Tag>(this->xk),
                           this->b, this->rk);

    Kokkos::deep_copy(this->pk.field, this->rk.field);  // p_0 // d_0
    real_t rk_norm = spinor_norm<rank, Nc, RepDim>(
        this->rk, this->norm_per_site);  //\delta_0
    int num_iter = 0;
    while (rk_norm > tol) {
      this->dirac_op.template apply<Tag>(this->pk, this->temp_D, this->apk);
      // z = Ad_k
      const complex_t rkrk = spinor_dot_product<rank, Nc, RepDim>(
          this->rk, this->rk, this->dot_product_per_site);
      const complex_t alpha =
          (rkrk / spinor_dot_product<rank, Nc, RepDim>(
                      this->pk, this->apk,
                      this->dot_product_per_site));  // Always real
      axpy<DSpinorFieldType>(alpha, this->pk, this->xk, this->xk);
      // xk = spinor_add_mul<rank, Nc, RepDim>(xk, pk, alpha);
      // xk = axpy<DSpinorFieldType>(alpha, pk, xk);
      axpy<DSpinorFieldType>(-alpha, this->apk, this->rk, this->rk);
      // rk = axpy<DSpinorFieldType>(-alpha, apk, rk);
      // rk = spinor_sub_mul<rank, Nc, RepDim>(rk, apk, alpha);
      const complex_t beta =
          (spinor_dot_product<rank, Nc, RepDim>(this->rk, this->rk,
                                                this->dot_product_per_site) /
           rkrk);
      axpy<DSpinorFieldType>(beta, this->pk, this->rk, this->pk);
      // pk = axpy<DSpinorFieldType>(beta, pk, rk);
      // pk = spinor_add_mul<rank, Nc, RepDim>(rk, pk, beta);
      // Check if swapping is needed of pk and rk, should be correct

      rk_norm = spinor_norm<rank, Nc, RepDim>(this->rk, this->norm_per_site);
      num_iter++;
      if (KLFT_VERBOSITY > 2) {
        printf("CG Iteration %d: rk_norm = %.15f\n", num_iter, rk_norm);
        if (KLFT_VERBOSITY > 3) {
          printf("Norm of (b - A*x) %.15f\n",
                 spinor_norm<rank, Nc, RepDim>(
                     axpy<DSpinorFieldType>(
                         -1.0, this->dirac_op.template apply<Tag>(this->xk),
                         this->b),
                     this->norm_per_site));
        }
      }
    }
    this->dirac_op.template apply<Tag>(this->xk, this->apk, this->temp_D);
    axpy<DSpinorFieldType>(-1, this->temp_D, this->b, this->temp_D);
    const real_t ex_res =
        spinor_norm<rank, Nc, RepDim>(this->temp_D, this->norm_per_site);

    if (Kokkos::abs(ex_res / spinor_norm<rank, Nc, RepDim>(
                                 this->xk, this->norm_per_site)) > tol) {
      printf(
          "EX_res: %.20f, Roundoff Error, relaunching CG solver with new "
          "initial guess\n",
          ex_res);
      this->template solve<Tag>(this->xk, tol);
    } else {
      if (KLFT_VERBOSITY > 1) {
        printf("CG solver converged in %d iterations\n", num_iter);
      }
      this->x = this->xk;
    }
  }
};

template <class DiracOpT>
class BiCGStab
    : public Solver<BiCGStab, DiracOpT> {
  // using DSpinorFieldType =
  //     typename DiracOpFieldTypeTraits<DiracOperator>::DSpinorFieldType;
  // using DGaugeFieldType =
  //     typename DiracOpFieldTypeTraits<DiracOperator>::DGaugeFieldType;

 public:
  using Base = Solver<BiCGStab, DiracOpT>;
   using Base::Base;
   using DSpinorFieldType = typename Base::DSpinorFieldType;
   using DGaugeFieldType = typename Base::DGaugeFieldType;
  using SpinorFieldType = typename Base::DSpinorFieldType::type;
  using GaugeFieldType = typename Base::DGaugeFieldType::type;
  constexpr static size_t rank =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank;
  constexpr static size_t Nc =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Nc;
  constexpr static size_t RepDim =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::RepDim;
  static_assert((rank == DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank) &&
                (Nc == DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Nc));



  template <typename Tag>
  void solve_int(const SpinorFieldType& x0, const real_t& tol) {
    auto dims = x0.dimensions;

    SpinorFieldType r0{dims, complex_t(0.0, 0.0)};

    SpinorFieldType temp_D{dims, complex_t(0.0, 0.0)};

    SpinorFieldType s(dims, complex_t(0.0, 0.0));
    SpinorFieldType t(dims, complex_t(0.0, 0.0));
    Kokkos::deep_copy(this->xk.field, x0.field);  // x_0
    axpy<DSpinorFieldType>(-1, this->dirac_op.template apply<Tag>(this->xk),
                           this->b, r0);
    Kokkos::deep_copy(this->rk.field, r0.field);

    Kokkos::deep_copy(this->pk.field, r0.field);         // p_0 // d_0
    real_t rk_norm = spinor_norm<rank, Nc, RepDim>(r0);  //\delta_0
    int num_iter = 0;
    while (rk_norm > tol) {
      this->dirac_op.template apply<Tag>(this->pk, this->temp_D, this->apk);
      // z = Ad_k
      const complex_t r0rk = spinor_dot_product<rank, Nc, RepDim>(r0, this->rk);
      const complex_t alpha = (r0rk / spinor_dot_product<rank, Nc, RepDim>(
                                          r0, this->apk));  // Always real
      axpy<DSpinorFieldType>(-alpha, this->apk, this->rk,
                             s);  // s = rk -alpha*apk
      this->dirac_op.template apply<Tag>(s, this->temp_D, t);  // t = xk
      // axpy<DSpinorFieldType>(alpha, pk, xk, xk);
      const complex_t omega = (spinor_dot_product<rank, Nc, RepDim>(t, s) /
                               spinor_dot_product<rank, Nc, RepDim>(t, t));
      axpy<DSpinorFieldType>(-omega, t, s, this->rk);        // rk= s-omega A*s
      axpy<DSpinorFieldType>(omega, s, this->xk, this->xk);  // xk = xk+omega s
      axpy<DSpinorFieldType>(alpha, this->pk, this->xk,
                             this->xk);  // xk = xk+alpha pk , so intotal xk =
                                         // xk +alpha pk +omega s

      const auto beta =
          (spinor_dot_product<rank, Nc, RepDim>(r0, this->rk) / r0rk) *
          (alpha / omega);
      axpy<DSpinorFieldType>(beta, this->pk, this->rk,
                             this->pk);  // pk = rk + beta * pk
      axpy<DSpinorFieldType>(
          -beta * omega, this->apk, this->pk,
          this->pk);  // pk = pk - beta * omega * apk; so in total
                      // pk = rk + beta * pk - beta * omega * apk;

      // pk = axpy<DSpinorFieldType>(beta, pk, rk);
      // pk = spinor_add_mul<rank, Nc, RepDim>(rk, pk, beta);
      // Check if swapping is needed of pk and rk, should be correct

      rk_norm = spinor_norm<rank, Nc, RepDim>(this->rk);
      num_iter++;
      if (KLFT_VERBOSITY > 2) {
        printf("BiCGstab Iteration %d: rk_norm = %.15f\n", num_iter, rk_norm);
        if (KLFT_VERBOSITY > 3) {
          printf("Norm of (b - A*x) %.15f\n",
                 spinor_norm<rank, Nc, RepDim>(axpy<DSpinorFieldType>(
                     -1.0, this->dirac_op.template apply<Tag>(this->xk),
                     this->b)));
        }
      }
    }
    this->dirac_op.template apply<Tag>(this->xk, this->apk, this->temp_D);
    axpy<DSpinorFieldType>(-1, this->temp_D, this->b, this->temp_D);
    const real_t ex_res = spinor_norm<rank, Nc, RepDim>(this->temp_D);

    if (Kokkos::abs(ex_res / spinor_norm<rank, Nc, RepDim>(this->xk)) > tol) {
      printf(
          "EX_res: %.20f, Roundoff Error, relaunching CG solver with new "
          "initial guess\n",
          ex_res);
      this->template solve<Tag>(this->xk, tol);
    } else {
      if (KLFT_VERBOSITY > 1) {
        printf("BiCGstab solver converged in %d iterations\n", num_iter);
      }
      this->x = this->xk;
    }
  }
};
// After CGSolver class definition
// template <typename D, typename S>
// CGSolver(S, S, D) -> CGSolver<D, S, typename D::DGaugeFieldType::type>;
}  // namespace klft
