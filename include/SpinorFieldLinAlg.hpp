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
#include "GLOBAL.hpp"
#include "Spinor.hpp"
#include "Tuner.hpp"

//  For  now in an external file, should be in SpinorField.hpp
namespace klft {

template <size_t rank, size_t Nc, size_t RepDim>
struct SpinorDotProduct {
  using SpinorFieldType =
      typename DeviceSpinorFieldType<rank, Nc, RepDim>::type;
  const SpinorFieldType a;
  const SpinorFieldType b;
  using FieldType = typename DeviceFieldType<rank>::type;
  FieldType dot_product_per_site;

  const IndexArray<rank> dimensions;
  SpinorDotProduct(const SpinorFieldType& a,
                   const SpinorFieldType& b,
                   FieldType& dot_product_per_site,
                   const IndexArray<rank>& dimensions)
      : a(a),
        b(b),
        dot_product_per_site(dot_product_per_site),
        dimensions(dimensions) {}

  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(const Indices... Idcs) const {
    complex_t temp_result = spinor_inner_prod(a(Idcs...), b(Idcs...));
    dot_product_per_site(Idcs...) = temp_result;
  }
};

template <size_t rank, size_t Nc, size_t RepDim>
KOKKOS_FORCEINLINE_FUNCTION complex_t spinor_dot_product(
    const typename DeviceSpinorFieldType<rank, Nc, RepDim>::type& a,
    const typename DeviceSpinorFieldType<rank, Nc, RepDim>::type& b) {
  assert(a.dimensions == b.dimensions);
  static_assert(
      Kokkos::SpaceAccessibility<
          typename decltype(a.field)::execution_space,
          typename decltype(b.field)::memory_space>::accessible,
      "Execution space of A cannot access memory space of B");  // allow only
                                                                // device-device
                                                                // or host-host
                                                                // interaction
  complex_t result = 0.0;
  IndexArray<rank> start;
  IndexArray<rank> end;
  for (index_t i = 0; i < rank; ++i) {
    start[i] = 0;
    end[i] = a.dimensions[i];
  }
  // temporary field for storing results per site
  // direct reduction is slow
  // this field will be summed over in the end
  using FieldType = typename DeviceFieldType<rank>::type;
  FieldType dot_product_per_site(end, complex_t(0.0, 0.0));
  SpinorDotProduct<rank, Nc, RepDim> SDP(a, b, dot_product_per_site, end);

  tune_and_launch_for<rank>("SpinorField_dot_product", start, end, SDP);
  Kokkos::fence();
  result = dot_product_per_site.sum();
  Kokkos::fence();
  return result;
}

template <size_t rank, size_t Nc, size_t RepDim>
struct SpinorNorm {
  using SpinorFieldType =
      typename DeviceSpinorFieldType<rank, Nc, RepDim>::type;
  const SpinorFieldType a;
  using FieldType = typename DeviceScalarFieldType<rank>::type;
  FieldType norm_per_site;
  const IndexArray<rank> dimensions;

  SpinorNorm(const SpinorFieldType& a,
             FieldType& norm_per_site,
             const IndexArray<rank>& dimensions)
      : a(a), norm_per_site(norm_per_site), dimensions(dimensions) {}

  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(const Indices... Idcs) const {
    real_t temp_result = sqnorm(a(Idcs...));
    norm_per_site(Idcs...) = temp_result;
  }
};

template <size_t rank, size_t Nc, size_t RepDim>
KOKKOS_FORCEINLINE_FUNCTION real_t spinor_norm_sq(
    const typename DeviceSpinorFieldType<rank, Nc, RepDim>::type& a) {
  real_t result = 0.0;
  IndexArray<rank> start;
  IndexArray<rank> end;
  for (index_t i = 0; i < rank; ++i) {
    start[i] = 0;
    end[i] = a.dimensions[i];
  }
  // temporary field for storing results per site
  // direct reduction is slow
  // this field will be summed over in the end
  using FieldType = typename DeviceScalarFieldType<rank>::type;
  real_t init = 0;
  FieldType norm_per_site(end, init);
  SpinorNorm<rank, Nc, RepDim> norm(a, norm_per_site, end);

  tune_and_launch_for<rank>("SpinorField_norm", start, end, norm);
  Kokkos::fence();
  result = norm_per_site.sum();
  Kokkos::fence();
  return result;
}
template <size_t rank, size_t Nc, size_t RepDim>
KOKKOS_FORCEINLINE_FUNCTION real_t
spinor_norm(const typename DeviceSpinorFieldType<rank, Nc, RepDim>::type& a) {
  return Kokkos::sqrt(spinor_norm_sq<rank, Nc, RepDim>(a));
}

template <size_t rank, size_t Nc, size_t RepDim>
struct axpyFunctor {
  using SpinorFieldType =
      typename DeviceSpinorFieldType<rank, Nc, RepDim>::type;
  const SpinorFieldType x;
  const SpinorFieldType y;
  const complex_t alpha;
  SpinorFieldType c;
  const IndexArray<rank> dimensions;
  axpyFunctor(const complex_t& alpha,
              const SpinorFieldType& x,
              const SpinorFieldType& y,
              SpinorFieldType& c,
              const IndexArray<rank>& dimensions)
      : x(x), y(y), c(c), alpha(alpha), dimensions(dimensions) {}
  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(const Indices... Idcs) const {
    axpy(alpha, x(Idcs...), y(Idcs...), c(Idcs...));
    // (y(Idcs...) += (alpha * x(Idcs...)));
  }
};
/// @brief Calculates alpha*x+y
/// @tparam rank
/// @tparam Nc
/// @tparam RepDim
/// @param alpha
/// @param x
/// @param y
/// @return c = alpha*x+y
template <size_t rank, size_t Nc, size_t RepDim>
typename DeviceSpinorFieldType<rank, Nc, RepDim>::type
    KOKKOS_FORCEINLINE_FUNCTION
    axpy(const complex_t& alpha,
         const typename DeviceSpinorFieldType<rank, Nc, RepDim>::type& x,
         const typename DeviceSpinorFieldType<rank, Nc, RepDim>::type& y) {
  assert(x.dimensions == y.dimensions);
  static_assert(
      Kokkos::SpaceAccessibility<
          typename decltype(x.field)::execution_space,
          typename decltype(y.field)::memory_space>::accessible,
      "Execution space of A cannot access memory space of B");  // allow only
                                                                // device-device
                                                                // or host-host
                                                                // interaction

  using SpinorFieldType =
      typename DeviceSpinorFieldType<rank, Nc, RepDim>::type;
  SpinorFieldType c(x.dimensions, complex_t(0.0, 0.0));
  IndexArray<rank> start{};
  axpyFunctor<rank, Nc, RepDim> add(alpha, x, y, c, x.dimensions);

  tune_and_launch_for<rank>("SpinorField_axpy", start, x.dimensions, add);
  Kokkos::fence();
  return c;
}

template <size_t rank, size_t Nc, size_t RepDim>
void KOKKOS_FORCEINLINE_FUNCTION
axpy(const complex_t& alpha,
     const typename DeviceSpinorFieldType<rank, Nc, RepDim>::type& x,
     const typename DeviceSpinorFieldType<rank, Nc, RepDim>::type& y,
     typename DeviceSpinorFieldType<rank, Nc, RepDim>::type& c) {
  assert(x.dimensions == y.dimensions);
  static_assert(
      Kokkos::SpaceAccessibility<
          typename decltype(x.field)::execution_space,
          typename decltype(y.field)::memory_space>::accessible,
      "Execution space of A cannot access memory space of B");  // allow only
                                                                // device-device
                                                                // or host-host
                                                                // interaction
  IndexArray<rank> start{};

  axpyFunctor<rank, Nc, RepDim> add(alpha, x, y, c, x.dimensions);

  tune_and_launch_for<rank>("SpinorField_axpy_inplace", start, x.dimensions,
                            add);
  Kokkos::fence();
}
}  // namespace klft