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
#include "GammaMatrix.hpp"
#include "Spinor.hpp"

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
    const typename DeviceSpinorFieldType<rank, Nc, RepDim>::type& b,
    typename DeviceFieldType<rank>::type& dot_product_per_site) {
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
  IndexArray<rank> start{};

  // temporary field for storing results per site
  // direct reduction is slow
  // this field will be summed over in the end
  // using FieldType = typename DeviceFieldType<rank>::type;
  // FieldType dot_product_per_site(end, complex_t(0.0, 0.0));
  SpinorDotProduct<rank, Nc, RepDim> SDP(a, b, dot_product_per_site,
                                         a.dimensions);

  KTune::parallel_for("SpinorField_dot_product",
                      Policy<rank>(start, a.dimensions), SDP);
  Kokkos::fence();
  result = dot_product_per_site.sum();
  Kokkos::fence();
  return result;
}
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

  // temporary field for storing results per site
  // direct reduction is slow
  // this field will be summed over in the end
  using FieldType = typename DeviceFieldType<rank>::type;
  FieldType dot_product_per_site(a.dimensions, complex_t(0.0, 0.0));

  Kokkos::fence();
  return spinor_dot_product<rank, Nc, RepDim>(a, b, dot_product_per_site);
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
KOKKOS_FORCEINLINE_FUNCTION real_t
spinor_norm_sq(const typename DeviceSpinorFieldType<rank, Nc, RepDim>::type& a,
               typename DeviceScalarFieldType<rank>::type& norm_per_site) {
  real_t result = 0.0;
  IndexArray<rank> start{};

  // temporary field for storing results per site
  // direct reduction is slow
  // this field will be summed over in the end
  // using FieldType = typename DeviceScalarFieldType<rank>::type;
  // real_t init = 0;
  // FieldType norm_per_site(end, init);
  SpinorNorm<rank, Nc, RepDim> norm(a, norm_per_site, a.dimensions);
  KTune::parallel_for("SpinorField_norm", Policy<rank>(start, a.dimensions),
                      norm);
  Kokkos::fence();
  result = norm_per_site.sum();
  Kokkos::fence();
  return result;
}
template <size_t rank, size_t Nc, size_t RepDim>
KOKKOS_FORCEINLINE_FUNCTION real_t spinor_norm_sq(
    const typename DeviceSpinorFieldType<rank, Nc, RepDim>::type& a) {
  // temporary field for storing results per site
  // direct reduction is slow
  // this field will be summed over in the end
  using FieldType = typename DeviceScalarFieldType<rank>::type;
  real_t init = 0;
  FieldType norm_per_site(a.dimensions, init);
  Kokkos::fence();

  return spinor_norm_sq<rank, Nc, RepDim>(a, norm_per_site);
}
template <size_t rank, size_t Nc, size_t RepDim>
KOKKOS_FORCEINLINE_FUNCTION real_t
spinor_norm(const typename DeviceSpinorFieldType<rank, Nc, RepDim>::type& a) {
  return Kokkos::sqrt(spinor_norm_sq<rank, Nc, RepDim>(a));
}
template <size_t rank, size_t Nc, size_t RepDim>
KOKKOS_FORCEINLINE_FUNCTION real_t
spinor_norm(const typename DeviceSpinorFieldType<rank, Nc, RepDim>::type& a,
            typename DeviceScalarFieldType<rank>::type& norm_per_site) {
  return Kokkos::sqrt(spinor_norm_sq<rank, Nc, RepDim>(a, norm_per_site));
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
    // axpy(alpha, x(Idcs...), y(Idcs...), c(Idcs...));
    c(Idcs...) = (y(Idcs...) + (alpha * x(Idcs...)));
  }
};

template <size_t rank, size_t Nc, size_t RepDim>
struct axpyG5Functor {
  using SpinorFieldType =
      typename DeviceSpinorFieldType<rank, Nc, RepDim>::type;
  const SpinorFieldType x;
  const SpinorFieldType y;
  const complex_t alpha;
  SpinorFieldType c;
  const IndexArray<rank> dimensions;
  axpyG5Functor(const complex_t& alpha,
                const SpinorFieldType& x,
                const SpinorFieldType& y,
                SpinorFieldType& c,
                const IndexArray<rank>& dimensions)
      : x(x), y(y), c(c), alpha(alpha), dimensions(dimensions) {}
  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(const Indices... Idcs) const {
    // axpy(alpha, x(Idcs...), y(Idcs...), c(Idcs...));
    c(Idcs...) = gamma5(y(Idcs...) + (alpha * x(Idcs...)));
  }
};
/// @brief Calculates alpha*x+y

/// @param alpha
/// @param x
/// @param y
/// @return c = alpha*x+y
template <typename DSpinorFieldType>
typename DSpinorFieldType::type KOKKOS_FORCEINLINE_FUNCTION
axpy(const complex_t& alpha,
     const typename DSpinorFieldType::type& x,
     const typename DSpinorFieldType::type& y) {
  constexpr static size_t rank =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank;
  constexpr static size_t Nc =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Nc;
  constexpr static size_t RepDim =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::RepDim;
  assert(x.dimensions == y.dimensions);
  static_assert(
      Kokkos::SpaceAccessibility<
          typename decltype(x.field)::execution_space,
          typename decltype(y.field)::memory_space>::accessible,
      "Execution space of A cannot access memory space of B");  // allow only
                                                                // device-device
                                                                // or host-host
                                                                // interaction

  using SpinorFieldType = typename DSpinorFieldType::type;
  SpinorFieldType c(x.dimensions, complex_t(0.0, 0.0));
  IndexArray<rank> start{};
  axpyFunctor<rank, Nc, RepDim> add(alpha, x, y, c, x.dimensions);

  KTune::parallel_for("SpinorField_axpy", Policy<rank>(start, x.dimensions),
                      add);
  Kokkos::fence();
  return c;
}
/// @brief Calculates alpha*x+y
/// @param alpha
/// @param x
/// @param y
/// @param c
/// @return c = alpha*x+y
template <typename DSpinorFieldType>
void KOKKOS_FORCEINLINE_FUNCTION axpy(const complex_t& alpha,
                                      const typename DSpinorFieldType::type& x,
                                      const typename DSpinorFieldType::type& y,
                                      typename DSpinorFieldType::type& c) {
  constexpr static size_t rank =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank;
  constexpr static size_t Nc =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Nc;
  constexpr static size_t RepDim =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::RepDim;
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

  KTune::parallel_for("SpinorField_axpy_inplace",
                      Policy<rank>(start, x.dimensions), add);
  Kokkos::fence();
}

/// @brief Calculates gamma5(alpha*x+y)

/// @param alpha
/// @param x
/// @param y
/// @return c = gamma5(alpha*x+y)
template <typename DSpinorFieldType>
typename DSpinorFieldType::type KOKKOS_FORCEINLINE_FUNCTION
axpyG5(const complex_t& alpha,
       const typename DSpinorFieldType::type& x,
       const typename DSpinorFieldType::type& y) {
  constexpr static size_t rank =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank;
  constexpr static size_t Nc =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Nc;
  constexpr static size_t RepDim =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::RepDim;
  assert(x.dimensions == y.dimensions);
  static_assert(
      Kokkos::SpaceAccessibility<
          typename decltype(x.field)::execution_space,
          typename decltype(y.field)::memory_space>::accessible,
      "Execution space of A cannot access memory space of B");  // allow only
                                                                // device-device
                                                                // or host-host
                                                                // interaction

  using SpinorFieldType = typename DSpinorFieldType::type;
  SpinorFieldType c(x.dimensions, complex_t(0.0, 0.0));
  IndexArray<rank> start{};
  axpyG5Functor<rank, Nc, RepDim> add(alpha, x, y, c, x.dimensions);

  KTune::parallel_for("SpinorField_G5axpy", Policy<rank>(start, x.dimensions),
                      add);
  Kokkos::fence();
  return c;
}
/// @brief Calculates alpha*x+y
/// @param alpha
/// @param x
/// @param y
/// @param c
/// @return c = gamma5(alpha*x+y)
template <typename DSpinorFieldType>
void KOKKOS_FORCEINLINE_FUNCTION
axpyG5(const complex_t& alpha,
       const typename DSpinorFieldType::type& x,
       const typename DSpinorFieldType::type& y,
       typename DSpinorFieldType::type& c) {
  constexpr static size_t rank =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank;
  constexpr static size_t Nc =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Nc;
  constexpr static size_t RepDim =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::RepDim;
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
  axpyG5Functor<rank, Nc, RepDim> add(alpha, x, y, c, x.dimensions);

  KTune::parallel_for("SpinorField_g5axpy_inplace",
                      Policy<rank>(start, x.dimensions), add);
  Kokkos::fence();
}
template <size_t rank, size_t Nc, size_t RepDim>
struct axFunctor {
  using SpinorFieldType =
      typename DeviceSpinorFieldType<rank, Nc, RepDim>::type;
  const SpinorFieldType x;
  const complex_t alpha;
  SpinorFieldType c;
  const IndexArray<rank> dimensions;
  axFunctor(const complex_t& alpha,
            const SpinorFieldType& x,
            SpinorFieldType& c,
            const IndexArray<rank>& dimensions)
      : x(x), c(c), alpha(alpha), dimensions(dimensions) {}
  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(const Indices... Idcs) const {
    // axpy(alpha, x(Idcs...), y(Idcs...), c(Idcs...));
    c(Idcs...) = (alpha * x(Idcs...));
  }
};
/// @brief Calculates alpha*x
/// @param alpha
/// @param x
/// @return c = alpha*x
template <typename DSpinorFieldType>
typename DSpinorFieldType::type KOKKOS_FORCEINLINE_FUNCTION
ax(const complex_t& alpha, const typename DSpinorFieldType::type& x) {
  constexpr static size_t rank =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank;
  constexpr static size_t Nc =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Nc;
  constexpr static size_t RepDim =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::RepDim;

  using SpinorFieldType = typename DSpinorFieldType::type;
  SpinorFieldType c(x.dimensions, complex_t(0.0, 0.0));
  IndexArray<rank> start{};
  axFunctor<rank, Nc, RepDim> add(alpha, x, c, x.dimensions);

  KTune::parallel_for("SpinorField_ax", Policy<rank>(start, x.dimensions), add);
  Kokkos::fence();
  return c;
}
/// @brief Calculates alpha*x
/// @param alpha
/// @param x
/// @param y
/// @param c
/// @return c = alpha*x
template <typename DSpinorFieldType>
void KOKKOS_FORCEINLINE_FUNCTION ax(const complex_t& alpha,
                                    const typename DSpinorFieldType::type& x,
                                    typename DSpinorFieldType::type& c) {
  constexpr static size_t rank =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank;
  constexpr static size_t Nc =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Nc;
  constexpr static size_t RepDim =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::RepDim;
  assert(x.dimensions == c.dimensions);
  static_assert(
      Kokkos::SpaceAccessibility<
          typename decltype(x.field)::execution_space,
          typename decltype(c.field)::memory_space>::accessible,
      "Execution space of A cannot access memory space of B");  // allow only
                                                                // device-device
                                                                // or host-host
                                                                // interaction
  IndexArray<rank> start{};
  axFunctor<rank, Nc, RepDim> add(alpha, x, c, x.dimensions);

  KTune::parallel_for("SpinorField_ax_inplace",
                      Policy<rank>(start, x.dimensions), add);
  Kokkos::fence();
}
template <size_t rank, size_t Nc, size_t RepDim>
struct axG5Functor {
  using SpinorFieldType =
      typename DeviceSpinorFieldType<rank, Nc, RepDim>::type;
  const SpinorFieldType x;
  const complex_t alpha;
  SpinorFieldType c;
  const IndexArray<rank> dimensions;
  axG5Functor(const complex_t& alpha,
              const SpinorFieldType& x,
              SpinorFieldType& c,
              const IndexArray<rank>& dimensions)
      : x(x), c(c), alpha(alpha), dimensions(dimensions) {}
  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(const Indices... Idcs) const {
    // axpy(alpha, x(Idcs...), y(Idcs...), c(Idcs...));
    c(Idcs...) = gamma5((alpha * x(Idcs...)));
  }
};
template <typename DSpinorFieldType>
typename DSpinorFieldType::type KOKKOS_FORCEINLINE_FUNCTION
axG5(const complex_t& alpha, const typename DSpinorFieldType::type& x) {
  constexpr static size_t rank =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank;
  constexpr static size_t Nc =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Nc;
  constexpr static size_t RepDim =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::RepDim;

  using SpinorFieldType = typename DSpinorFieldType::type;
  SpinorFieldType c(x.dimensions, complex_t(0.0, 0.0));
  IndexArray<rank> start{};
  axG5Functor<rank, Nc, RepDim> add(alpha, x, c, x.dimensions);

  KTune::parallel_for("SpinorField_g5ax", Policy<rank>(start, x.dimensions),
                      add);
  Kokkos::fence();
  return c;
}
/// @brief Calculates alpha*x
/// @param alpha
/// @param x
/// @param y
/// @param c
/// @return c = alpha*x
template <typename DSpinorFieldType>
void KOKKOS_FORCEINLINE_FUNCTION axG5(const complex_t& alpha,
                                      const typename DSpinorFieldType::type& x,
                                      typename DSpinorFieldType::type& c) {
  constexpr static size_t rank =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank;
  constexpr static size_t Nc =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Nc;
  constexpr static size_t RepDim =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::RepDim;
  assert(x.dimensions == c.dimensions);
  static_assert(
      Kokkos::SpaceAccessibility<
          typename decltype(x.field)::execution_space,
          typename decltype(c.field)::memory_space>::accessible,
      "Execution space of A cannot access memory space of B");  // allow only
                                                                // device-device
                                                                // or host-host
                                                                // interaction
  IndexArray<rank> start{};
  axG5Functor<rank, Nc, RepDim> add(alpha, x, c, x.dimensions);

  KTune::parallel_for("SpinorField_g5ax_inplace",
                      Policy<rank>(start, x.dimensions), add);
  Kokkos::fence();
}
template <size_t rank, size_t Nc, size_t RepDim>
struct axpbyG5Functor {
  using SpinorFieldType =
      typename DeviceSpinorFieldType<rank, Nc, RepDim>::type;
  const SpinorFieldType x;
  const SpinorFieldType y;
  const complex_t alpha;
  const complex_t beta;
  SpinorFieldType c;
  const IndexArray<rank> dimensions;
  axpbyG5Functor(const complex_t& alpha,
                 const SpinorFieldType& x,
                 const complex_t& beta,
                 const SpinorFieldType& y,
                 SpinorFieldType& c,
                 const IndexArray<rank>& dimensions)
      : x(x), y(y), c(c), beta(beta), alpha(alpha), dimensions(dimensions) {}
  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(const Indices... Idcs) const {
    // axpy(alpha, x(Idcs...), y(Idcs...), c(Idcs...));
    c(Idcs...) = gamma5(beta * y(Idcs...) + (alpha * x(Idcs...)));
  }
};
/// @brief Calculates gamma5(alpha*x+beta*y)

/// @param alpha
/// @param x
/// @param beta
/// @param y
/// @return c = gamma5(alpha*x+betay)
template <typename DSpinorFieldType>
typename DSpinorFieldType::type KOKKOS_FORCEINLINE_FUNCTION
axpbyG5(const complex_t& alpha,
        const typename DSpinorFieldType::type& x,
        const complex_t& beta,
        const typename DSpinorFieldType::type& y) {
  constexpr static size_t rank =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank;
  constexpr static size_t Nc =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Nc;
  constexpr static size_t RepDim =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::RepDim;
  assert(x.dimensions == y.dimensions);
  static_assert(
      Kokkos::SpaceAccessibility<
          typename decltype(x.field)::execution_space,
          typename decltype(y.field)::memory_space>::accessible,
      "Execution space of A cannot access memory space of B");  // allow only
                                                                // device-device
                                                                // or
                                                                // host-host
                                                                // interaction

  using SpinorFieldType = typename DSpinorFieldType::type;
  SpinorFieldType c(x.dimensions, complex_t(0.0, 0.0));
  IndexArray<rank> start{};
  axpbyG5Functor<rank, Nc, RepDim> add(alpha, x, beta, y, c, x.dimensions);

  KTune::parallel_for("SpinorField_g5axpby", Policy<rank>(start, x.dimensions),
                      add);
  Kokkos::fence();
  return c;
}

/// @brief Calculates gamma5(alpha*x+beta*y)

/// @param alpha
/// @param x
/// @param beta
/// @param y
/// @return c = gamma5(alpha*x+betay)
template <typename DSpinorFieldType>
void KOKKOS_FORCEINLINE_FUNCTION
axpbyG5(const complex_t& alpha,
        const typename DSpinorFieldType::type& x,
        const complex_t& beta,
        const typename DSpinorFieldType::type& y,
        typename DSpinorFieldType::type& c) {
  constexpr static size_t rank =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank;
  constexpr static size_t Nc =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Nc;
  constexpr static size_t RepDim =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::RepDim;
  assert(x.dimensions == y.dimensions);
  static_assert(
      Kokkos::SpaceAccessibility<
          typename decltype(x.field)::execution_space,
          typename decltype(y.field)::memory_space>::accessible,
      "Execution space of A cannot access memory space of B");  // allow only
                                                                // device-device
                                                                // or
                                                                // host-host
                                                                // interaction
  IndexArray<rank> start{};
  axpbyG5Functor<rank, Nc, RepDim> add(alpha, x, beta, y, c, x.dimensions);

  KTune::parallel_for("SpinorField_g5axpby_inplace",
                      Policy<rank>(start, x.dimensions), add);
  Kokkos::fence();
}

/*------------------------------*/

template <size_t rank, size_t Nc, size_t RepDim>
struct axpbyFunctor {
  using SpinorFieldType =
      typename DeviceSpinorFieldType<rank, Nc, RepDim>::type;
  const SpinorFieldType x;
  const SpinorFieldType y;
  const complex_t alpha;
  const complex_t beta;
  SpinorFieldType c;
  const IndexArray<rank> dimensions;
  axpbyFunctor(const complex_t& alpha,
               const SpinorFieldType& x,
               const complex_t& beta,
               const SpinorFieldType& y,
               SpinorFieldType& c,
               const IndexArray<rank>& dimensions)
      : x(x), y(y), c(c), beta(beta), alpha(alpha), dimensions(dimensions) {}
  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(const Indices... Idcs) const {
    // axpy(alpha, x(Idcs...), y(Idcs...), c(Idcs...));
    c(Idcs...) = (beta * y(Idcs...) + (alpha * x(Idcs...)));
  }
};
/// @brief Calculates (alpha*x+beta*y)

/// @param alpha
/// @param x
/// @param beta
/// @param y
/// @return c = (alpha*x+betay)
template <typename DSpinorFieldType>
typename DSpinorFieldType::type KOKKOS_FORCEINLINE_FUNCTION
axpby(const complex_t& alpha,
      const typename DSpinorFieldType::type& x,
      const complex_t& beta,
      const typename DSpinorFieldType::type& y) {
  constexpr static size_t rank =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank;
  constexpr static size_t Nc =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Nc;
  constexpr static size_t RepDim =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::RepDim;
  assert(x.dimensions == y.dimensions);
  static_assert(
      Kokkos::SpaceAccessibility<
          typename decltype(x.field)::execution_space,
          typename decltype(y.field)::memory_space>::accessible,
      "Execution space of A cannot access memory space of B");  // allow only
                                                                // device-device
                                                                // or
                                                                // host-host
                                                                // interaction

  using SpinorFieldType = typename DSpinorFieldType::type;
  SpinorFieldType c(x.dimensions, complex_t(0.0, 0.0));
  IndexArray<rank> start{};
  axpbyFunctor<rank, Nc, RepDim> add(alpha, x, beta, y, c, x.dimensions);

  KTune::parallel_for("SpinorField_axpby", Policy<rank>(start, x.dimensions),
                      add);
  Kokkos::fence();
  return c;
}
/// @brief Calculates (alpha*x+beta*y)

/// @param alpha
/// @param x
/// @param beta
/// @param y
/// @return c = (alpha*x+betay)
template <typename DSpinorFieldType>
void KOKKOS_FORCEINLINE_FUNCTION axpby(const complex_t& alpha,
                                       const typename DSpinorFieldType::type& x,
                                       const complex_t& beta,
                                       const typename DSpinorFieldType::type& y,
                                       typename DSpinorFieldType::type& c) {
  constexpr static size_t rank =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank;
  constexpr static size_t Nc =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Nc;
  constexpr static size_t RepDim =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::RepDim;
  assert(x.dimensions == y.dimensions);
  static_assert(
      Kokkos::SpaceAccessibility<
          typename decltype(x.field)::execution_space,
          typename decltype(y.field)::memory_space>::accessible,
      "Execution space of A cannot access memory space of B");  // allow only
                                                                // device-device
                                                                // or
                                                                // host-host
                                                                // interaction
  IndexArray<rank> start{};
  axpbyFunctor<rank, Nc, RepDim> add(alpha, x, beta, y, c, x.dimensions);

  KTune::parallel_for("SpinorField_axpby_inplace",
                      Policy<rank>(start, x.dimensions), add);
  Kokkos::fence();
}

}  // namespace klft
