#pragma once

#include "GLOBAL.hpp"
#include "Tuner.hpp"

namespace klft {
template <size_t Nd>
struct deviceLinkScalarField {
  deviceLinkScalarField() = default;

  // initialize all sites to a given value
  deviceLinkScalarField(const index_t L0,
                        const index_t L1,
                        const index_t L2,
                        const index_t L3,
                        const real_t init)
      : dimensions({L0, L1, L2, L3}) {
    do_init(L0, L1, L2, L3, field, init);
  }

  deviceLinkScalarField(const IndexArray<Nd>& dimensions, const real_t init)
      : dimensions(dimensions) {
    do_init(field);
  }

  LinkScalarField<Nd> field;
  IndexArray<Nd> dimensions;
  void do_init(LinkScalarField<Nd>& V) {
    Kokkos::realloc(Kokkos::WithoutInitializing, V, dimensions[0],
                    dimensions[1], dimensions[2], dimensions[3]);
    tune_and_launch_for<Nd>(
        "init_LinkScalarField", IndexArray<Nd>{0}, dimensions,
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2,
                      const index_t i3) {
#pragma unroll
          for (index_t mu = 0; mu < Nd; ++mu) {
            V(i0, i1, i2, i3, mu) = 0;
          }
        });
    Kokkos::fence();
  }
  // define accessors for the field
  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION real_t& operator()(const indexType i0,
                                                 const indexType i1,
                                                 const indexType i2,
                                                 const indexType i3,
                                                 const index_t mu) const {
    return field(i0, i1, i2, i3, mu);
  }

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION real_t& operator()(const indexType i0,
                                                 const indexType i1,
                                                 const indexType i2,
                                                 const indexType i3,
                                                 const index_t mu) {
    return field(i0, i1, i2, i3, mu);
  }

  // define accessors with 4D Kokkos array
  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION real_t& operator()(
      const Kokkos::Array<indexType, 4> site,
      const index_t mu) const {
    return field(site[0], site[1], site[2], site[3], mu);
  }

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION real_t& operator()(
      const Kokkos::Array<indexType, 4> site,
      const index_t mu) {
    return field(site[0], site[1], site[2], site[3], mu);
  }
  real_t sum() const {
    real_t sum = 0.0;
    Kokkos::parallel_reduce(
        "sum_deviceField", Policy<4>({0, 0, 0, 0}, dimensions),
        KOKKOS_CLASS_LAMBDA(const index_t i0, const index_t i1,
                            const index_t i2, const index_t i3, real_t& lsum) {
          for (index_t mu = 0; mu < Nd; mu++) {
            lsum += field(i0, i1, i2, i3, mu);
          }
        },
        Kokkos::Sum<real_t>(sum));
    return sum;
  }

  real_t avg() const {
    real_t sum = 0.0;
    sum = this->sum();
    return sum /
           (dimensions[0] * dimensions[1] * dimensions[2] * dimensions[3] * Nd);
  }
};
template <size_t Nd>
struct deviceLinkScalarField3D {
  deviceLinkScalarField3D() = default;

  // initialize all sites to a given value
  deviceLinkScalarField3D(const index_t L0,
                          const index_t L1,
                          const index_t L2,

                          const real_t init)
      : dimensions({L0, L1, L2}) {
    do_init(L0, L1, L2, field, init);
  }

  deviceLinkScalarField3D(const IndexArray<Nd>& dimensions, const real_t init)
      : dimensions(dimensions) {
    do_init(field);
  }
  LinkScalarField3D<Nd> field;
  IndexArray<Nd> dimensions;
  void do_init(LinkScalarField3D<Nd>& V) {
    Kokkos::realloc(Kokkos::WithoutInitializing, V, dimensions[0],
                    dimensions[1], dimensions[2]);
    tune_and_launch_for<Nd>(
        "init_LinkScalarField", IndexArray<Nd>{0}, dimensions,
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2) {
#pragma unroll
          for (index_t mu = 0; mu < Nd; ++mu) {
            V(i0, i1, i2, mu) = 0;
          }
        });
    Kokkos::fence();
  }
  // define accessors for the field
  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION real_t& operator()(const indexType i0,
                                                 const indexType i1,
                                                 const indexType i2,
                                                 const index_t mu) const {
    return field(i0, i1, i2, mu);
  }

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION real_t& operator()(const indexType i0,
                                                 const indexType i1,
                                                 const indexType i2,
                                                 const index_t mu) {
    return field(i0, i1, i2, mu);
  }

  // define accessors with 3D Kokkos array
  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION real_t& operator()(
      const Kokkos::Array<indexType, 3> site,
      const index_t mu) const {
    return field(site[0], site[1], site[2], mu);
  }

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION real_t& operator()(
      const Kokkos::Array<indexType, 3> site,
      const index_t mu) {
    return field(site[0], site[1], site[2], mu);
  }
  real_t sum() const {
    real_t sum = 0.0;
    Kokkos::parallel_reduce(
        "sum_deviceField3D", Policy<3>({0, 0, 0}, dimensions),
        KOKKOS_CLASS_LAMBDA(const index_t i0, const index_t i1,
                            const index_t i2, real_t& lsum) {
          for (index_t mu = 0; mu < Nd; mu++) {
            lsum += field(i0, i1, i2, mu);
          }
        },
        Kokkos::Sum<real_t>(sum));
    return sum;
  }
  real_t avg() const {
    real_t sum = 0.0;
    sum = this->sum();
    return sum / (dimensions[0] * dimensions[1] * dimensions[2] * Nd);
  }
};

template <size_t Nd>
struct deviceLinkScalarField2D {
  deviceLinkScalarField2D() = default;

  // initialize all sites to a given value
  deviceLinkScalarField2D(const index_t L0, const index_t L1, const real_t init)
      : dimensions({L0, L1}) {
    do_init(L0, L1, field, init);
  }

  deviceLinkScalarField2D(const IndexArray<Nd>& dimensions, const real_t init)
      : dimensions(dimensions) {
    do_init(field);
  }

  LinkScalarField2D<Nd> field;
  IndexArray<Nd> dimensions;
  void do_init(LinkScalarField2D<Nd>& V) {
    Kokkos::realloc(Kokkos::WithoutInitializing, V, dimensions[0],
                    dimensions[1]);
    tune_and_launch_for<Nd>(
        "init_LinkScalarField", IndexArray<Nd>{0}, dimensions,
        KOKKOS_LAMBDA(const index_t i0, const index_t i1) {
#pragma unroll
          for (index_t mu = 0; mu < Nd; ++mu) {
            V(i0, i1, mu) = 0;
          }
        });
    Kokkos::fence();
  }

  // define accessors for the field
  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION real_t& operator()(const indexType i0,
                                                 const indexType i1,
                                                 const index_t mu) const {
    return field(i0, i1, mu);
  }

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION real_t& operator()(const indexType i0,
                                                 const indexType i1,
                                                 const index_t mu) {
    return field(i0, i1, mu);
  }

  // define accessors with 2D Kokkos array
  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION real_t& operator()(
      const Kokkos::Array<indexType, 2> site,
      const index_t mu) const {
    return field(site[0], site[1], mu);
  }

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION real_t& operator()(
      const Kokkos::Array<indexType, 2> site,
      const index_t mu) {
    return field(site[0], site[1], mu);
  }

  real_t sum() const {
    real_t sum = 0.0;
    Kokkos::parallel_reduce(
        "sum_deviceField2D", Policy<2>({0, 0}, dimensions),
        KOKKOS_CLASS_LAMBDA(const index_t i0, const index_t i1, real_t& lsum) {
          for (index_t mu = 0; mu < Nd; mu++) {
            lsum += field(i0, i1, mu);
          }
        },
        Kokkos::Sum<real_t>(sum));
    return sum;
  }
  real_t avg() const {
    real_t sum = 0.0;
    sum = this->sum();
    return sum / (dimensions[0] * dimensions[1] * Nd);
  }
};
}  // namespace klft
