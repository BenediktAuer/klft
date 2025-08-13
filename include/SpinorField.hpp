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

// define structs for initializing SUN field

#pragma once
#include "GLOBAL.hpp"
#include "Tuner.hpp"
// Nc number of colors
// RepDim Dimension of Gamma matrices, Nd = RepDim
namespace klft {
template <size_t _Nc, size_t _RepDim>
struct deviceSpinorField {
  static const size_t rank = 4;  // SpinorField is always 4D
  static const size_t Nc = _Nc;
  static const size_t RepDim =
      _RepDim;  // RepDim is the dimension of the Gamma matrices
  deviceSpinorField() = default;

  // initialize all sites to a given value

  deviceSpinorField(const index_t L0,
                    const index_t L1,
                    const index_t L2,
                    const index_t L3,
                    const complex_t init)
      : dimensions({L0, L1, L2, L3}) {
    do_init(L0, L1, L2, L3, field, init);
  }

  deviceSpinorField(const IndexArray<4>& dimensions, const complex_t init)
      : dimensions(dimensions) {
    do_init(dimensions[0], dimensions[1], dimensions[2], dimensions[3], field,
            init);
  }

  // initialize all sites to given Spinor
  deviceSpinorField(const index_t L0,
                    const index_t L1,
                    const index_t L2,
                    const index_t L3,
                    const Spinor<Nc, RepDim>& init)
      : dimensions({L0, L1, L2, L3}) {
    do_init(L0, L1, L2, L3, field, init);
  }
  deviceSpinorField(const IndexArray<4>& dimensions,
                    const Spinor<Nc, RepDim>& init)
      : dimensions(dimensions) {
    do_init(dimensions[0], dimensions[1], dimensions[2], dimensions[3], field,
            init);
  }

  // initialize all latice size to random value drawn from a Normal Distribution
  // N(mean,var)
  template <class RNG>
  deviceSpinorField(const IndexArray<4>& dimensions,
                    RNG& rng,
                    const real_t& mean,
                    const real_t& var)
      : dimensions(dimensions) {
    do_init(dimensions[0], dimensions[1], dimensions[2], dimensions[3], field,
            rng, mean, var);
  }

  // initialize all latice size to random value drawn from a Normal Distribution
  // N(mean,var)
  template <class RNG>
  deviceSpinorField(const index_t L0,
                    const index_t L1,
                    const index_t L2,
                    const index_t L3,
                    RNG& rng,
                    const real_t& mean,
                    const real_t& var)
      : dimensions({L0, L1, L2, L3}) {
    do_init(dimensions[0], dimensions[1], dimensions[2], dimensions[3], field,
            rng, mean, var);
  }

  // Initialize SpinorField with Single Value
  void do_init(const index_t L0,
               const index_t L1,
               const index_t L2,
               const index_t L3,
               SpinorField<Nc, RepDim>& V,
               const complex_t init) {
    Kokkos::realloc(Kokkos::WithoutInitializing, V, L0, L1, L2, L3);
    tune_and_launch_for<4>(
        "init_deviceSpinorField", IndexArray<4>{0, 0, 0, 0},
        IndexArray<4>{L0, L1, L2, L3},
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2,
                      const index_t i3) {
#pragma unroll
          for (index_t c1 = 0; c1 < Nc; ++c1) {
#pragma unroll
            for (index_t c2 = 0; c2 < RepDim; ++c2) {
              V(i0, i1, i2, i3)[c1][c2] = init;
            }
          }
        });
    Kokkos::fence();
  }
  void do_init(const index_t L0,
               const index_t L1,
               const index_t L2,
               const index_t L3,
               SpinorField<Nc, RepDim>& V,
               const Spinor<Nc, RepDim>& init) {
    Kokkos::realloc(Kokkos::WithoutInitializing, V, L0, L1, L2, L3);
    tune_and_launch_for<4>(
        "init_deviceSpinorField", IndexArray<4>{0, 0, 0, 0},
        IndexArray<4>{L0, L1, L2, L3},
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2,
                      const index_t i3) { V(i0, i1, i2, i3) = init; });
    Kokkos::fence();
  }
  template <class RNG>
  void do_init(const index_t L0,
               const index_t L1,
               const index_t L2,
               const index_t L3,
               SpinorField<Nc, RepDim>& V,
               RNG& rng,
               const real_t& mean,
               const real_t& std) {
    Kokkos::realloc(Kokkos::WithoutInitializing, V, L0, L1, L2, L3);
    tune_and_launch_for<4>(
        "init_deviceSpinorField", IndexArray<4>{0, 0, 0, 0},
        IndexArray<4>{L0, L1, L2, L3},
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2,
                      const index_t i3) {
          auto generator = rng.get_state();
#pragma unroll
          for (index_t c1 = 0; c1 < Nc; ++c1) {
#pragma unroll
            for (index_t c2 = 0; c2 < RepDim; ++c2) {
              V(i0, i1, i2, i3)
              [c1][c2] = complex_t(generator.normal(mean, std),
                                   generator.normal(mean, std));
            }
          }
          // free state here corrct in l.138 of GaugeField it isn't done.
          rng.free_state(generator);
        });
    Kokkos::fence();
  }

  SpinorField<Nc, RepDim> field;
  IndexArray<4> dimensions;

  // define accessors
  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, RepDim>& operator()(
      const indexType i0,
      const indexType i1,
      const indexType i2,
      const indexType i3) const {
    return field(i0, i1, i2, i3);
  }

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, RepDim>& operator()(
      const indexType i0,
      const indexType i1,
      const indexType i2,
      const indexType i3) {
    return field(i0, i1, i2, i3);
  }

  // define accessors with 4D Kokkos array
  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, RepDim>& operator()(
      const Kokkos::Array<indexType, 4> site) const {
    return field(site[0], site[1], site[2], site[3]);
  }

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, RepDim>& operator()(
      const Kokkos::Array<indexType, 4> site) {
    return field(site[0], site[1], site[2], site[3]);
  }
};
template <size_t _Nc, size_t _RepDim>
struct deviceSpinorFieldConst {
  static const size_t rank = 4;  // SpinorField is always 4D
  static const size_t Nc = _Nc;
  static const size_t RepDim =
      _RepDim;  // RepDim is the dimension of the Gamma matrices
  deviceSpinorFieldConst(const deviceSpinorField<Nc, RepDim>& a)
      : field(a.field), dimensions(a.dimensions) {}
  deviceSpinorFieldConst() = default;

  constSpinorField<Nc, RepDim> field;
  IndexArray<4> dimensions;

  // define accessors
  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, RepDim> operator()(
      const indexType i0,
      const indexType i1,
      const indexType i2,
      const indexType i3) const {
    return field(i0, i1, i2, i3);
  }

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, RepDim>& operator()(
      const indexType i0,
      const indexType i1,
      const indexType i2,
      const indexType i3) {
    return field(i0, i1, i2, i3);
  }

  // define accessors with 4D Kokkos array
  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, RepDim> operator()(
      const Kokkos::Array<indexType, 4> site) const {
    return field(site[0], site[1], site[2], site[3]);
  }

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, RepDim>& operator()(
      const Kokkos::Array<indexType, 4> site) {
    return field(site[0], site[1], site[2], site[3]);
  }
};

template <size_t _Nc, size_t _RepDim>
struct deviceSpinorField3D {
  static const size_t rank = 3;  // SpinorField is always 4D
  static const size_t Nc = _Nc;
  static const size_t RepDim =
      _RepDim;  // RepDim is the dimension of the Gamma matrices
  deviceSpinorField3D() = default;

  // initialize all sites to a given value

  deviceSpinorField3D(const index_t L0,
                      const index_t L1,
                      const index_t L2,
                      const complex_t init)
      : dimensions({L0, L1, L2}) {
    do_init(L0, L1, L2, field, init);
  }

  deviceSpinorField3D(const IndexArray<rank>& dimensions, const complex_t init)
      : dimensions(dimensions) {
    do_init(dimensions[0], dimensions[1], dimensions[2], field, init);
  }

  // initialize all sites to given Spinor
  deviceSpinorField3D(const index_t L0,
                      const index_t L1,
                      const index_t L2,
                      const Spinor<Nc, RepDim>& init)
      : dimensions({L0, L1, L2}) {
    do_init(L0, L1, L2, field, init);
  }
  deviceSpinorField3D(const IndexArray<rank>& dimensions,
                      const Spinor<Nc, RepDim>& init)
      : dimensions(dimensions) {
    do_init(dimensions[0], dimensions[1], dimensions[2], field, init);
  }

  // initialize all latice size to random value drawn from a Normal Distribution
  // N(mean,var)
  template <class RNG>
  deviceSpinorField3D(const IndexArray<rank>& dimensions,
                      RNG& rng,
                      const real_t& mean,
                      const real_t& var)
      : dimensions(dimensions) {
    do_init(dimensions[0], dimensions[1], dimensions[2], field, rng, mean, var);
  }

  // initialize all latice size to random value drawn from a Normal Distribution
  // N(mean,var)
  template <class RNG>
  deviceSpinorField3D(const index_t L0,
                      const index_t L1,
                      const index_t L2,
                      RNG& rng,
                      const real_t& mean,
                      const real_t& var)
      : dimensions({L0, L1, L2}) {
    do_init(dimensions[0], dimensions[1], dimensions[2], field, rng, mean, var);
  }

  // Initialize SpinorField with Single Value
  void do_init(const index_t L0,
               const index_t L1,
               const index_t L2,
               SpinorField3D<Nc, RepDim>& V,
               const complex_t init) {
    Kokkos::realloc(Kokkos::WithoutInitializing, V, L0, L1, L2);
    tune_and_launch_for<rank>(
        "init_deviceSpinorField3D", IndexArray<rank>{0, 0, 0},
        IndexArray<rank>{L0, L1, L2},
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2) {
#pragma unroll
          for (index_t c1 = 0; c1 < Nc; ++c1) {
#pragma unroll
            for (index_t c2 = 0; c2 < RepDim; ++c2) {
              V(i0, i1, i2)[c1][c2] = init;
            }
          }
        });
    Kokkos::fence();
  }
  void do_init(const index_t L0,
               const index_t L1,
               const index_t L2,
               SpinorField3D<Nc, RepDim>& V,
               const Spinor<Nc, RepDim>& init) {
    Kokkos::realloc(Kokkos::WithoutInitializing, V, L0, L1, L2);
    tune_and_launch_for<rank>(
        "init_deviceSpinorField3D", IndexArray<rank>{0, 0, 0},
        IndexArray<rank>{L0, L1, L2},
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2) {
          V(i0, i1, i2) = init;
        });
    Kokkos::fence();
  }
  template <class RNG>
  void do_init(const index_t L0,
               const index_t L1,
               const index_t L2,
               SpinorField3D<Nc, RepDim>& V,
               RNG& rng,
               const real_t& mean,
               const real_t& std) {
    Kokkos::realloc(Kokkos::WithoutInitializing, V, L0, L1, L2);
    tune_and_launch_for<rank>(
        "init_deviceSpinorField3D", IndexArray<rank>{0, 0, 0},
        IndexArray<rank>{L0, L1, L2},
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2) {
          auto generator = rng.get_state();
#pragma unroll
          for (index_t c1 = 0; c1 < Nc; ++c1) {
#pragma unroll
            for (index_t c2 = 0; c2 < RepDim; ++c2) {
              V(i0, i1, i2)
              [c1][c2] = complex_t(generator.normal(mean, std),
                                   generator.normal(mean, std));
            }
          }
          // free state here corrct in l.138 of GaugeField it isn't done.
          rng.free_state(generator);
        });
    Kokkos::fence();
  }

  SpinorField3D<Nc, RepDim> field;
  IndexArray<rank> dimensions;

  // define accessors
  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, RepDim>&
  operator()(const indexType i0, const indexType i1, const indexType i2) const {
    return field(i0, i1, i2);
  }

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, RepDim>&
  operator()(const indexType i0, const indexType i1, const indexType i2) {
    return field(i0, i1, i2);
  }

  // define accessors with 4D Kokkos array
  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, RepDim>& operator()(
      const Kokkos::Array<indexType, rank> site) const {
    return field(site[0], site[1], site[2]);
  }

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, RepDim>& operator()(
      const Kokkos::Array<indexType, rank> site) {
    return field(site[0], site[1], site[2]);
  }
};
template <size_t _Nc, size_t _RepDim>
struct deviceSpinorFieldConst3D {
  static const size_t rank = 3;  // SpinorField is always 4D
  static const size_t Nc = _Nc;
  static const size_t RepDim =
      _RepDim;  // RepDim is the dimension of the Gamma matrices
  deviceSpinorFieldConst3D(const deviceSpinorField<Nc, RepDim>& a)
      : field(a.field), dimensions(a.dimensions) {}
  deviceSpinorFieldConst3D() = default;

  constSpinorField3D<Nc, RepDim> field;
  IndexArray<4> dimensions;

  // define accessors
  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, RepDim>
  operator()(const indexType i0, const indexType i1, const indexType i2) const {
    return field(i0, i1, i2);
  }

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, RepDim>&
  operator()(const indexType i0, const indexType i1, const indexType i2) {
    return field(i0, i1, i2);
  }

  // define accessors with 4D Kokkos array
  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, RepDim> operator()(
      const Kokkos::Array<indexType, rank> site) const {
    return field(site[0], site[1], site[2]);
  }

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, RepDim>& operator()(
      const Kokkos::Array<indexType, rank> site) {
    return field(site[0], site[1], site[2]);
  }
};

template <size_t _Nc, size_t _RepDim>
struct deviceSpinorField2D {
  static const size_t rank = 2;  // SpinorField is always 4D
  static const size_t Nc = _Nc;
  static const size_t RepDim =
      _RepDim;  // RepDim is the dimension of the Gamma matrices
  deviceSpinorField2D() = default;

  // initialize all sites to a given value

  deviceSpinorField2D(const index_t L0, const index_t L1, const complex_t init)
      : dimensions({L0, L1}) {
    do_init(L0, L1, field, init);
  }

  deviceSpinorField2D(const IndexArray<rank>& dimensions, const complex_t init)
      : dimensions(dimensions) {
    do_init(dimensions[0], dimensions[1], field, init);
  }

  // initialize all sites to given Spinor
  deviceSpinorField2D(const index_t L0,
                      const index_t L1,
                      const Spinor<Nc, RepDim>& init)
      : dimensions({L0, L1}) {
    do_init(L0, L1, field, init);
  }
  deviceSpinorField2D(const IndexArray<rank>& dimensions,
                      const Spinor<Nc, RepDim>& init)
      : dimensions(dimensions) {
    do_init(dimensions[0], dimensions[1], field, init);
  }

  // initialize all latice size to random value drawn from a Normal Distribution
  // N(mean,var)
  template <class RNG>
  deviceSpinorField2D(const IndexArray<rank>& dimensions,
                      RNG& rng,
                      const real_t& mean,
                      const real_t& var)
      : dimensions(dimensions) {
    do_init(dimensions[0], dimensions[1], field, rng, mean, var);
  }

  // initialize all latice size to random value drawn from a Normal Distribution
  // N(mean,var)
  template <class RNG>
  deviceSpinorField2D(const index_t L0,
                      const index_t L1,
                      RNG& rng,
                      const real_t& mean,
                      const real_t& var)
      : dimensions({L0, L1}) {
    do_init(dimensions[0], dimensions[1], field, rng, mean, var);
  }

  // Initialize SpinorField with Single Value
  void do_init(const index_t L0,
               const index_t L1,
               SpinorField2D<Nc, RepDim>& V,
               const complex_t init) {
    Kokkos::realloc(Kokkos::WithoutInitializing, V, L0, L1);
    tune_and_launch_for<rank>(
        "init_deviceSpinorField2D", IndexArray<rank>{0, 0},
        IndexArray<rank>{L0, L1},
        KOKKOS_LAMBDA(const index_t i0, const index_t i1) {
#pragma unroll
          for (index_t c1 = 0; c1 < Nc; ++c1) {
#pragma unroll
            for (index_t c2 = 0; c2 < RepDim; ++c2) {
              V(i0, i1)[c1][c2] = init;
            }
          }
        });
    Kokkos::fence();
  }
  void do_init(const index_t L0,
               const index_t L1,
               SpinorField2D<Nc, RepDim>& V,
               const Spinor<Nc, RepDim>& init) {
    Kokkos::realloc(Kokkos::WithoutInitializing, V, L0, L1);
    tune_and_launch_for<rank>(
        "init_deviceSpinorField2D", IndexArray<rank>{0, 0},
        IndexArray<rank>{L0, L1},
        KOKKOS_LAMBDA(const index_t i0, const index_t i1) {
          V(i0, i1) = init;
        });
    Kokkos::fence();
  }
  template <class RNG>
  void do_init(const index_t L0,
               const index_t L1,
               SpinorField2D<Nc, RepDim>& V,
               RNG& rng,
               const real_t& mean,
               const real_t& std) {
    Kokkos::realloc(Kokkos::WithoutInitializing, V, L0, L1);
    tune_and_launch_for<rank>(
        "init_deviceSpinorField2D", IndexArray<rank>{0, 0},
        IndexArray<rank>{L0, L1},
        KOKKOS_LAMBDA(const index_t i0, const index_t i1) {
          auto generator = rng.get_state();
#pragma unroll
          for (index_t c1 = 0; c1 < Nc; ++c1) {
#pragma unroll
            for (index_t c2 = 0; c2 < RepDim; ++c2) {
              V(i0, i1)
              [c1][c2] = complex_t(generator.normal(mean, std),
                                   generator.normal(mean, std));
            }
          }
          // free state here corrct in l.138 of GaugeField it isn't done.
          rng.free_state(generator);
        });
    Kokkos::fence();
  }

  SpinorField2D<Nc, RepDim> field;
  IndexArray<rank> dimensions;

  // define accessors
  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, RepDim>& operator()(
      const indexType i0,
      const indexType i1) const {
    return field(i0, i1);
  }

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, RepDim>& operator()(
      const indexType i0,
      const indexType i1) {
    return field(i0, i1);
  }

  // define accessors with 4D Kokkos array
  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, RepDim>& operator()(
      const Kokkos::Array<indexType, rank> site) const {
    return field(site[0], site[1]);
  }

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, RepDim>& operator()(
      const Kokkos::Array<indexType, rank> site) {
    return field(site[0], site[1]);
  }
};

template <size_t _Nc, size_t _RepDim>
struct deviceSpinorFieldConst2D {
  static const size_t rank = 2;  // SpinorField is always 4D
  static const size_t Nc = _Nc;
  static const size_t RepDim =
      _RepDim;  // RepDim is the dimension of the Gamma matrices
  deviceSpinorFieldConst2D(const deviceSpinorField2D<Nc, RepDim>& a)
      : field(a.field), dimensions(a.dimensions) {}
  deviceSpinorFieldConst2D() = default;

  constSpinorField2D<Nc, RepDim> field;
  IndexArray<rank> dimensions;

  // define accessors
  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, RepDim> operator()(
      const indexType i0,
      const indexType i1) const {
    return field(i0, i1);
  }

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, RepDim>& operator()(
      const indexType i0,
      const indexType i1) {
    return field(i0, i1);
  }

  // define accessors with 4D Kokkos array
  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, RepDim> operator()(
      const Kokkos::Array<indexType, rank> site) const {
    return field(site[0], site[1]);
  }
  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, RepDim>& operator()(
      const Kokkos::Array<indexType, rank> site) {
    return field(site[0], site[1]);
  }
};

}  // namespace klft