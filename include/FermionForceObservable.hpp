#include "FieldTypeHelper.hpp"
#include "GLOBAL.hpp"
namespace klft {

// return the maximum value of ReTr(1 - plaquette) of the lattice
template <typename DLinkScalarField>
real_t get_MaxForce(const typename DLinkScalarField::type field) {
  real_t rtn = 0.0;
  constexpr index_t Nd =
      DeviceLinkScalarFieldTypeTraits<DLinkScalarField>::Rank;
  if constexpr (Nd == 4) {
    auto policy = Policy<Nd>({0, 0, 0, 0}, field.dimensions);
    Kokkos::parallel_reduce(
        "get h (sp_max)", policy,
        KOKKOS_LAMBDA(size_t i0, size_t i1, size_t i2, size_t i3,
                      real_t& local_max) {
          // GPlaq(i0, i1, i2, i3);
          real_t s = field(i0, i1, i2, i3, 0);
#pragma unroll
          for (index_t mu = 1; mu < Nd; ++mu) {
            real_t tmp = field(i0, i1, i2, i3, mu);
            s = tmp > s ? tmp : s;
          }
          local_max = Kokkos::max(local_max, s);
        },
        Kokkos::Max<real_t>(rtn));
    Kokkos::fence();
  }
  return rtn;
}
// return the maximum value of ReTr(1 - plaquette) of the lattice
template <typename DAdjFieldType>
typename DeviceLinkScalarFieldType<
    DeviceAdjFieldTypeTraits<DAdjFieldType>::Rank>::type
get_force_per_site(const typename DAdjFieldType::type& adj_field) {
  static const size_t Nd = DeviceAdjFieldTypeTraits<DAdjFieldType>::Rank;
  // static_assert(Nd == 4);
  static const size_t Nc = DeviceAdjFieldTypeTraits<DAdjFieldType>::Nc;

  using AdjFieldType = typename DAdjFieldType::type;
  using FieldType = typename DeviceLinkScalarFieldType<Nd>::type;
  FieldType force_per_site(adj_field.dimensions, 0.0);
  if constexpr (Nd == 4) {
    tune_and_launch_for<Nd>(
        "get force", IndexArray<Nd>{0, 0, 0, 0}, adj_field.dimensions,
        KOKKOS_LAMBDA(size_t i0, size_t i1, size_t i2, size_t i3) {
          // GPlaq(i0, i1, i2, i3);

          for (index_t mu = 0; mu < Nd; ++mu) {
            force_per_site(i0, i1, i2, i3, mu) =
                norm2(adj_field(i0, i1, i2, i3, mu));
          }
        });
    Kokkos::fence();
  }

  return force_per_site;
}
}  // namespace klft