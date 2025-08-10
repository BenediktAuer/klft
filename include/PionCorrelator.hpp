#include "FermionParams.hpp"
#include "FieldTypeHelper.hpp"
#include "GDiracOperator.hpp"
#include "GLOBAL.hpp"
#include "PropagatorMatrix.hpp"
#include "Solver.hpp"
#include "Spinor.hpp"
#include "SpinorFieldLinAlg.hpp"
#include "SpinorPointSource.hpp"
#include "Tuner.hpp"
namespace klft {

template <typename DSpinorFieldType,
          typename DGaugeFieldType,
          template <template <typename, typename> class DiracOpT,
                    typename,
                    typename> class _Solver,
          template <typename, typename> class DiracOpT>
std::vector<real_t> PionCorrelator(
    const typename DGaugeFieldType::type& g_in,
    const diracParams<DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank,
                      DeviceFermionFieldTypeTraits<DSpinorFieldType>::RepDim>&
        params,
    const real_t& tol) {
  static_assert(isDeviceGaugeFieldType<DGaugeFieldType>::value);
  constexpr static size_t rank =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank;
  constexpr static size_t Nc = DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Nc;
  constexpr static size_t RepDim =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::RepDim;
  using SpinorFieldSource =
      typename WithSpinorFieldKind<DSpinorFieldType,
                                   SpinorFieldKind::PointSource>::type;
  using SpinorField = typename DSpinorFieldType::type;
  using DiracOperator =
      DiracOperator<DiracOpT, DSpinorFieldType, DGaugeFieldType>;
  using Solver = _Solver<DiracOpT, DSpinorFieldType, DGaugeFieldType>;
  DiracOperator dirac_op(g_in, params);
  auto Nt = g_in.field.extent(0);

  std::vector<real_t> result_vec;

  if constexpr (rank == 4) {
    typename DevicePropagator<rank, Nc, RepDim>::type result(
        g_in.dimensions, complex_t(0.0, 0.0));
    for (index_t alpha0 = 0; alpha0 < Nc * RepDim; alpha0++) {
      SpinorFieldSource source(g_in.dimensions, IndexArray<rank>{}, alpha0);
      SpinorField x(g_in.dimensions, 0);
      SpinorField x0(g_in.dimensions, 0);
      Solver solver(SpinorField(source), x, dirac_op);
      solver.template solve<Tags::TagDdaggerD>(x0, tol);
      auto prop = dirac_op.template apply<Tags::TagDdagger>(solver.x);
      tune_and_launch_for<rank>(
          "init_deviceSpinorField", IndexArray<RepDim>{0, 0, 0, 0},
          g_in.dimensions,
          KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2,
                        const index_t i3) {
            add_inplace(result(i0, i1, i2, i3), prop(i0, i1, i2, i3), alpha0);
          });
      Kokkos::fence();
      // Function
    }
    // at the end vecotor with length Nt, maybe new view with only one dimension
    // to do the device Reduction
    for (size_t i0 = 0; i0 < g_in.dimensions[0]; i0++) {
      real_t res;

      Kokkos::parallel_reduce(
          "Reductor",
          Policy<rank - 1>(
              IndexArray<rank - 1>{},
              IndexArray<rank - 1>{g_in.dimensions[1], g_in.dimensions[2],
                                   g_in.dimensions[3]}),
          KOKKOS_LAMBDA(const size_t& i1, const size_t& i2, const size_t& i3,
                        real_t& upd) {
#pragma unroll
            for (size_t alpha = 0; alpha < Nc * RepDim; alpha++) {
#pragma unroll
              for (size_t beta = 0; beta < Nc * RepDim; beta++) {
                upd += (result(i0, i1, i2, i3)[beta][alpha] *
                        conj(result(i0, i1, i2, i3)[beta][alpha]))
                           .real();
              }
            }
          },
          res);

      result_vec.push_back(res);
    }
  }
  return result_vec;
}

}  // namespace klft
