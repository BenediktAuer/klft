#pragma once
#include "FieldTypeHelper.hpp"
#include "GLOBAL.hpp"
#include "Gauge_Util.hpp"
namespace klft {
struct APESmearingParams {
  index_t n_steps;
  real_t alpha;

  APESmearingParams() {
    n_steps = 9;
    alpha = 0.5;
  };
};

template <typename DGaugeFieldType>
struct APESmearingFunctor {
  // implement the Wilson flow, for now the field will not be copied, but it
  // will be flown in place -> copying needs to be done before
  constexpr static const size_t rank =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank;
  constexpr static const size_t Nc =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Nc;
  constexpr static const GaugeFieldKind Kind =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Kind;
  const APESmearingParams params;

  // get the correct deviceGaugeFieldType
  using GaugeFieldT = typename DGaugeFieldType::type;
  const GaugeFieldT s_in;
  GaugeFieldT tmp_staple;
  GaugeFieldT s_out;

  APESmearingFunctor() = delete;

  APESmearingFunctor(GaugeFieldT& s_out,
                     const GaugeFieldT& s_in,
                     GaugeFieldT& staple_field,
                     const APESmearingParams& _params)
      : params(_params), s_in(s_in), s_out(s_out), tmp_staple(staple_field) {}

  // execute the wilson flow
  void smear() {  // todo: check this once by saving a staple field and once
                  // by locally calculating the staple

    spatialstapleField<DGaugeFieldType>(this->s_in, this->tmp_staple);
    // Kokkos::fence();

    KTune::parallel_for("APESmearing",
                        Policy<rank>(IndexArray<rank>{}, s_out.dimensions),
                        *this);
    // Kokkos::fence();
  }

  template <typename... Indices>
  KOKKOS_INLINE_FUNCTION void operator()(const Indices... Idcs) const {
    // spatial directions only
    for (index_t mu = 0; mu < rank - 1; ++mu) {
      auto U = (1.0 - params.alpha) * s_in(Idcs..., mu) +
               params.alpha / (2.0 * (rank - 2)) * tmp_staple(Idcs..., mu);
      restoreSUN(U);
      s_out(Idcs..., mu) = U;
    }

    // copy temporal link unchanged
    s_out(Idcs..., rank - 1) = s_in(Idcs..., rank - 1);
  }
};
template <typename DGaugeFieldType>
void APEsmearing(const typename DGaugeFieldType::type& g_in,
                 typename DGaugeFieldType::type& g_temp,
                 typename DGaugeFieldType::type& g_out,
                 typename DGaugeFieldType::type& staple_field,
                 const APESmearingParams& params) {
  constexpr static const size_t rank =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank;
  constexpr static const size_t Nc =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Nc;
  constexpr static const GaugeFieldKind Kind =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Kind;
  APESmearingFunctor<DGaugeFieldType> f_in_to_out(g_out, g_in, staple_field,
                                                  params);
  f_in_to_out.smear();

  // Remaining steps: ping-pong between g_out and g_temp
  APESmearingFunctor<DGaugeFieldType> f_out_to_temp(g_temp, g_out, staple_field,
                                                    params);
  APESmearingFunctor<DGaugeFieldType> f_temp_to_out(g_out, g_temp, staple_field,
                                                    params);

  for (int i = 1; i < params.n_steps; i++) {
    if (i % 2 == 1)
      f_out_to_temp.smear();
    else
      f_temp_to_out.smear();
  }

  // If n_steps was even, the final result is in g_temp, so copy it to g_out
  if (params.n_steps > 0 && params.n_steps % 2 == 0) {
    Kokkos::deep_copy(g_out.field, g_temp.field);
  }
}
template <typename DGaugeFieldType>
auto APEsmearing(const typename DGaugeFieldType::type& g_in,
                 const APESmearingParams& params) {
  typename DGaugeFieldType::type g_temp(g_in.dimensions, complex_t(0));
  typename DGaugeFieldType::type g_out(g_in.dimensions, complex_t(0));
  typename DGaugeFieldType::type staple_field(g_in.dimensions, complex_t(0));
  APEsmearing<DGaugeFieldType>(g_in, g_temp, g_out, staple_field, params);
  return g_out;
}
}  // namespace klft
