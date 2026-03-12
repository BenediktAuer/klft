#pragma once
#include "FieldTypeHelper.hpp"
#include "GLOBAL.hpp"
#include "Gauge_Util.hpp"
namespace klft {
struct JacobiSmearingParams {
  index_t n_steps;
  real_t kappa;

  JacobiSmearingParams() {
    n_steps = 9;
    kappa = 0.5;
  };
};

template <typename DGaugeFieldType, typename DSpinorFieldType>
struct JacobiSmearingFunctor {
  // implement the Wilson flow, for now the field will not be copied, but it
  // will be flown in place -> copying needs to be done before
  constexpr static const size_t rank =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank;
  constexpr static const size_t Nc =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Nc;
  constexpr static size_t RepDim =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::RepDim;
  constexpr static const GaugeFieldKind Kind =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Kind;
  using precision = DeviceGaugeFieldTypeTraits<DGaugeFieldType>::value_type;

  const JacobiSmearingParams params;

  // get the correct deviceGaugeFieldType
  using GaugeFieldT = typename DGaugeFieldType::type;
  using SpinorFieldT = typename DSpinorFieldType::type;
  const GaugeFieldT g_in_even;
  const GaugeFieldT g_in_odd;

  SpinorFieldT s_out_even;

  SpinorFieldT s_out_odd;
  SpinorFieldT s_in_even;
  SpinorFieldT s_in_odd;
  const IndexArray<rank> dims;

  class TagEven {};
  class TagOdd {};
  JacobiSmearingFunctor() = delete;

  JacobiSmearingFunctor(SpinorFieldT& s_out_even,
                        SpinorFieldT& s_out_odd,
                        const GaugeFieldT& g_in_even,
                        const GaugeFieldT& g_in_odd,
                        const SpinorFieldT& s_in_even,
                        const SpinorFieldT& s_in_odd,
                        const JacobiSmearingParams& _params,
                        const IndexArray<rank>& dims)
      : params(_params),
        s_out_even(s_out_even),
        s_out_odd(s_out_odd),
        s_in_even(s_in_even),
        s_in_odd(s_in_odd),
        g_in_even(g_in_even),
        g_in_odd(g_in_odd),
        dims(dims)

  {}

  // execute the wilson flow
  void smear() {  // todo: check this once by saving a staple field and once
                  // by locally calculating the staple

    KTune::parallel_for(
        "JacobiSmearingEven",
        Policy<rank, TagEven>(IndexArray<rank>{}, s_in_even.dimensions), *this);
    KTune::parallel_for(
        "JacobiSmearingOdd",
        Policy<rank, TagOdd>(IndexArray<rank>{}, s_in_odd.dimensions), *this);
    Kokkos::fence();
  }

  template <typename... Indices>
  KOKKOS_INLINE_FUNCTION void operator()(TagEven, const Indices... Idcs) const {
    // spatial directions only
    Kokkos::Array<size_t, rank> idx{Idcs...};
    auto full_idx = index_half_to_full(idx, 0);
    Spinor<Nc, RepDim, precision> temp{};
    for (index_t mu = 0; mu < rank - 1; ++mu) {
      auto xm =
          shift_index_minus_bc<rank, index_t>(full_idx, mu, 1, 3, -1, dims);
      auto xp =
          shift_index_plus_bc<rank, index_t>(full_idx, mu, 1, 3, -1, dims);
      temp += this->g_in_even(Idcs..., mu) *
              this->s_in_odd(index_full_to_half(xp.first).first);
      temp += conj(this->g_in_odd(index_full_to_half(xm.first).first, mu)) *
              this->s_in_odd(index_full_to_half(xm.first).first);
    }
    s_out_even(Idcs...) =
        1 / (1 + 6 * params.kappa) * (s_in_even(Idcs...) + params.kappa * temp);
  }
  template <typename... Indices>
  KOKKOS_INLINE_FUNCTION void operator()(TagOdd, const Indices... Idcs) const {
    // spatial directions only
    Kokkos::Array<size_t, rank> idx{Idcs...};
    auto full_idx = index_half_to_full(idx, 1);
    Spinor<Nc, RepDim, precision> temp{};
    for (index_t mu = 0; mu < rank - 1; ++mu) {
      auto xm =
          shift_index_minus_bc<rank, index_t>(full_idx, mu, 1, 3, -1, dims);
      auto xp =
          shift_index_plus_bc<rank, index_t>(full_idx, mu, 1, 3, -1, dims);
      temp += this->g_in_odd(Idcs..., mu) *
              this->s_in_even(index_full_to_half(xp.first).first);
      temp += conj(this->g_in_even(index_full_to_half(xm.first).first, mu)) *
              this->s_in_even(index_full_to_half(xm.first).first);
    }
    s_out_odd(Idcs...) =
        1 / (1 + 6 * params.kappa) * (s_in_odd(Idcs...) + params.kappa * temp);
  }
};
template <typename DGaugeFieldType, typename DSpinorFieldType>
void JacobiSmearing(const typename DGaugeFieldType::type& g_even,
                    const typename DGaugeFieldType::type& g_odd,
                    typename DSpinorFieldType::type s_in_even,
                    typename DSpinorFieldType::type s_in_odd,
                    typename DSpinorFieldType::type s_out_even,
                    typename DSpinorFieldType::type s_out_odd,
                    const JacobiSmearingParams& params) {
  constexpr static const size_t rank =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank;
  constexpr static const size_t Nc =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Nc;
  constexpr static const GaugeFieldKind Kind =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Kind;

  JacobiSmearingFunctor<DGaugeFieldType, DSpinorFieldType> f_in_to_out(
      s_out_even, s_out_odd, g_even, g_odd, s_in_even, s_in_odd, params,
      g_even.dimensions);
  f_in_to_out.smear();

  // Remaining steps: ping-pong between g_out and g_temp
  JacobiSmearingFunctor<DGaugeFieldType, DSpinorFieldType> f_out_to_inn(
      s_in_even, s_in_odd, g_even, g_odd, s_out_even, s_out_odd, params,
      g_even.dimensions);

  for (int i = 1; i < params.n_steps; i++) {
    if (i % 2 == 1)
      f_out_to_inn.smear();
    else
      f_in_to_out.smear();
  }

  if (params.n_steps > 0 && params.n_steps % 2 == 1) {
    Kokkos::deep_copy(s_in_even.field, s_out_even.field);
    Kokkos::deep_copy(s_in_odd.field, s_out_odd.field);
  }
}
template <typename DGaugeFieldType, typename DSpinorFieldType>
void JacobiSmearing(const typename DGaugeFieldType::type& g_in,
                    typename DSpinorFieldType::type s_in_even,
                    typename DSpinorFieldType::type s_in_odd,
                    typename DSpinorFieldType::type s_out_even,
                    typename DSpinorFieldType::type s_out_odd,
                    const JacobiSmearingParams& params) {
  constexpr static const size_t rank =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank;
  auto dims = s_in_even.dimensions;
  auto g_even = typename DGaugeFieldType::type(dims, complex_t(0.0));
  auto g_odd = typename DGaugeFieldType::type(dims, complex_t(0.0));

  alignGaugeFieldEvenOddFunctor<DGaugeFieldType> even(g_even, g_in, 0);
  alignGaugeFieldEvenOddFunctor<DGaugeFieldType> odd(g_odd, g_in, 1);
  KTune::parallel_for("init_evengaugefield",
                      Policy<rank>(IndexArray<rank>{}, g_even.dimensions),
                      even);
  KTune::parallel_for("init_oddgaugefield",
                      Policy<rank>(IndexArray<rank>{}, g_odd.dimensions), odd);
  JacobiSmearing<DGaugeFieldType, DSpinorFieldType>(
      g_even, g_odd, s_in_even, s_in_odd, s_out_even, s_out_odd, params);
}
template <typename DGaugeFieldType, typename DSpinorFieldType>
auto JacobiSmearing(const typename DGaugeFieldType::type& g_in,
                    typename DSpinorFieldType::type s_in_even,
                    const JacobiSmearingParams& params) {
  typename DSpinorFieldType::type s_in_odd(s_in_even.dimensions, complex_t(0));
  typename DSpinorFieldType::type s_out_even(s_in_even.dimensions,
                                             complex_t(0));
  typename DSpinorFieldType::type s_out_odd(s_in_even.dimensions, complex_t(0));
  JacobiSmearing<DGaugeFieldType, DSpinorFieldType>(
      g_in, s_in_even, s_in_odd, s_out_even, s_out_odd, params);
  return Kokkos::pair{s_out_even, s_out_odd};
}
template <typename DGaugeFieldType, typename DSpinorFieldType>
auto JacobiSmearing(const typename DGaugeFieldType::type& g_even,
                    const typename DGaugeFieldType::type& g_odd,
                    typename DSpinorFieldType::type s_in_even,
                    const JacobiSmearingParams& params) {
  typename DSpinorFieldType::type s_in_odd(s_in_even.dimensions, complex_t(0));
  typename DSpinorFieldType::type s_out_even(s_in_even.dimensions,
                                             complex_t(0));
  typename DSpinorFieldType::type s_out_odd(s_in_even.dimensions, complex_t(0));
  JacobiSmearing<DGaugeFieldType, DSpinorFieldType>(
      g_even, g_odd, s_in_even, s_in_odd, s_out_even, s_out_odd, params);
  return Kokkos::pair{s_out_even, s_out_odd};
}
}  // namespace klft
