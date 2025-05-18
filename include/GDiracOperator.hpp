#pragma once
#include "FieldTypeHelper.hpp"
#include "GammaMatrix.hpp"
#include "IndexHelper.hpp"
#include "Spinor.hpp"
namespace klft {

template <size_t rank, size_t Nc, size_t RepDim>
struct diracParameters {
  using VecGammaMatrix = Kokkos::Array<GammaMat<RepDim>, 4>;
  const VecGammaMatrix gammas;
  const GammaMat<RepDim> gamma_id = get_identity<RepDim>();
  const GammaMat<RepDim> gamma5;
  const real_t kappa;
  const IndexArray<rank> dimensions;
  diracParameters(const IndexArray<rank> _dimensions,
                  const VecGammaMatrix& _gammas,
                  const GammaMat<RepDim>& _gamma5, const real_t& _kappa)
      : dimensions(_dimensions),
        gammas(_gammas),
        gamma5(_gamma5),
        kappa(_kappa) {}
};

template <typename T>
struct BaseDiracOperator {
  // using GaugeFieldType = typename DeviceGaugeFieldType<rank, Nc>::type;
  // using VecGammaMatrix = Kokkos::Array<GammaMat<RepDim>, 4>;
  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(const Indices... Idcs) const {
    static_cast<const T*>(this)->operator()(Idcs...);  // delegate to derived
  }
};
// Define Global apply function

template <typename T, size_t rank, size_t Nc, size_t RepDim>
typename DeviceSpinorFieldType<rank, Nc, RepDim>::type applyD(
    const typename DeviceSpinorFieldType<rank, Nc, RepDim>::type& s_in,
    const typename DeviceGaugeFieldType<rank, Nc>::type& g_in,
    const diracParameters<rank, Nc, RepDim>& params) {
  typename DeviceSpinorFieldType<rank, Nc, RepDim>::type s_out(
      params.dimensions, complex_t(0.0, 0.0));
  IndexArray<rank> start{};
  T D(s_out, s_in, g_in, params);
  tune_and_launch_for<rank>("apply_Dirac_Operator", start, params.dimensions,
                            D);
  Kokkos::fence();
  return s_out;
}

template <size_t rank, size_t Nc, size_t RepDim>
struct WilsonDiracOperator
    : public BaseDiracOperator<WilsonDiracOperator<4, 3, 4>> {
  constexpr static const size_t Nd = rank;

  using SpinorFieldType =
      typename DeviceSpinorFieldType<rank, Nc, RepDim>::type;
  // Treated as read only but var. throughout a simulation
  const SpinorFieldType s_in;
  SpinorFieldType s_out;
  using GaugeFieldType = typename DeviceGaugeFieldType<rank, Nc>::type;
  const GaugeFieldType g_in;
  using VecGammaMatrix = Kokkos::Array<GammaMat<RepDim>, 4>;
  const VecGammaMatrix gammas;
  const GammaMat<RepDim> gamma_id = get_identity<RepDim>();
  const IndexArray<rank> dimensions;
  const real_t kappa;
  WilsonDiracOperator(SpinorFieldType& s_out, const SpinorFieldType& s_in,
                      const GaugeFieldType& g_in,
                      const diracParameters<rank, Nc, RepDim>& params)
      : s_out(s_out),
        s_in(s_in),
        g_in(g_in),
        gammas(params.gammas),

        dimensions(params.dimensions),
        kappa(params.kappa) {}

  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(const Indices... Idcs) const {
    Spinor<Nc, RepDim> temp = zeroSpinor<Nc, RepDim>();
#pragma unroll
    for (size_t mu = 0; mu < rank; ++mu) {
      auto xm = shift_index_minus<rank, size_t>(
          Kokkos::Array<size_t, rank>{Idcs...}, mu, 1, dimensions);
      auto xp = shift_index_plus<rank, size_t>(
          Kokkos::Array<size_t, rank>{Idcs...}, mu, 1, dimensions);

      temp += (gamma_id - gammas[mu]) * 0.5 * (g_in(Idcs..., mu) * s_in(xp));
      temp += (gamma_id + gammas[mu]) * 0.5 * (conj(g_in(xm, mu)) * s_in(xm));
    }
    // Is the +4 correct? Instead of += only = depending on how s_out is
    // initialized or used!
    s_out(Idcs...) += s_in(Idcs...) - kappa * temp;
  }
};
}  // namespace klft
