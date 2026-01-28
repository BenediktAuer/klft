#pragma once
#include "DiracOperator.hpp"
namespace klft {

template <typename DSpinorFieldType,
          typename DGaugeFieldType,
          bool HasMassShift = false>
class WilsonDiracOperator : public DiracOperator<WilsonDiracOperator,
                                                 DSpinorFieldType,
                                                 DGaugeFieldType,
                                                 HasMassShift> {
 public:
  constexpr static size_t Nc =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Nc;
  constexpr static size_t RepDim =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::RepDim;
  constexpr static size_t rank =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank;

  ~WilsonDiracOperator() = default;
  using Base = DiracOperator<WilsonDiracOperator,
                             DSpinorFieldType,
                             DGaugeFieldType,
                             HasMassShift>;
  using Base::Base;
  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(typename Tags::TagD,
                                              const Indices... Idcs) const {
    Spinor<Nc, RepDim> temp;
    Kokkos::Array<size_t, rank> idx{Idcs...};
#pragma unroll
    for (size_t mu = 0; mu < rank; ++mu) {
      auto xm = shift_index_minus_bc<rank, size_t>(idx, mu, 1, 3, -1,
                                                   this->s_in.dimensions);
      auto xp = shift_index_plus_bc<rank, size_t>(idx, mu, 1, 3, -1,
                                                  this->s_in.dimensions);
      //   if (idx == Kokkos::Array<size_t, 4>({2, 2, 0, 0})) {
      //     printf("Normal D Op:\n");
      //     printf(
      //         "Local output index (full): [%i,%i,%i,%i];  Shifted Index in"
      //         "dir  mu -%i full: "
      //         "[%i,%i,%i,%i] , (half idx): "
      //         "[%i,%i,%i,%i] with parity %i\n",
      //         idx[0], idx[1], idx[2], idx[3], mu, xm.first[0], xm.first[1],
      //         xm.first[2], xm.first[3],
      //         index_full_to_half(xm.first).first[0],
      //         index_full_to_half(xm.first).first[1],
      //         index_full_to_half(xm.first).first[2],
      //         index_full_to_half(xm.first).first[3],
      //         index_full_to_half(xm.first).second);
      //   }
      auto temp1 =
          this->g_in(Idcs..., mu) * project(mu, -1, this->s_in(xp.first));

      auto temp2 =
          conj(this->g_in(xm.first, mu)) * project(mu, 1, this->s_in(xm.first));
      temp += reconstruct(mu, -1, (this->params.kappa * xp.second) * temp1) +
              reconstruct(mu, 1, (this->params.kappa * xm.second) * temp2);
    }

    this->s_out(Idcs...) = this->s_in(Idcs...) - temp;
    // this->s_out(Idcs...) = 1 * temp;
  }

  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(typename Tags::TagDdagger,
                                              const Indices... Idcs) const {
    Spinor<Nc, RepDim> temp;
    Kokkos::Array<size_t, rank> idx{Idcs...};

#pragma unroll
    for (size_t mu = 0; mu < rank; ++mu) {
      auto xm = shift_index_minus_bc<rank, size_t>(idx, mu, 1, 3, -1,
                                                   this->s_in.dimensions);
      auto xp = shift_index_plus_bc<rank, size_t>(idx, mu, 1, 3, -1,
                                                  this->s_in.dimensions);

      auto temp1 =
          this->g_in(Idcs..., mu) * project(mu, 1, this->s_in(xp.first));

      //
      auto temp2 = conj(this->g_in(xm.first, mu)) *
                   project(mu, -1, this->s_in(xm.first));
      temp += reconstruct(mu, 1, (this->params.kappa * xp.second) * temp1) +
              reconstruct(mu, -1, (this->params.kappa * xm.second) * temp2);
    }
    this->s_out(Idcs...) = this->s_in(Idcs...) - temp;
  }
};
