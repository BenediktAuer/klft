#pragma once
#include "DiracOperator.hpp"
namespace klft {

template <typename DSpinorFieldType,
          typename DGaugeFieldType,
          bool HasMassShift = false>
class FWilsonDiracOperator : public DiracOperator<FWilsonDiracOperator,
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

  ~FWilsonDiracOperator() = default;
  using Base = DiracOperator<FWilsonDiracOperator,
                             DSpinorFieldType,
                             DGaugeFieldType,
                             HasMassShift>;
  using Base::Base;
  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(typename Tags::TagD,
                                              const Indices... Idcs) const {
    Spinor<Nc, RepDim> temp;
    Kokkos::Array<size_t, rank> idx{Idcs...};
    // mu =0, +1 direction (x)
    {
      hop<rank, size_t, 0, 1>(idx, this->s_in.dimensions[0]);
      wilson_hop<Nc, RepDim, 0, 1>(temp, this->g_in(Idcs..., 0),
                                   this->s_in(idx));
      hop<rank, size_t, 0, -1>(idx, this->s_in.dimensions[0]);
    }
    // mu =0, -1 direction (x)
    {
      hop<rank, size_t, 0, -1>(idx, this->s_in.dimensions[0]);
      wilson_hop<Nc, RepDim, 0, -1>(temp, conj(this->g_in(idx, 0)),
                                    this->s_in(idx));
      hop<rank, size_t, 0, 1>(idx, this->s_in.dimensions[0]);
    }

    {
      hop<rank, size_t, 1, 1>(idx, this->s_in.dimensions[1]);
      wilson_hop<Nc, RepDim, 1, 1>(temp, this->g_in(Idcs..., 1),
                                   this->s_in(idx));
      hop<rank, size_t, 1, -1>(idx, this->s_in.dimensions[1]);
    }
    // mu =1, -1 direction (x)
    {
      hop<rank, size_t, 1, -1>(idx, this->s_in.dimensions[1]);
      wilson_hop<Nc, RepDim, 1, -1>(temp, conj(this->g_in(idx, 1)),
                                    this->s_in(idx));
      hop<rank, size_t, 1, 1>(idx, this->s_in.dimensions[1]);
    }
    // mu =2, 1 direction (x)
    {
      hop<rank, size_t, 2, 1>(idx, this->s_in.dimensions[2]);
      wilson_hop<Nc, RepDim, 2, 1>(temp, this->g_in(Idcs..., 2),
                                   this->s_in(idx));
      hop<rank, size_t, 2, -1>(idx, this->s_in.dimensions[2]);
    }
    // mu =2, -1 direction (x)
    {
      hop<rank, size_t, 2, -1>(idx, this->s_in.dimensions[2]);
      wilson_hop<Nc, RepDim, 2, -1>(temp, conj(this->g_in(idx, 2)),
                                    this->s_in(idx));
      hop<rank, size_t, 2, 1>(idx, this->s_in.dimensions[2]);
    }
    // mu =3, -1 direction (x)
    real_t bc = 0;
    {
      hop_temp<rank, size_t, 3, 1>(idx, this->s_in.dimensions[3], bc);
      wilson_hop<Nc, RepDim, 3, 1>(temp, bc * this->g_in(Idcs..., 3),
                                   this->s_in(idx));
      hop_temp<rank, size_t, 3, -1>(idx, this->s_in.dimensions[3], bc);
    }
    // mu =3, -1 direction (x)
    {
      hop_temp<rank, size_t, 3, -1>(idx, this->s_in.dimensions[3], bc);
      wilson_hop<Nc, RepDim, 3, -1>(temp, bc * conj(this->g_in(idx, 3)),
                                    this->s_in(idx));
      hop_temp<rank, size_t, 3, 1>(idx, this->s_in.dimensions[3], bc);
    }
    this->s_out(Idcs...) = this->s_in(Idcs...) - (this->params.kappa * temp);
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
}  // namespace klft