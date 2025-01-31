#pragma once
#include "GLOBAL.hpp"
#include "HamiltonianField.hpp"

namespace klft {

  typedef enum MonomialType_s {
    GAUGE_MONOMIAL = 0,
    FERMION_MONOMIAL
  } MonomialType;

  template <typename T, class Group, class Adjoint, int Ndim = 4, int Nc = 3>
  class Monomial {
    public:
      MonomialType monomial_type;
      T H_old, H_new;

      Monomial() : H_old(0.0), H_new(0.0) {}

      Monomial(unsigned int _time_scale) : H_old(0.0), H_new(0.0) {}

      virtual MonomialType get_monomial_type() {
        return monomial_type;
      }
      virtual void heatbath(HamiltonianField<T,Group,Adjoint,Ndim,Nc> h) = 0;
      virtual void accept(HamiltonianField<T,Group,Adjoint,Ndim,Nc> h) = 0;
      virtual void derivative(AdjointField<T,Adjoint,Ndim,Nc> deriv, HamiltonianField<T,Group,Adjoint,Ndim,Nc> h) = 0;

      void reset() {
        H_old = 0.0;
        H_new = 0.0;
      }

      // void set_time_scale(const T &_time_scale) {
      //   time_scale = _time_scale;
      // }

      // T get_time_scale() {
      //   return time_scale;
      // }

      T get_delta_H() {
        return H_new - H_old;
      }
  };

  template <typename T, class Group, class Adjoint, int Ndim = 4, int Nc = 3>
  class KineticMonomial : public Monomial<T,Group,Adjoint,Ndim,Nc> {
  public:
    KineticMonomial(unsigned int _time_scale) : Monomial<T,Group,Adjoint,Ndim,Nc>(_time_scale) {}
    void heatbath(HamiltonianField<T,Group,Adjoint,Ndim,Nc> h) override {
      Monomial<T,Group,Adjoint,Ndim,Nc>::H_old = h.kinetic_energy();
    }
    void accept(HamiltonianField<T,Group,Adjoint,Ndim,Nc> h) override {
      Monomial<T,Group,Adjoint,Ndim,Nc>::H_new = h.kinetic_energy();
    }
    void derivative(AdjointField<T,Adjoint,Ndim,Nc> deriv, HamiltonianField<T,Group,Adjoint,Ndim,Nc> h) override {}
  };

} // namespace klft