#pragma once
#include "DiracOperator.hpp"
namespace klft {
template <typename T>
struct DiracOpFieldTypeTraits;
template <template <typename, typename> class _Derived,
          typename _DSpinorFieldType, typename _DGaugeFieldType>
struct DiracOpFieldTypeTraits<
    DiracOperator<_Derived, _DSpinorFieldType, _DGaugeFieldType>> {
  using Derived = _Derived<_DSpinorFieldType, _DGaugeFieldType>;
  using DSpinorFieldType = _DSpinorFieldType;
  using DGaugeFieldType = _DGaugeFieldType;
};
template <typename _DSpinorFieldType, typename _DGaugeFieldType>
struct DiracOpFieldTypeTraits<
    WilsonDiracOperator<_DSpinorFieldType, _DGaugeFieldType>> {
  using Derived = WilsonDiracOperator<_DSpinorFieldType, _DGaugeFieldType>;
  using DSpinorFieldType = _DSpinorFieldType;
  using DGaugeFieldType = _DGaugeFieldType;
};
template <typename _DSpinorFieldType, typename _DGaugeFieldType>
struct DiracOpFieldTypeTraits<
    HWilsonDiracOperator<_DSpinorFieldType, _DGaugeFieldType>> {
  using Derived = HWilsonDiracOperator<_DSpinorFieldType, _DGaugeFieldType>;
  using DSpinorFieldType = _DSpinorFieldType;
  using DGaugeFieldType = _DGaugeFieldType;
};

}  // namespace klft
