#pragma once
#include "FermionParams.hpp"
#include "FieldTypeHelper.hpp"
#include "GammaMatrix.hpp"
#include "IndexHelper.hpp"
#include "Spinor.hpp"
#include "SpinorFieldLinAlg.hpp"

namespace klft {
// Base Class for Single Dirac Operator
namespace Tags {
struct TagD {};
struct TagDdagger {};
// applys Composid Operator M= DDdagger*s_in
struct TagDDdagger {};
// applys Composid Operator Mdagger= DdaggerD*s_in
// If spinorfield are Checkerboard the operator S_e is used
struct TagDdaggerD {};
struct TagHoe {};
struct TagHeo {};
struct TagSe {};
struct TagSo {};
struct TagG5Se {};
struct TagG5So {};
}  // namespace Tags
template <class _Derived,
          typename _DSpinorFieldType,
          typename _DGaugeFieldType,
          bool _HasMassshift = false>
class BaseDiracOperator {
  static_assert(isDeviceGaugeFieldType<_DGaugeFieldType>::value);
  static_assert(isDeviceFermionFieldType<_DSpinorFieldType>::value);
  constexpr static size_t rank =
      DeviceFermionFieldTypeTraits<_DSpinorFieldType>::Rank;
  constexpr static size_t Nc =
      DeviceFermionFieldTypeTraits<_DSpinorFieldType>::Nc;
  constexpr static size_t RepDim =
      DeviceFermionFieldTypeTraits<_DSpinorFieldType>::RepDim;
  static_assert((rank == DeviceGaugeFieldTypeTraits<_DGaugeFieldType>::Rank) &&
                (Nc == DeviceGaugeFieldTypeTraits<_DGaugeFieldType>::Nc));
  constexpr static SpinorFieldLayout Layout =
      DeviceFermionFieldTypeTraits<_DSpinorFieldType>::Layout;

  // using Derived = _Derived<_Derived, _DSpinorFieldType, DGaugeFieldType>;
  // Define Tags for template dispatch:
 public:
  constexpr static bool HasMassshift = _HasMassshift;
  using DSpinorFieldType = _DSpinorFieldType;
  using DGaugeFieldType = _DGaugeFieldType;
  using SpinorFieldType = typename _DSpinorFieldType::type;
  using GaugeFieldType = typename DeviceGaugeFieldType<rank, Nc>::type;

  BaseDiracOperator(const GaugeFieldType& g_in, const diracParams& params)
      : g_in(g_in), params(params) {}
  ~BaseDiracOperator() = default;
  // Define callabale apply functions
  template <typename Tag>
  KOKKOS_FORCEINLINE_FUNCTION SpinorFieldType
  apply(const SpinorFieldType& s_in) {
    this->s_in = s_in;
    this->s_out = SpinorFieldType(s_in.dimensions, complex_t(0.0, 0.0));
    // Apply the operator

    return static_cast<_Derived&>(*this).apply_(Tag{});
  }

  /// @brief applys the DiracOperator to the field s_in and stores the result in
  /// s_out
  /// @tparam Tag Can either be TagD or TagDdagger or TagDDdagger or TagDdaggerD
  /// @param s_in
  /// @param s_out
  /// @return Nothing, result is stored in s_out
  template <typename Tag>
  KOKKOS_FORCEINLINE_FUNCTION void apply(const SpinorFieldType& s_in,
                                         const SpinorFieldType& s_out) {
    this->s_in = s_in;

    this->s_out = s_out;

    // forward to Derived class only for TagHeo
    static_cast<_Derived&>(*this).apply_(Tag{});
  }

  // Special overload t reduce allocations in solver further, only works with
  // composed operators

  template <typename Tag>
  KOKKOS_FORCEINLINE_FUNCTION void apply(const SpinorFieldType& s_in,
                                         const SpinorFieldType& s_temp,
                                         const SpinorFieldType& s_out) {
    this->s_in = s_in;
    this->s_out = s_temp;

    // Apply the operator
    static_cast<_Derived&>(*this).apply_(Tag{}, s_out);
  }

 public:
  SpinorFieldType s_in;
  SpinorFieldType s_out;
  const GaugeFieldType g_in;
  const diracParams params;

 protected:
  BaseDiracOperator() = default;
};
template <template <typename, typename, bool> class _Derived,
          typename DSpinorFieldType,
          typename DGaugeFieldType,
          bool HasMassShift = false>
class DiracOperator : public BaseDiracOperator<DiracOperator<_Derived,
                                                             DSpinorFieldType,
                                                             DGaugeFieldType,
                                                             HasMassShift>,
                                               DSpinorFieldType,
                                               DGaugeFieldType,
                                               HasMassShift> {
  static_assert(isDeviceGaugeFieldType<DGaugeFieldType>::value);
  static_assert(isDeviceFermionFieldType<DSpinorFieldType>::value);
  constexpr static size_t rank =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank;
  constexpr static size_t Nc =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Nc;
  constexpr static size_t RepDim =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::RepDim;
  static_assert((rank == DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank) &&
                (Nc == DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Nc));
  constexpr static SpinorFieldLayout Layout =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Layout;

  using BaseDiracOperator<
      DiracOperator<_Derived, DSpinorFieldType, DGaugeFieldType, HasMassShift>,
      DSpinorFieldType,
      DGaugeFieldType,
      HasMassShift>::BaseDiracOperator;
  using Derived = _Derived<DSpinorFieldType, DGaugeFieldType, HasMassShift>;
  using SpinorFieldType = typename DSpinorFieldType::type;

 public:
  SpinorFieldType apply_(Tags::TagD) {
    // Apply the operator
    tune_and_launch_for<rank, Tags::TagD>(
        typeid(Derived).name(), IndexArray<rank>{}, this->s_in.dimensions,
        static_cast<Derived&>(*this));
    Kokkos::fence();
    return this->s_out;
  }

  SpinorFieldType apply_(Tags::TagDdagger) {
    // Apply the operator
    tune_and_launch_for<rank, Tags::TagDdagger>(
        typeid(Derived).name(), IndexArray<rank>{}, this->s_in.dimensions,
        static_cast<Derived&>(*this));
    Kokkos::fence();
    return this->s_out;
  }
  // applys Composid Operator M= DDdagger*s_in

  SpinorFieldType apply_(Tags::TagDDdagger) {
    auto cached_out = this->s_out;
    this->s_out = SpinorFieldType(this->s_in.dimensions, complex_t(0.0, 0.0));

    return this->apply_(Tags::TagDDdagger{}, cached_out);
  }

  // applys Composid Operator Mdagger= DdaggerD*s_in
  SpinorFieldType apply_(Tags::TagDdaggerD) {
    auto cached_out = this->s_out;
    this->s_out = SpinorFieldType(this->s_in.dimensions, complex_t(0.0, 0.0));

    return this->apply_(Tags::TagDdaggerD{}, cached_out);
  }

  SpinorFieldType apply_(Tags::TagDDdagger, const SpinorFieldType& s_out) {
    if constexpr (Layout == SpinorFieldLayout::Checkerboard) {
      // printf("Iam a CheckerboardLayout\n");
      static_cast<Derived&>(*this).apply_(Tags::TagDDdagger{}, s_out);
      // auto temp_field = this->s_in;
      // apply<Tags::TagHoe>(this->s_in, this->s_out);
      // this->s_in = this->s_out;
      // this->s_out = s_out;
      // apply<Tags::TagHeo>(this->s_in, this->s_out);
      // axpy<DSpinorFieldType>(-1, this->s_out, temp_field, this->s_out);
      return this->s_out;
    }

    this->apply_(Tags::TagDdagger{});
    this->s_in = this->s_out;
    this->s_out = s_out;
    return this->apply_(Tags::TagD{});
  }

  SpinorFieldType apply_(Tags::TagDdaggerD, const SpinorFieldType& s_out) {
    if constexpr (Layout == SpinorFieldLayout::Checkerboard) {
      static_cast<Derived&>(*this).apply_(Tags::TagDdaggerD{}, s_out);
      // printf("Iam a CheckerboardLayout\n");
      // auto temp_field = this->s_in;
      // this->apply_(Tags::TagD{});
      // this->s_in = this->s_out;
      // this->s_out = s_out;
      // this->apply_(Tags::TagDdagger{});
      // axpy<DSpinorFieldType>(-1, this->s_out, temp_field, this->s_out);
      return this->s_out;
    }
    this->apply_(Tags::TagD{});
    this->s_in = this->s_out;
    this->s_out = s_out;
    return this->apply_(Tags::TagDdagger{});
  }
};

template <template <typename, typename, bool> class _Derived,
          typename DSpinorFieldType,
          typename DGaugeFieldType,
          bool HasMassShift = false>
class EODiracOperator
    : public BaseDiracOperator<EODiracOperator<_Derived,
                                               DSpinorFieldType,
                                               DGaugeFieldType,
                                               HasMassShift>,
                               DSpinorFieldType,
                               DGaugeFieldType,
                               HasMassShift> {
  static_assert(isDeviceGaugeFieldType<DGaugeFieldType>::value);
  static_assert(isDeviceFermionFieldType<DSpinorFieldType>::value);
  constexpr static size_t rank =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank;
  constexpr static size_t Nc =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Nc;
  constexpr static size_t RepDim =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::RepDim;
  static_assert((rank == DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank) &&
                (Nc == DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Nc));
  constexpr static SpinorFieldLayout Layout =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Layout;

 public:
  using BaseDiracOperator<EODiracOperator<_Derived,
                                          DSpinorFieldType,
                                          DGaugeFieldType,
                                          HasMassShift>,
                          DSpinorFieldType,
                          DGaugeFieldType,
                          HasMassShift>::BaseDiracOperator;
  using Derived = _Derived<DSpinorFieldType, DGaugeFieldType, HasMassShift>;
  using SpinorFieldType = typename DSpinorFieldType::type;

  SpinorFieldType s_in_same_parity;
  SpinorFieldType temp;
  struct Tag1minusHeo {};
  struct Tag1minusHoe {};
  SpinorFieldType apply_(Tags::TagDDdagger) {
    auto cached_out = this->s_out;
    this->s_out = SpinorFieldType(this->s_in.dimensions, complex_t(0.0, 0.0));

    return this->apply_(Tags::TagDDdagger{}, cached_out);
  }

  // applys Composid Operator Mdagger= DdaggerD*s_in
  SpinorFieldType apply_(Tags::TagDdaggerD) {
    auto cached_out = this->s_out;
    this->s_out = SpinorFieldType(this->s_in.dimensions, complex_t(0.0, 0.0));

    return this->apply_(Tags::TagDdaggerD{}, cached_out);
  }

  SpinorFieldType apply_(Tags::TagHeo) {
    // this->s_out = SpinorFieldType(this->this->s_in.dimensions, complex_t(0.0,
    // 0.0));
    tune_and_launch_for<rank, Tags::TagHeo>(
        typeid(Derived).name(), IndexArray<rank>{}, this->s_in.dimensions,
        static_cast<Derived&>(*this));
    return this->s_out;
  }
  SpinorFieldType apply_(Tags::TagHoe) {
    // this->s_out = SpinorFieldType(this->this->s_in.dimensions, complex_t(0.0,
    // 0.0));

    tune_and_launch_for<rank, Tags::TagHoe>(
        typeid(Derived).name(), IndexArray<rank>{}, this->s_in.dimensions,
        static_cast<Derived&>(*this));
    return this->s_out;
  }

  SpinorFieldType apply_(Tags::TagSe) {
    auto cached_out = this->s_out;
    this->s_out = SpinorFieldType(this->s_in.dimensions, complex_t(0.0, 0.0));
    return this->apply_(Tags::TagSe{}, cached_out);
  }

  SpinorFieldType apply_(Tags::TagSo) {
    auto cached_out = this->s_out;
    this->s_out = SpinorFieldType(this->s_in.dimensions, complex_t(0.0, 0.0));
    return this->apply_(Tags::TagSo{}, cached_out);
  }

  SpinorFieldType apply_(Tags::TagSe, const SpinorFieldType& s_out) {
    // printf("iam a EO Se\n");
    // printf("%i", this->test);
    this->apply_(Tags::TagHoe{});

    auto temp = this->s_in;
    this->s_in = this->s_out;
    this->s_out = s_out;

    tune_and_launch_for<rank, Tags::TagHeo>(
        typeid(Derived).name(), IndexArray<rank>{}, this->s_in.dimensions,
        static_cast<Derived&>(*this));
    axpy<DSpinorFieldType>(-this->params.kappa * this->params.kappa,
                           this->s_out, temp, this->s_out);
    return this->s_out;
  }
  SpinorFieldType apply_(Tags::TagSo, const SpinorFieldType& s_out) {
    this->apply_(Tags::TagHeo{});

    auto temp = this->s_in;
    this->s_in = this->s_out;
    this->s_out = s_out;

    tune_and_launch_for<rank, Tags::TagHoe>(
        typeid(Derived).name(), IndexArray<rank>{}, this->s_in.dimensions,
        static_cast<Derived&>(*this));
    axpy<DSpinorFieldType>(-this->params.kappa * this->params.kappa,
                           this->s_out, temp, this->s_out);
    return this->s_out;
  }

  SpinorFieldType apply_(Tags::TagG5Se) {
    auto cached_out = this->s_out;
    this->s_out = SpinorFieldType(this->s_in.dimensions, complex_t(0.0, 0.0));
    return this->apply_(Tags::TagG5Se{}, cached_out);
  }

  SpinorFieldType apply_(Tags::TagG5So) {
    auto cached_out = this->s_out;
    this->s_out = SpinorFieldType(this->s_in.dimensions, complex_t(0.0, 0.0));
    return this->apply_(Tags::TagG5So{}, cached_out);
  }

  SpinorFieldType apply_(Tags::TagG5Se, const SpinorFieldType& s_out) {
    // printf("iam a EO Se\n");
    // printf("%i", this->test);
    this->apply_(Tags::TagHoe{});

    auto temp = this->s_in;
    this->s_in = this->s_out;
    this->s_out = s_out;

    tune_and_launch_for<rank, Tags::TagHeo>(
        typeid(Derived).name(), IndexArray<rank>{}, this->s_in.dimensions,
        static_cast<Derived&>(*this));
    axpyG5<DSpinorFieldType>(-this->params.kappa * this->params.kappa,
                             this->s_out, temp, this->s_out);
    return this->s_out;
  }
  SpinorFieldType apply_(Tags::TagG5So, const SpinorFieldType& s_out) {
    this->apply_(Tags::TagHeo{});

    auto temp = this->s_in;
    this->s_in = this->s_out;
    this->s_out = s_out;

    tune_and_launch_for<rank, Tags::TagHoe>(
        typeid(Derived).name(), IndexArray<rank>{}, this->s_in.dimensions,
        static_cast<Derived&>(*this));
    axpyG5<DSpinorFieldType>(-this->params.kappa * this->params.kappa,
                             this->s_out, temp, this->s_out);
    return this->s_out;
  }

  SpinorFieldType apply_(Tags::TagDDdagger, const SpinorFieldType& s_out) {
    if (!temp.field.is_allocated()) {
      this->temp = SpinorFieldType(this->s_in.dimensions, 0);
    }

    auto cached_s_out = this->s_out;
    apply_(Tags::TagG5Se{}, this->temp);
    this->s_in = this->temp;
    this->s_out = cached_s_out;

    apply_(Tags::TagG5Se{}, s_out);
    return s_out;
  }
  SpinorFieldType apply_(Tags::TagDdaggerD, const SpinorFieldType& s_out) {
    apply_(Tags::TagDDdagger{}, s_out);
    return s_out;
  }
  SpinorFieldType apply_(Tags::TagD) {
    // Apply the operator
    tune_and_launch_for<rank, Tags::TagD>(
        typeid(Derived).name(), IndexArray<rank>{}, this->s_in.dimensions,
        static_cast<Derived&>(*this));
    Kokkos::fence();
    return this->s_out;
  }

  SpinorFieldType apply_(Tags::TagDdagger) {
    // Apply the operator
    tune_and_launch_for<rank, Tags::TagDdagger>(
        typeid(Derived).name(), IndexArray<rank>{}, this->s_in.dimensions,
        static_cast<Derived&>(*this));
    Kokkos::fence();
    return this->s_out;
  }
};
}  // namespace klft
