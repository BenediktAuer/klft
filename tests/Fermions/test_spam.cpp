#include <Kokkos_Complex.hpp>
#include <Kokkos_Core.hpp>

#include "GLOBAL.hpp"
// #include "FieldTypeHelper.hpp"
#include "../../include/DiracOperator.hpp"
#include "../../include/FermionMonomial.hpp"
#include "../../include/GammaMatrix.hpp"
#include "../../include/Solvercopy.hpp"
#include "../../include/Spinor.hpp"
#include "../../include/SpinorField.hpp"
#include "../../include/SpinorFieldLinAlg.hpp"
#include "../../include/WilsonDiracOperator.hpp"
#include "../../include/klft.hpp"
#define HLINE "=========================================================\n"

using namespace klft;
template <size_t Nc, size_t Nd>
void print_spinor(const Spinor<Nc, Nd>& s, const char* name = "Spinor") {
  printf("%s:\n", name);
  for (size_t c = 0; c < Nc; ++c) {
    printf("  Color %zu:\n", c);
    for (size_t d = 0; d < Nd; ++d) {
      double re = s[c][d].real();
      double im = s[c][d].imag();
      printf("    [%zu] = (% .6f, % .6f i)\n", d, re, im);
    }
  }
}
template <typename T, typename precision>
struct getSpinorField {
  using type = DeviceSpinorFieldType<DeviceFermionFieldTypeTraits<T>::Rank,
                                     DeviceFermionFieldTypeTraits<T>::Nc,
                                     DeviceFermionFieldTypeTraits<T>::RepDim,
                                     precision,
                                     DeviceFermionFieldTypeTraits<T>::Kind,
                                     DeviceFermionFieldTypeTraits<T>::Layout>;
};
int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  KTune::initialize();
  int RETURNVALUE = 0;
  {
    using DSpinorFieldType =
        DeviceSpinorFieldType<4, 3, 4, complex_t, SpinorFieldKind::Standard,
                              SpinorFieldLayout::Checkerboard>;
    getSpinorField<DSpinorFieldType, complexsingle_t>::type::type even_true(
        2 / 2, 2, 2, 2, 1);
    deviceGaugeField<4, 3> gauge(2, 2, 2, 2, 1);
    diracParams params(0.15);
    EOWilsonDiracOperator<DSpinorFieldType,
                          DeviceGaugeFieldType<4, 3, complex_t>>
        D_pre(gauge, params);
    EOWilsonDiracOperator<DSpinorFieldType,
                          DeviceGaugeFieldType<4, 3, complex_t>>::
        rebind<getSpinorField<DSpinorFieldType, complexsingle_t>::type>
            D2_pre(gauge, params);
  }
  Kokkos::finalize();
  return RETURNVALUE;
}