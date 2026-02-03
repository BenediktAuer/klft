#include <Kokkos_Complex.hpp>
#include <Kokkos_Core.hpp>

#include "GLOBAL.hpp"
// #include "FieldTypeHelper.hpp"
#include "../../include/DiracOperator.hpp"
#include "../../include/GammaMatrix.hpp"
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

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  int RETURNVALUE = 0;
  {
    constexpr int count = 500;
    setVerbosity(1);
    setTuning(1);
    printf("%i", KLFT_TUNING);
    printf("%i", KLFT_VERBOSITY);
    printf("\n=== Testing DiracOperator SU(3)  ===\n");
    printf("\n= Testing hermiticity =\n");
    index_t L0 = 32, L1 = 32, L2 = 32, L3 = 32;
    auto gammas = get_gammas<4>();
    GammaMat<4> gamma5 = get_gamma5();
    diracParams params(0.156);
    printf("Lattice Dimension %ix%ix%ix%i \n", L0, L1, L2, L3);
    printf("Generate SpinorFields...\n");

    Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/1234);
    deviceSpinorField<2, 4, complex_t> u(L0 / 2, L1, L2, L3, random_pool, 0,
                                         1.0 / 1.41);
    deviceSpinorField<2, 4, complex_t> Mu(L0, L1, L2, L3, 0);
    deviceSpinorField<2, 4, complex_t> temp(L0, L1, L2, L3, 0);

    printf("Generating Random Gauge Config\n");
    deviceGaugeField<4, 2> gauge(L0, L1, L2, L3, random_pool, 1);
    printf("Instantiate DiracOperator...\n");
    EOWilsonDiracOperator<
        DeviceSpinorFieldType<4, 2, 4, complex_t, SpinorFieldKind::Standard,
                              SpinorFieldLayout::Checkerboard>,
        DeviceGaugeFieldType<4, 2>>
        D(gauge, params);
    D.s_in_same_parity = u;

    printf("Apply DiracOperator...\n");
    DeviceSpinorFieldType<4, 2, 4, complex_t, SpinorFieldKind::Standard,
                          SpinorFieldLayout::Checkerboard>::type
        u_norm_out(L0 / 2, L1, L2, L3, 0);
    DeviceSpinorFieldType<4, 2, 4, complex_t, SpinorFieldKind::Standard,
                          SpinorFieldLayout::Checkerboard>::type
        u_axpy_out(L0 / 2, L1, L2, L3, 0);
    DeviceSpinorFieldType<4, 2, 4, complex_t, SpinorFieldKind::Standard,
                          SpinorFieldLayout::Checkerboard>::type
        u_axpy_out2(L0 / 2, L1, L2, L3, 0);
    printf("Launching Kernels for tuning...\n");
    D.template apply<Tags::TagSe>(u, u_norm_out);
    axpy<DeviceSpinorFieldType<4, 2, 4, complex_t>>(1, u_norm_out, u,
                                                    u_norm_out);
    printf("Tuning done, now timing...\n");
    Kokkos::Timer timer;
    real_t diracTime = std::numeric_limits<real_t>::max();
    for (size_t i = 0; i < count; i++) {
      D.template apply<Tags::TagDdaggerD>(u, u_norm_out);
    }
    auto diracTime1 = std::min(diracTime, timer.seconds());
    printf("Se Kernel Time:     %11.4e s\n", diracTime1 / count);
    printf("Se_normal total time: %11.4e s\n", diracTime1);
    timer.reset();
    for (size_t i = 0; i < count; i++) {
      D.template apply<Tags::TagHoe>(u, u_axpy_out);
      D.template apply<Tags::TagHeo>(u_axpy_out, u_axpy_out2);
      axpyG5<DeviceSpinorFieldType<4, 2, 4, complex_t>>(
          -params.kappa * params.kappa, u_axpy_out2, u, u_axpy_out);
      D.template apply<Tags::TagHoe>(u, u_axpy_out);
      D.template apply<Tags::TagHeo>(u_axpy_out, u_axpy_out2);
      axpyG5<DeviceSpinorFieldType<4, 2, 4, complex_t>>(
          -params.kappa * params.kappa, u_axpy_out2, u, u_axpy_out);
    }
    auto diracTime2 = std::min(diracTime, timer.seconds());
    printf("Se axpy Kernel Time:     %11.4e s\n", diracTime2 / count);
    printf("Se_ axpy total time: %11.4e s\n", diracTime2);
  }
  Kokkos::finalize();
  return RETURNVALUE;
}