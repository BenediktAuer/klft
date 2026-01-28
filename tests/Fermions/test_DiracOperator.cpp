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
#include "FasterWDirac.hpp"
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
    deviceSpinorField<2, 4> u(L0, L1, L2, L3, random_pool, 0, 1.0 / 1.41);
    deviceSpinorField<2, 4> Mu(L0, L1, L2, L3, 0);
    deviceSpinorField<2, 4> temp(L0, L1, L2, L3, 0);

    printf("Generating Random Gauge Config\n");
    deviceGaugeField<4, 2> gauge(L0, L1, L2, L3, random_pool, 1);
    printf("Instantiate DiracOperator...\n");
    WilsonDiracOperator<DeviceSpinorFieldType<4, 2, 4>,
                        DeviceGaugeFieldType<4, 2>>
        D(gauge, params);
    FWilsonDiracOperator<DeviceSpinorFieldType<4, 2, 4>,
                         DeviceGaugeFieldType<4, 2>>
        Df(gauge, params);

    printf("Apply DiracOperator...\n");
    printf("Launching Kernels for tuning...\n");
    D.template apply<Tags::TagD>(u, Mu);
    Df.template apply<Tags::TagD>(u, Mu);

    printf("Tuning done, now timing...\n");
    Kokkos::Timer timer;
    real_t diracTime = std::numeric_limits<real_t>::max();
    for (size_t i = 0; i < count; i++) {
      D.template apply<Tags::TagD>(u, Mu);
    }
    auto diracTime1 = std::min(diracTime, timer.seconds());
    printf("D Kernel Time:     %11.4e s\n", diracTime1 / count);
    printf("D total time: %11.4e s\n", diracTime1);
    timer.reset();
    for (size_t i = 0; i < count; i++) {
      Df.template apply<Tags::TagD>(u, Mu);
    }
    real_t diracTime2 = std::numeric_limits<real_t>::max();
    auto diracTime3 = std::min(diracTime2, timer.seconds());
    printf("Df Kernel Time:     %11.4e s\n", diracTime3 / count);
    printf("Df total time: %11.4e s\n", diracTime3);
  }
  Kokkos::finalize();
  return RETURNVALUE;
}