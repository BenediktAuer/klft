#include <Kokkos_Complex.hpp>
#include <Kokkos_Core.hpp>

#include "GLOBAL.hpp"
// #include "FieldTypeHelper.hpp"
#include "../../include/GDiracOperator.hpp"
#include "../../include/GammaMatrix.hpp"
#include "../../include/SpinorField.hpp"
#include "../../include/SpinorFieldLinAlg.hpp"
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
    constexpr int count = 100;
    setVerbosity(5);
    printf("%i", KLFT_VERBOSITY);
    printf("\n=== Testing DiracOperator SU(3)  ===\n");
    printf("\n= Testing hermiticity =\n");
    index_t L0 = 32, L1 = 32, L2 = 32, L3 = 32;
    auto gammas = get_gammas<4>();
    GammaMat<4> gamma5 = get_gamma5();
    diracParams<4, 4> params(IndexArray<4>{L0, L1, L2, L3}, gammas, gamma5,
                             -0.5);
    printf("Lattice Dimension %ix%ix%ix%i \n", L0, L1, L2, L3);
    printf("Generate SpinorFields...\n");

    Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/1234);
    deviceSpinorField<3, 4> u(L0, L1, L2, L3, random_pool, 0, 1.0 / 1.41);
    deviceSpinorFieldConst<3, 4> u_const(u);

    printf("Generating Random Gauge Config\n");
    deviceGaugeField<4, 3> gauge(L0, L1, L2, L3, random_pool, 1);
    printf("Instantiate DiracOperator...\n");
    HWilsonDiracOperator<DeviceSpinorFieldType<4, 3, 4>,
                         DeviceGaugeFieldType<4, 3>>
        D(gauge, params);
    printf("Apply DiracOperator...\n");
    Kokkos::Timer timer;

    real_t diracTime = std::numeric_limits<real_t>::max();
    for (size_t i = 0; i < count; i++) {
      deviceSpinorField Mu = D.template apply<Tags::TagD>(u);
    }
    auto diracTime1 = std::min(diracTime, timer.seconds());
    printf("D^ Kernel Time:     %11.4e s\n", diracTime1 / count);
  }
  Kokkos::finalize();
  return RETURNVALUE;
}