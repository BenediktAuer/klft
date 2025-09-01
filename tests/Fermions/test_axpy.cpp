#include <Kokkos_Complex.hpp>
#include <Kokkos_Core.hpp>

#include "GLOBAL.hpp"
// #include "FieldTypeHelper.hpp"
#include "../../include/DiracOperator.hpp"
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
    constexpr int count = 500;
    setVerbosity(5);
    printf("%i", KLFT_VERBOSITY);
    printf("\n=== Testing DiracOperator SU(3)  ===\n");

    index_t L0 = 32, L1 = 32, L2 = 32, L3 = 32;
    printf("Lattice Dimension %ix%ix%ix%i \n", L0, L1, L2, L3);

    auto gammas = get_gammas<4>();
    GammaMat<4> gamma5 = get_gamma5();

    printf("Generate SpinorFields...\n");

    Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/1234);
    deviceSpinorField<2, 4> a(L0, L1, L2, L3, random_pool, 0, 1.0 / 1.41);
    deviceSpinorField<2, 4> b(L0, L1, L2, L3, 0);
    deviceSpinorField<2, 4> res(L0, L1, L2, L3, 0);

    Kokkos::Timer timer;

    real_t diracTime = std::numeric_limits<real_t>::max();
    for (size_t i = 0; i < count; i++) {
      axpy<4, 2, 4>(2.0, a, b, res);
    }
    auto diracTime1 = std::min(diracTime, timer.seconds());
    printf("Axpy Time:     %11.4e s\n", diracTime1 / count);
  }
  Kokkos::finalize();
  return RETURNVALUE;
}