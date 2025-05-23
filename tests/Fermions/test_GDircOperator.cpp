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
    setVerbosity(5);
    printf("%i", KLFT_VERBOSITY);
    printf("\n=== Testing DiracOperator SU(3)  ===\n");
    printf("\n= Testing hermiticity =\n");
    index_t L0 = 32, L1 = 32, L2 = 32, L3 = 32;
    auto gammas = get_gammas<4>();
    GammaMat<4> gamma5 = get_gamma5();
    printf("Lattice Dimension %ix%ix%ix%i \n", L0, L1, L2, L3);
    printf("Generate SpinorFields...\n");

    Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/1234);
    deviceSpinorField<3, 4> u(L0, L1, L2, L3, random_pool, 0, 1.0 / 1.41);
    deviceSpinorField<3, 4> v(L0, L1, L2, L3, random_pool, 0, 1.0 / 1.41);
    real_t norm = spinor_norm<4, 3, 4>(u);
    norm *= spinor_norm<4, 3, 4>(v);
    norm = Kokkos::sqrt(norm);

    printf("Generating Random Gauge Config\n");
    deviceGaugeField<4, 3> gauge(L0, L1, L2, L3, random_pool, 1);

    printf("Generate DiracOperator...\n");
    WilsonDiracOperator D() deviceSpinorField<3, 4> Mu =
        apply_HD<4, 3, 4>(u, gauge, gammas, gamma5, -0.5);
    deviceSpinorField<3, 4> Mv =
        apply_HD<4, 3, 4>(v, gauge, gammas, gamma5, -0.5);
    // deviceSpinorField<3, 4> Mu = apply_D<4, 3, 4>(u, gauge, gammas, -0.5);
    // deviceSpinorField<3, 4> Mv = apply_D<4, 3, 4>(v, gauge, gammas, -0.5);
  }
