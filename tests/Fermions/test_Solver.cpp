#include <getopt.h>

#include <Kokkos_Complex.hpp>
#include <Kokkos_Core.hpp>
#include "../../include/GDiracOperator.hpp"
#include "../../include/GammaMatrix.hpp"
#include "../../include/Solver.hpp"
#include "../../include/SpinorField.hpp"
#include "GLOBAL.hpp"
// #include "../../include/SpinorFieldLinAlg.hpp"
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
      printf("    [%zu] = (% .20f, % .20f i)\n", d, re, im);
    }
  }
}
int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  int RETURNVALUE = 0;
  {
    // timer
    Kokkos::Timer timer;

    real_t diracTime = std::numeric_limits<real_t>::max();
    setVerbosity(1);
    printf("%i", KLFT_VERBOSITY);
    printf("\n=== Testing DiracOperator SU(3)  ===\n");
    printf("\n= Testing hermiticity =\n");
    index_t L0 = 32, L1 = 32, L2 = 32, L3 = 32;
    auto gammas = get_gammas<4>();
    GammaMat<4> gamma5 = get_gamma5();
    IndexArray<4> dims = {L0, L1, L2, L3};
    diracParameters<4, 3, 4> param(dims, gammas, gamma5, 0.1);

    printf("Lattice Dimension %ix%ix%ix%i \n", L0, L1, L2, L3);
    printf("Generate SpinorFields...\n");

    Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/1234);
    deviceSpinorField<3, 4> u(L0, L1, L2, L3, random_pool, 0, 1.0 / 1.41);
    deviceSpinorField<3, 4> x(L0, L1, L2, L3, complex_t(0.0, 0.0));
    deviceSpinorField<3, 4> x0(L0, L1, L2, L3, complex_t(0.0, 0.0));
    deviceGaugeField<4, 3> gauge(L0, L1, L2, L3, random_pool, 1);
    printf("Instantiate DiracOperator...\n");
    HWilsonDiracOperator D(gauge, param);
    printf("Apply dirac Operator...\n");
    print_spinor(u(0, 0, 0, 0));
    timer.reset();
    auto test = D.applyDdagger(D.applyD(u));
    auto diracTime1 = std::min(diracTime, timer.seconds());
    printf("QQ^\\dagger Kernel Time:     %11.4e s\n", diracTime1);
    printf("Initialize Solver...\n");
    CGSolver solver(test, x, D);
    printf("Apply Solver...\n");
    auto eps = 1e-13;
    timer.reset();

    solver.solve(x0, eps);
    auto diracTime2 = std::min(diracTime, timer.seconds());
    printf("Solver Kernel Time:     %11.4e s\n", diracTime2);
    printf("Comparing Solver result to expected result...\n");
    print_spinor<3, 4>(solver.x(0, 0, 0, 0) - u(0, 0, 0, 0), "Solver Result");
    auto res_norm =
        spinor_norm<4, 3, 4>(spinor_sub_mul<4, 3, 4>(u, solver.x, 1));
    auto norm = spinor_norm<4, 3, 4>(u);

    printf("Norm of Residual: %.20f\n", res_norm / norm);
    printf("Is the residual norm smaller than %.2e ? %i\n", eps,
           res_norm / norm < eps);
    // printf("i%i\n", solver.x(0, 0, 0, 0) == u(0, 0, 0, 0));
  }

  Kokkos::finalize();
  printf(HLINE);
  printf("%i Errors durring Testing\n", RETURNVALUE);
  printf(HLINE);
  RETURNVALUE = !(RETURNVALUE == 0);
  return RETURNVALUE;
}