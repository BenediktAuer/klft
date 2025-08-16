#include <assert.h>

#include <iomanip>

#include "AdjointSUN.hpp"
#include "GLOBAL.hpp"
#include "IndexHelper.hpp"
#include "SUN.hpp"
#include "Spinor.hpp"
using namespace klft;

#define HLINE "=========================================================\n"

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/1234);
    auto rng = random_pool.get_state();
    SUN<3> a, U;
    randSUN(a, rng, 0.1);
    random_pool.free_state(rng);
    /*
  U.c00 = +0.3391 -0.1635*I;
  U.c01 = -0.2357 +0.5203*I;
  U.c02 = +0.5609 +0.4663*I;
  U.c10 = -0.0740 -0.4204*I;
  U.c11 = -0.7706 -0.1863*I;
  U.c12 = +0.1191 -0.4185*I;
  U.c20 = +0.5351 -0.6243*I;
  U.c21 = +0.1825 +0.1089*I;
  U.c22 = -0.5279 -0.0022*I;
    */
    // a[0][0] = -0.2994;
    // a[0][1] = complex_t(0.5952, 1.3123);
    // a[0][2] = complex_t(-0.7943, 0.0913);
    // a[1][1] = -1.1430;
    // a[1][2] = complex_t(-2.0025, 0.2978);
    // a[2][2] = 1.4424;
    // a[1][0] = Kokkos::conj(a[0][1]);
    // a[2][0] = Kokkos::conj(a[0][2]);
    // a[2][1] = Kokkos::conj(a[1][2]);

    // U[0][0] = complex_t(0.3391, -0.1635);
    // U[0][1] = complex_t(-0.2357, 0.5203);
    // U[0][2] = complex_t(0.5609, 0.4663);
    // U[1][0] = complex_t(-0.0740, -0.4204);
    // U[1][1] = complex_t(-0.7706, -0.1863);
    // U[1][2] = complex_t(0.1191, -0.4185);
    // U[2][0] = complex_t(0.5351, -0.6243);
    // U[2][1] = complex_t(0.1825, 0.1089);
    // U[2][2] = complex_t(-0.5279, -0.0022);

    print_SUN(a, "a");
    // print_SUN(U, "U");
    // printf("%f", trace(traceLessAntiHermitian(a);
    // printf("Trace: %f", trace(a));
    auto adj = traceT((a));
    printf("iamg_det = %f \n", imag_det_SU3(adj));
    print_SUNAdj(adj, "Adjoint of a");

    auto a_from_adj = expoSUN(adj);
    // printf("%f, %fi\n ", det(a_from_adj).real(), det(a_from_adj).imag());
    print_SUN(a_from_adj, "");
  }
  Kokkos::finalize();
  return 0;
}