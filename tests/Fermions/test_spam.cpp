#include <Kokkos_Complex.hpp>
#include <Kokkos_Core.hpp>

#include "GLOBAL.hpp"
// #include "FieldTypeHelper.hpp"
#include <stdio.h>
#include "../../include/DiracOperator.hpp"
#include "../../include/FermionMonomial.hpp"
#include "../../include/GammaMatrix.hpp"
#include "../../include/Solvercopy.hpp"
#include "../../include/Spinor.hpp"
#include "../../include/SpinorField.hpp"
#include "../../include/SpinorFieldLinAlg.hpp"
#include "../../include/WilsonDiracOperator.hpp"
#include "../../include/klft.hpp"
#include "IndexHelper.hpp"
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
    IndexArray<4> test{0, 0, 0, 0};

    hop<4, index_t, 0, 1>(test, 3);
    printf("%i,%i,%i,%i\n", test[0], test[1], test[2], test[3]);
    hop<4, index_t, 0, 1>(test, 3);
    printf("%i,%i,%i,%i\n", test[0], test[1], test[2], test[3]);

    hop<4, index_t, 0, 1>(test, 3);
    printf("%i,%i,%i,%i\n", test[0], test[1], test[2], test[3]);
    hop<4, index_t, 0, -1>(test, 3);
    printf("%i,%i,%i,%i\n", test[0], test[1], test[2], test[3]);
  }
  Kokkos::finalize();
  return RETURNVALUE;
}