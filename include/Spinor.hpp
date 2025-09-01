//******************************************************************************/
//
// This file is part of the Kokkos Lattice Field Theory (KLFT) library.
//
// KLFT is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// KLFT is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with KLFT.  If not, see <http://www.gnu.org/licenses/>.
//
//******************************************************************************/

// define Spinor operations

#pragma once
#include "GLOBAL.hpp"
#include "GammaMatrix.hpp"

namespace klft {
template <size_t Nc, size_t Nd>
KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, Nd> operator*(
    const SUN<Nc>& U,
    const Spinor<Nc, Nd>& spinor) {
  Spinor<Nc, Nd> res{};
#pragma unroll
  for (size_t k = 0; k < Nd; k++) {
#pragma unroll
    for (size_t i = 0; i < Nc; i++) {
#pragma unroll
      for (size_t j = 0; j < Nc; j++) {
        res(k, i) += U[i][j] * spinor(k, j);
      }
    }
  }
  return res;
}
template <size_t Nc, size_t Nd>
KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, Nd> operator*(
    const Spinor<Nc, Nd>& spinor,
    const SUN<Nc>& U) {
  Spinor<Nc, Nd> res{};

#pragma unroll
  for (size_t k = 0; k < Nd; ++k) {
#pragma unroll
    for (size_t j = 0; j < Nc; ++j) {
#pragma unroll
      for (size_t i = 0; i < Nc; ++i) {
        res(k, i) += spinor(k, j) * U[j][i];
      }
    }
  }

  return res;
}

// *= makes no sense f spinor gauge link

template <size_t Nc, size_t Nd>
KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, Nd> operator*(
    const complex_t& scalar,
    const Spinor<Nc, Nd>& spinor) {
  Spinor<Nc, Nd> res;
#pragma unroll
  for (size_t j = 0; j < Nd; j++) {
#pragma unroll
    for (size_t i = 0; i < Nc; i++) {
      res(j, i) = spinor(j, i) * scalar;
    }
  }
  return res;
}
// this is for construction of the force matrix, no implicit conjugation,
// however this would be better for performance
template <size_t Nc, size_t Nd>
KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> operator*(const Spinor<Nc, Nd>& a,
                                              const Spinor<Nc, Nd>& b) {
  SUN<Nc> res{};
#pragma unroll
  for (size_t k = 0; k < Nd; ++k) {
#pragma unroll
    for (size_t i = 0; i < Nc; ++i) {
#pragma unroll
      for (size_t j = 0; j < Nc; ++j) {
        res[i][j] += a(k, i) * (b(k, j));
      }
    }
  }
  return res;
}

template <size_t Nc, size_t Nd>
KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, Nd> operator*=(Spinor<Nc, Nd>& spinor,
                                                      const complex_t& scalar) {
  Spinor<Nc, Nd> res = scalar * spinor;
  spinor = res;
  return spinor;
}

template <size_t Nc, size_t Nd>
KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, Nd> operator*(
    const real_t& scalar,
    const Spinor<Nc, Nd>& spinor) {
  Spinor<Nc, Nd> res;

#pragma unroll
  for (size_t i = 0; i < Nc * Nd; i++) {
    res(i) = scalar * spinor(i);
  }

  return res;
}
template <size_t Nc, size_t Nd>
KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, Nd> operator*=(Spinor<Nc, Nd>& spinor,
                                                      const real_t& scalar) {
  Spinor<Nc, Nd> res = scalar * spinor;
  spinor = res;
  return spinor;
}

template <size_t Nc, size_t Nd>
KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, Nd> operator+(
    const Spinor<Nc, Nd>& spinor1,
    const Spinor<Nc, Nd>& spinor2) {
  Spinor<Nc, Nd> res;
#pragma unroll
  for (size_t i = 0; i < Nc * Nd; i++) {
    res(i) = spinor2(i) + spinor1(i);
  }

  return res;
}
template <size_t Nc, size_t Nd>
KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, Nd> operator+=(
    Spinor<Nc, Nd>& spinor1,
    const Spinor<Nc, Nd>& spinor2) {
  Spinor<Nc, Nd> res = spinor1 + spinor2;
  spinor1 = res;
  return spinor1;
}

template <size_t Nc, size_t Nd>
KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, Nd> axpy(
    const complex_t& alpha,
    const Spinor<Nc, Nd>& spinor1,
    const Spinor<Nc, Nd>& spinor2) {  // returns alpha*spinor1 + spinor2
  Spinor<Nc, Nd> res;
#pragma unroll
  for (size_t i = 0; i < Nd * Nc; i++) {
    res(i) = spinor2(i) + alpha * spinor1(i);
  }
  return res;
}

template <size_t Nc, size_t Nd>
KOKKOS_FORCEINLINE_FUNCTION void axpy(const complex_t& alpha,
                                      const Spinor<Nc, Nd>& spinor1,
                                      Spinor<Nc, Nd>& spinor2,
                                      Spinor<Nc, Nd>& res) {
#pragma unroll
  for (size_t i = 0; i < Nc * Nd; i++) {
    spinor2(i) += alpha * spinor1(i);
    spinor2(i) += alpha * spinor1(i);
  }
}

template <size_t Nc, size_t Nd>
KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, Nd> operator-(
    const Spinor<Nc, Nd>& spinor1,
    const Spinor<Nc, Nd>& spinor2) {
  Spinor<Nc, Nd> res;
#pragma unroll
  for (size_t j = 0; j < Nd; j++) {
#pragma unroll
    for (size_t i = 0; i < Nc; i++) {
      res(j, i) = spinor1(j, i) - spinor2(j, i);
    }
  }
  return res;
}
template <size_t Nc, size_t Nd>
KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, Nd> operator-=(
    Spinor<Nc, Nd>& spinor1,
    const Spinor<Nc, Nd>& spinor2) {
  Spinor<Nc, Nd> res = spinor1 - spinor2;
  spinor1 = res;
  return spinor1;
}
template <size_t Nc, size_t Nd>
KOKKOS_FORCEINLINE_FUNCTION real_t sqnorm(const Spinor<Nc, Nd>& spinor) {
  real_t res = 0;
#pragma unroll
  for (size_t j = 0; j < Nd; j++) {
#pragma unroll
    for (size_t i = 0; i < Nc; i++) {
      res += spinor(j, i).imag() * spinor(j, i).imag() +
             spinor(j, i).real() * spinor(j, i).real();
    }
  }
  return res;
}

// Define Gamma Spinor interaction
// Dirac index and gamma matrix have to have the same dimension

// This is ineficnet because of the sparsity of the gamma matrices
template <size_t Nc, size_t Nd>
KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, Nd> operator*(
    const GammaMat<Nd>& matrix,
    const Spinor<Nc, Nd>& spinor) {
  Spinor<Nc, Nd> c;
#pragma unroll
  for (size_t j = 0; j < Nd; j++) {
#pragma unroll
    for (size_t i = 0; i < Nc; i++) {
      complex_t val = 0.0;
#pragma unroll
      for (size_t k = 0; k < Nd; k++) {
        val += matrix(j, k) * spinor(k, i);
      }
      c(j, i) = val;
    }
  }
  return c;
}

template <size_t Nc, size_t Nd>
KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, Nd> operator*(
    const Spinor<Nc, Nd>& spinor,
    const GammaMat<Nd>& matrix) {
  Spinor<Nc, Nd> c;
#pragma unroll
  for (size_t j = 0; j < Nd; ++j) {
#pragma unroll
    for (size_t i = 0; i < Nc; ++i) {
      complex_t val = 0.0;
#pragma unroll
      for (size_t k = 0; k < Nd; ++k) {
        val += spinor(k, i) * matrix(k, j);
      }
      c(j, i) = val;
    }
  }
  return c;
}

// Random generation of Spinors
template <size_t Nc, size_t Nd, class RNG>
KOKKOS_FORCEINLINE_FUNCTION void randSpinor(Spinor<Nc, Nd>& r,
                                            RNG& generator,
                                            const real_t& mean,
                                            const real_t& var) {
#pragma unroll
  for (size_t j = 0; j < Nd; ++j) {
#pragma unroll
    for (size_t i = 0; i < Nc; ++i) {
      r[j][i] =
          complex_t(generator.normal(mean, var), generator.normal(mean, var));
    }
  }
}

// calculate a^\dagger b
template <size_t Nc, size_t Nd>
KOKKOS_FORCEINLINE_FUNCTION complex_t
spinor_inner_prod(const Spinor<Nc, Nd>& a, const Spinor<Nc, Nd>& b) {
  complex_t res(0.0, 0.0);
#pragma unroll
  for (size_t j = 0; j < Nd; ++j) {
#pragma unroll
    for (size_t i = 0; i < Nc; ++i) {
      res += conj(a(j, i)) * b(j, i);
    }
  }
  return res;
}
template <size_t Nc, size_t Nd>
KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, Nd> conj(const Spinor<Nc, Nd>& a) {
  Spinor<Nc, Nd> res;
#pragma unroll
  for (size_t j = 0; j < Nd; ++j) {
#pragma unroll
    for (size_t i = 0; i < Nc; ++i) {
      res(j, i) = conj(a(j, i));
    }
  }
  return res;
}
template <size_t Nc, size_t Nd>
void print_spinor_int(const Spinor<Nc, Nd>& s, const char* name = "Spinor") {
  printf("%s:\n", name);
  for (size_t d = 0; d < Nd; ++d) {
    printf("  Spin %zu:\n", d);
    for (size_t c = 0; c < Nc; ++c) {
      double re = s(d, c).real();
      double im = s(d, c).imag();
      printf("    [%zu] = (% .20f, % .20f i)\n", c, re, im);
    }
  }
}
}  // namespace klft