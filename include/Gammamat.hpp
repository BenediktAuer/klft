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

// Define the global types and views for Kokkos Lattice Field Theory (KLFT)
// This file contains the definitions for the types used in KLFT, including
// the real and complex types, gauge field types, and field view types.
// It also includes the definitions for the policies used in Kokkos parallel
// programming.

#pragma once
#include "GLOBAL.hpp"

// Implemtation of Dirac Gamma Matricies
namespace klft {
// using RepDim =size_t 4;

KOKKOS_FORCEINLINE_FUNCTION
GammaMat<4> gamma0() {
  GammaMat<4> g;
  g[0][0] = complex_t(0, 0);
  g[0][1] = complex_t(0, 0);
  g[0][2] = complex_t(-1, 0);
  g[0][3] = complex_t(0, 0);
  g[1][0] = complex_t(0, 0);
  g[1][1] = complex_t(0, 0);
  g[1][2] = complex_t(0, 0);
  g[1][3] = complex_t(-1, 0);
  g[2][0] = complex_t(-1, 0);
  g[2][1] = complex_t(0, 0);
  g[2][2] = complex_t(0, 0);
  g[2][3] = complex_t(0, 0);
  g[3][0] = complex_t(0, 0);
  g[3][1] = complex_t(-1, 0);
  g[3][2] = complex_t(0, 0);
  g[3][3] = complex_t(0, 0);
  return g;
}
KOKKOS_FORCEINLINE_FUNCTION
GammaMat<4> gamma1() {
  GammaMat<4> g;
  g[0][0] = complex_t(0, 0);
  g[0][1] = complex_t(0, 0);
  g[0][2] = complex_t(0, 0);
  g[0][3] = complex_t(0, -1);
  g[1][0] = complex_t(0, 0);
  g[1][1] = complex_t(0, 0);
  g[1][2] = complex_t(0, -1);
  g[1][3] = complex_t(0, 0);
  g[2][0] = complex_t(0, 0);
  g[2][1] = complex_t(0, 1);
  g[2][2] = complex_t(0, 0);
  g[2][3] = complex_t(0, 0);
  g[3][0] = complex_t(0, 1);
  g[3][1] = complex_t(0, 0);
  g[3][2] = complex_t(0, 0);
  g[3][3] = complex_t(0, 0);
  return g;
}
KOKKOS_FORCEINLINE_FUNCTION
GammaMat<4> gamma2() {
  GammaMat<4> g;
  g[0][0] = complex_t(0, 0);
  g[0][1] = complex_t(0, 0);
  g[0][2] = complex_t(0, 0);
  g[0][3] = complex_t(-1, 0);
  g[1][0] = complex_t(0, 0);
  g[1][1] = complex_t(0, 0);
  g[1][2] = complex_t(1, 0);
  g[1][3] = complex_t(0, 0);
  g[2][0] = complex_t(0, 0);
  g[2][1] = complex_t(1, 0);
  g[2][2] = complex_t(0, 0);
  g[2][3] = complex_t(0, 0);
  g[3][0] = complex_t(-1, 0);
  g[3][1] = complex_t(0, 0);
  g[3][2] = complex_t(0, 0);
  g[3][3] = complex_t(0, 0);
  return g;
}

KOKKOS_FORCEINLINE_FUNCTION
GammaMat<4> gamma3() {
  GammaMat<4> g;
  g[0][0] = complex_t(0, 0);
  g[0][1] = complex_t(0, 0);
  g[0][2] = complex_t(0, -1);
  g[0][3] = complex_t(0, 0);
  g[1][0] = complex_t(0, 0);
  g[1][1] = complex_t(0, 0);
  g[1][2] = complex_t(0, 0);
  g[1][3] = complex_t(0, 1);
  g[2][0] = complex_t(0, 1);
  g[2][1] = complex_t(0, 0);
  g[2][2] = complex_t(0, 0);
  g[2][3] = complex_t(0, 0);
  g[3][0] = complex_t(0, 0);
  g[3][1] = complex_t(0, 1);
  g[3][2] = complex_t(0, 0);
  g[3][3] = complex_t(0, 0);
  return g;
}
KOKKOS_FORCEINLINE_FUNCTION
GammaMat<4> gamma5() {
  GammaMat<4> g;
  g[0][0] = complex_t(1, 0);
  g[0][1] = complex_t(0, 0);
  g[0][2] = complex_t(0, 0);
  g[0][3] = complex_t(0, 0);
  g[1][0] = complex_t(0, 0);
  g[1][1] = complex_t(1, 0);
  g[1][2] = complex_t(0, 0);
  g[1][3] = complex_t(0, 0);
  g[2][0] = complex_t(0, 0);
  g[2][1] = complex_t(0, 0);
  g[2][2] = complex_t(-1, 0);
  g[2][3] = complex_t(0, 0);
  g[3][0] = complex_t(0, 0);
  g[3][1] = complex_t(0, 0);
  g[3][2] = complex_t(0, 0);
  g[3][3] = complex_t(-1, 0);
  return g;
}
template <size_t RepDim>
KOKKOS_FORCEINLINE_FUNCTION GammaMat<RepDim> operator*(
    const GammaMat<RepDim> &a, const GammaMat<RepDim> &b) {
  GammaMat<RepDim> c;
#pragma unroll
  for (size_t i = 0; i < RepDim; ++i) {
#pragma unroll
    for (size_t j = 0; j < RepDim; ++j) {
      c[i][j] = a[i][0] * b[0][j];
#pragma unroll
      for (size_t k = 1; k < RepDim; ++k) {
        c[i][j] += a[i][k] * b[k][j];
      }
    }
  }
  return c;
}

template <size_t RepDim>
KOKKOS_FORCEINLINE_FUNCTION GammaMat<RepDim> operator*=(
    GammaMat<RepDim> &a, const GammaMat<RepDim> &b) {
  GammaMat<RepDim> c = a * b;
  a = c;
  return a;
}
template <size_t RepDim>
KOKKOS_FORCEINLINE_FUNCTION GammaMat<RepDim> operator*(
    const GammaMat<RepDim> &a, const real_t &b) {
  GammaMat<RepDim> c;
#pragma unroll
  for (size_t i = 0; i < RepDim; ++i) {
#pragma unroll
    for (size_t j = 0; j < RepDim; ++j) {
      c[i][j] = a[i][j] * b;
    }
  }
  return c;
}
KOKKOS_FORCEINLINE_FUNCTION GammaMat<RepDim> operator*(
    const GammaMat<RepDim> &a, const complex_t &b) {
  GammaMat<RepDim> c;
#pragma unroll
  for (size_t i = 0; i < RepDim; ++i) {
#pragma unroll
    for (size_t j = 0; j < RepDim; ++j) {
      c[i][j] = a[i][j] * b;
    }
  }
  return c;
}

}  // namespace klft