add_executable(test_gamma_matrices test_gamma_matricies.cpp)
target_link_libraries(test_gamma_matrices PUBLIC Kokkos::kokkos)

add_executable(test_spinor test_spinor.cpp)
target_link_libraries(test_spinor PUBLIC Kokkos::kokkos)

add_executable(test_deviceSpinorField test_SpinorFields.cpp)
target_link_libraries(test_deviceSpinorField PUBLIC Kokkos::kokkos)

add_executable(test_DiracOperator test_DiracOperator.cpp)
target_link_libraries(test_DiracOperator PUBLIC Kokkos::kokkos)

add_executable(test_GDircOperator test_GDircOperator.cpp)
target_link_libraries(test_GDircOperator PUBLIC Kokkos::kokkos)

add_executable(test_Solver test_Solver.cpp)
target_link_libraries(test_Solver PUBLIC Kokkos::kokkos)