#include <Kokkos_Complex.hpp>
#include <Kokkos_Core.hpp>
#include <memory>
#include "GLOBAL.hpp"
// #include "FieldTypeHelper.hpp"

#include "../include/SpinorField.hpp"
#include "../include/SpinorFieldLinAlg.hpp"
#include "../include/WilsonDiracOperator.hpp"
#include "../include/klft.hpp"
#include "GaugePlaquette.hpp"
#include "InputParsermeasurements.hpp"
#include "MeasurmentManger.hpp"
#include "Measurments.hpp"
#include "WriteManager.hpp"

#define HLINE "=========================================================\n"

using namespace klft;
template <size_t Nc, size_t Nd>
void print_spinor(const Spinor<Nc, Nd>& s, const char* name = "Spinor") {
  printf("%s:\n", name);
  for (size_t c = 0; c < Nc; ++c) {
    printf("  Color %zu:\n", c);
    for (size_t d = 0; d < Nd; ++d) {
      Kokkos::printf("    [%zu] = (% .6f, % .6f i)\n", d, s[c][d].real(),
                     s[c][d].imag());
    }
  }
}

int main(int argc, char* argv[]) {
  // MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  int RETURNVALUE = 0;
  int rank = 0;
  // MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::string input_file = "../../../input.yaml";

  {
    setVerbosity(5);
    printf("%i", KLFT_VERBOSITY);
    printf("\n=== Testing IO  ===\n");

    index_t L0 = 8, L1 = 8, L2 = 8, L3 = 8;
    diracParams params(-0.5);
    printf("Lattice Dimension %ix%ix%ix%i \n", L0, L1, L2, L3);

    Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/1234 * 100 * rank);

    printf("Generating Random Gauge Config\n");
    deviceGaugeField<4, 2> gauge(L0, L1, L2, L3, random_pool, 1);
    using MyContext = MeasurementContext<DeviceGaugeFieldType<4, 2>>;
    WilsonFlowParams wflowparams{};
    auto ctx = std::make_unique<MyContext>(gauge, wflowparams);
    auto ctx_sim =
        std::make_unique<MeasuremntIOContext>(1, 0.69, 0.32, 1, -0.12);

    ctx->set_measure_rank(1);
    ctx->increase_step();
    ctx_sim->increase_step();
    auto meas_manager = std::make_shared<MeasurementManager<MyContext>>();
    auto simLogger = std::make_shared<SimLogMeasurmentManager>("SimLog", 1);
    simLogger->register_measurement(
        std::make_unique<AcceptRateMeasurement>(1, 0));
    simLogger->register_measurement(std::make_unique<TimeMeasurement>(1, 0));
    simLogger->register_measurement(std::make_unique<ObsTimeMeasurement>(1, 0));
    auto writeManagerSimLog =
        WriteManagerSimLog(FileMode::On, ConsoleMode::On, "./", "SimLog", 1);
    WriteManagerParams wMparam{"./"};
    parseInputFile<MeasurementContext<DeviceGaugeFieldType<4, 2>>>(input_file, "./", meas_manager, wMparam);
    // MPI_Barrier(MPI_COMM_WORLD);
    // meas_manager.register_measurement(
    //     std::make_unique<PlaquetteMeasurement<MyContext>>(1, 1));
    // meas_manager.register_measurement(
    //     std::make_unique<TopologicalChargeMeasurment<MyContext>>(2, 0));
    // std::vector<Kokkos::Array<index_t, 2>> w_temp_loops;
    // w_temp_loops.push_back(Kokkos::Array<index_t, 2>({1, 2}));
    // w_temp_loops.push_back(Kokkos::Array<index_t, 2>({1, 2}));
    // w_temp_loops.push_back(Kokkos::Array<index_t, 2>({1, 3}));
    // meas_manager.register_measurement(
    //     std::make_unique<WilsonLoopTemporalMeasurement<MyContext>>(
    //         1, 0, w_temp_loops));
    // std::cout << w_temp_loops;
    // meas_manager.register_measurement(
    //     std::make_unique<WilsonLoop_mu_nuMeasurement<MyContext>>(
    //         1, 0, w_temp_loops, w_temp_loops));
    // meas_manager.register_measurement(
    //     std::make_unique<SpMaxMeasurment<MyContext>>(1, 0));
    auto writeManager =
        WriteManager<MyContext>(FileMode::On, ConsoleMode::On, "./", "", 1);
    writeManager.register_measurments(meas_manager, rank);
    writeManagerSimLog.register_measurments(simLogger, rank);
    meas_manager->measure(ctx);
    simLogger->measure(ctx_sim);
    // MPI_Barrier(MPI_COMM_WORLD);
    writeManager.flush(1);
    writeManagerSimLog.flush(1);
    meas_manager->measure(ctx);
    writeManager.flush(2);
    auto plaq = GaugePlaquette<4, 2>(gauge);
    // printf("Plaquette: %.21f\n", plaq);
    // printf("Store Gauge Config\n");
    // gauge_U1.save("gauge_U1.dat");
    // printf("Load Gauge Config\n");
    deviceGaugeField<4, 2> gauge_load(L0, L1, L2, L3,
                                      std::string("step_5_gaugeconfig.txt"));
    auto plaq1 = GaugePlaquette<4, 2>(gauge_load, true);
    // printf("Plaquette: %.21f\n", plaq1);
    if (std::abs(plaq - plaq1) > 1e-14) {
      printf("RANK: %dError: plaquette differs after load/save by %.21f \n",
             rank, std::abs(plaq - plaq1));
      RETURNVALUE++;
    } else {
      printf("Passed load/save test with plaquette difference of %.21f \n",
             std::abs(plaq - plaq1));
    }
  }

  printf(HLINE);
  printf("%i Errors durring Testing\n", RETURNVALUE);
  printf(HLINE);
  // RETURNVALUE = !(RETURNVALUE == 0);
  Kokkos::finalize();
  // MPI_Finalize();
  return 0;
}
