#pragma once
#include <mpi.h>

#include <random>
#include <sstream>

#include "FieldTypeHelper.hpp"
#include "GLOBAL.hpp"
#include "GaugeObservable.hpp"
#include "HMC.hpp"
#include "HMC_Params.hpp"
#include "HamiltonianField.hpp"
#include "IndexHelper.hpp"
#include "SimulationLogging.hpp"
using RNGType = Kokkos::Random_XorShift64_Pool<Kokkos::DefaultExecutionSpace>;

namespace klft {

struct PTBCParams {
  // Define parameters for the PTBC algorithm
  index_t n_sims;
  std::vector<real_t>
      defects;            // a vector that hold the different defect values
  index_t defect_length;  // size of the defect on the lattice
  GaugeMonomial_Params gauge_params;  // HMC parameters´
  PTBCSimulationLoggingParams ptbcSimLogParams;
  GaugeObservableParams gaugeObsParams;
  SimulationLoggingParams simLogParams;

  std::string to_string() const {
    std::ostringstream oss;
    oss << "PTBCParams: n_sims = " << n_sims
        << ", defect_length = " << defect_length << ", defects = [";
    for (const auto& defect : defects) {
      oss << defect << " ";
    }
    oss << "]";
    return oss.str();
  }
};

template <typename DGaugeFieldType, typename DAdjFieldType, class RNG>
class PTBC {  // do I need the AdjFieldType here?
  // template argument deduction and safety
  static_assert(DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Kind ==
                GaugeFieldKind::PTBC);
  static_assert(isDeviceGaugeFieldType<DGaugeFieldType>::value);
  static_assert(isDeviceAdjFieldType<DAdjFieldType>::value);
  constexpr static size_t Nd =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank;
  constexpr static size_t Nc = DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Nc;
  static_assert((Nd == DeviceAdjFieldTypeTraits<DAdjFieldType>::Rank) &&
                (Nc == DeviceAdjFieldTypeTraits<DAdjFieldType>::Nc));

  using GaugeField = typename DGaugeFieldType::type;
  using AdjointField = typename DAdjFieldType::type;
  using HField = HamiltonianField<DGaugeFieldType, DAdjFieldType>;
  using HMCType = HMC<DGaugeFieldType, DAdjFieldType, RNG>;

 public:
  PTBCParams& params;  // parameters for the PTBC algorithm
  HMCType& hmc;
  // std::shared_ptr<LeapFrog> integrator; // TODO: make integrator agnostic
  RNG& rng;
  std::mt19937 mt;
  std::uniform_real_distribution<real_t> dist;

  index_t current_index;

  std::vector<bool> swap_accepts;   // a vector that hold the last values shown
                                    // if a given swap was accepted
  std::vector<real_t> swap_deltas;  // a vector that holds the partial Delta_S
                                    // values for each swap
  int _swap_start;                  // holds the rank of the last swap start

  typedef enum {
    TAG_DELTAS = 0,
    TAG_ACCEPT = 1,
    TAG_INDEX = 2,
    TAG_SWAPSTART = 3,
    TAG_SHIFTDEFECT = 4
  } MPI_Tags;

  PTBC() = delete;  // default constructor is not allowed

  PTBC(PTBCParams& params_, HMCType& hmc_, RNG& rng_,
       std::uniform_real_distribution<real_t> dist_, std::mt19937 mt_)
      : params(params_), rng(rng_), dist(dist_), mt(mt_), hmc(hmc_) {
    device_id = Kokkos::device_id();  // default device id
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    current_index = rank;

    swap_accepts.resize(params.defects.size());
    swap_deltas.resize(params.defects.size());

    hmc.hamiltonian_field.gauge_field.template set_defect<index_t>(
        params.defects[current_index]);
    Kokkos::fence();

    // host only fallback
    if (device_id == -1) {
      device_id = 0;
      partner_device_id = 0;  // default to first two devices
    } else {
      partner_device_id =
          (device_id + 1) % Kokkos::num_devices();  // default partner device id
    }
  }

  real_t getDefectValue() const {
    // return the defect value for the current index
    return params.defects[current_index];
  }

  void measure(PTBCSimulationLoggingParams& ptbcSimLogParams,
               const size_t step) {
    // measure the simulation logging observables
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (rank == 0) {
      addPTBCLogData(ptbcSimLogParams, step, _swap_start, &swap_accepts,
                     &swap_deltas, &params.defects);  // add the data to the log
    }
  }

  void measure(GaugeObservableParams& gaugeObsParams, size_t step) {
    // measure the gauge observables
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int compute_rank{0};
    int dummy_rank{0};

    if (rank == 0 || getDefectValue() >= 0.99999) {
      // only the computing rank or the defect rank measures the observables
      MPI_Reduce(
          &rank, &compute_rank, 1, MPI_INT, MPI_SUM, 0,
          MPI_COMM_WORLD);  // sum the ranks to get the computing rank in rank 0
      if (getDefectValue() >= 0.99999) {
        // if the defect value is close to 1, we measure the observables
        // for the PTBC case
        if (KLFT_VERBOSITY > 1) {
          printf("Measuring PTBC observables at step %zu\n", step);
        }
        measureGaugeObservablesPTBC<DGaugeFieldType>(
            hmc.hamiltonian_field.gauge_field, gaugeObsParams, step, 0, true);
      } else {
        measureGaugeObservablesPTBC<DGaugeFieldType>(
            hmc.hamiltonian_field.gauge_field, gaugeObsParams, step,
            compute_rank, false);
      }
    } else {
      MPI_Reduce(&dummy_rank, &compute_rank, 1, MPI_INT, MPI_SUM, 0,
                 MPI_COMM_WORLD);
      return;  // skip measurement for other ranks
    }
  }

  void measure(SimulationLoggingParams& simLogParams, index_t step,
               real_t acc_rate, bool accept, real_t time) {
    // measure the simulation logging observables
    addLogData(simLogParams, step, hmc.delta_H, acc_rate, accept, time);
  }

  int step() {
    auto rtn = hmc.hmc_step();
    return rtn;
  }

  real_t swap_partner(index_t partner_rank) {
    // swaps the Defect with the partner rank and returns the partial Delta_S

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    real_t S_ss = getActionAroundDefect();
    // DEBUG_MPI_PRINT("S_ss = %f", S_ss);
    // now do the index/defect swapping
    hmc.hamiltonian_field.gauge_field.template set_defect<index_t>(
        params.defects[partner_rank]);  // set the defect value of the current
    //
    real_t S_sr = getActionAroundDefect();
    // DEBUG_MPI_PRINT("S_sr = %f", S_sr);
    return S_sr - S_ss;  // return the partial Delta_S
  }

  void reverse_swap(bool accept) {
    if (!accept) {
      hmc.hamiltonian_field.gauge_field.template set_defect<index_t>(
          params
              .defects[current_index]);  // set the defect value of the current
    }
  }

  void checkForNANs() {
    if constexpr (Nd == 4) {
#ifdef DEBUG_MPI
      // Create a host mirror and copy data from device to host
      auto host_field = Kokkos::create_mirror_view_and_copy(
          Kokkos::HostSpace(), hmc.hamiltonian_field.gauge_field.field);
      auto dimensions = hmc.hamiltonian_field.gauge_field.dimensions;

      for (index_t i = 0; i < dimensions[0]; ++i)
        for (index_t j = 0; j < dimensions[1]; ++j)
          for (index_t k = 0; k < dimensions[2]; ++k)
            for (index_t l = 0; l < dimensions[3]; ++l)
              for (index_t mu = 0; mu < Nd; ++mu) {
                auto link = host_field(i, j, k, l, mu);
                for (index_t c1 = 0; c1 < Nc; ++c1)
                  for (index_t c2 = 0; c2 < Nc; ++c2) {
                    if (std::isnan(Kokkos::real(link[c1][c2]))) {
                      DEBUG_MPI_PRINT("NaN at (%d,%d,%d,%d,mu=%d)", i, j, k, l,
                                      mu);
                    }
                  }
              }
#endif
    }
  }
  void shift_defect() {
    int size, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    // shift[0] is the direction, shift [1] is the amount (1 or -1)
    int shift[2];
    if (rank == 0) {
      shift[0] =
          int(dist(mt) *
              (Nd - 1));  // random direction but not in t direction ( so 0,1,2)
      shift[1] = (dist(mt) > 0.5) ? 1 : -1;
    }
    MPI_Bcast(shift, 2, MPI_INT, 0, MPI_COMM_WORLD);
    auto old_position =
        hmc.hamiltonian_field.gauge_field.dParams.defect_position;
    auto new_position = old_position;
    new_position[shift[0]] =
        (old_position[shift[0]] + shift[1] +
         (old_position[shift[0]] == 0) * (shift[1] < 0) *
             hmc.hamiltonian_field.gauge_field.dimensions[shift[0]]) %
        hmc.hamiltonian_field.gauge_field.dimensions[shift[0]];
    hmc.hamiltonian_field.gauge_field.shift_defect(new_position);
  }
  int swap() {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int partner_rank;
    bool accept = false;
    real_t Delta_S{0};
    int swap_start{0};
    int swap_rank{0};

    // Rank 0 determines swap_start and broadcasts
    if (rank == 0) {
      swap_start = int(dist(mt) * (size));
    }

    MPI_Bcast(&swap_start, 1, MPI_INT, 0, MPI_COMM_WORLD);

    for (index_t i = 0; i < size - 1; ++i) {
      swap_rank = (swap_start + i) % size;
      partner_rank = (swap_rank + 1) % size;

      // SWAP RANK sends its Delta_S
      if (rank == swap_rank) {
        // DEBUG_MPI_PRINT(
        //     "%s, DefectLength(Field): %d", params.to_string().c_str(),
        //     hmc.hamiltonian_field.gauge_field.dParams.defect_length);
        // DEBUG_MPI_PRINT("Iteration %d: swap_rank=%d, defect(PTBC)=%f , "
        //                 "defect(Field)=%f \n\t "
        //                 "partner_rank = % d defect(PTBC) = % f ",
        //                 i, swap_rank, params.defects[swap_rank],
        //                 hmc.hamiltonian_field.gauge_field.get_defect(),
        //                 partner_rank, params.defects[partner_rank]);

        real_t temp = swap_partner(partner_rank);

        MPI_Send(&temp, 1, mpi_real_t(), 0, TAG_DELTAS, MPI_COMM_WORLD);
      }

      // PARTNER RANK sends its Delta_S
      if (rank == partner_rank) {
        real_t temp = swap_partner(swap_rank);
        MPI_Send(&temp, 1, mpi_real_t(), 0, TAG_DELTAS, MPI_COMM_WORLD);
      }

      // RANK 0 gathers both values
      if (rank == 0) {
        real_t Delta_S_partner1, Delta_S_partner2;
        MPI_Recv(&Delta_S_partner1, 1, mpi_real_t(), swap_rank, TAG_DELTAS,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        MPI_Recv(&Delta_S_partner2, 1, mpi_real_t(), partner_rank, TAG_DELTAS,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        Delta_S = Delta_S_partner1 + Delta_S_partner2;

        // Decide accept/reject
        accept = true;
        if (Delta_S > 0) {
          if (dist(mt) > Kokkos::exp(-Delta_S)) {
            accept = false;
          }
        }

        MPI_Send(&accept, 1, MPI_C_BOOL, swap_rank, TAG_ACCEPT, MPI_COMM_WORLD);
        MPI_Send(&accept, 1, MPI_C_BOOL, partner_rank, TAG_ACCEPT,
                 MPI_COMM_WORLD);
      }

      // Swap ranks receive accept flag
      if (rank == swap_rank || rank == partner_rank) {
        MPI_Recv(&accept, 1, MPI_C_BOOL, 0, TAG_ACCEPT, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

        reverse_swap(accept);
      }

      // Rank 0 performs the swap if accepted
      if (rank == 0 && accept) {
        std::swap(params.defects[swap_rank], params.defects[partner_rank]);
      }

      MPI_Barrier(MPI_COMM_WORLD);  // synchronize all ranks after each swap
      MPI_Bcast(params.defects.data(), params.defects.size(), mpi_real_t(), 0,
                MPI_COMM_WORLD);

      if (rank == 0) {
        std::ostringstream oss;
        oss << "Defects after broadcast: [";
        for (size_t i = 0; i < params.defects.size(); ++i) {
          oss << params.defects[i];
          if (i + 1 < params.defects.size()) oss << ", ";
        }
        oss << "]";
        // DEBUG_MPI_PRINT("%s", oss.str().c_str());
      }
      MPI_Barrier(MPI_COMM_WORLD);  // synchronize all ranks after each swap
      if (rank == 0) {              // add the swap data to the logs
        swap_accepts[swap_rank] = accept;
        swap_deltas[swap_rank] = Delta_S;
        _swap_start = swap_start;  // store the swap start rank
      }
    }
    // TODO: shift the defect by one lattice spacing in a random direction
    shift_defect();
    MPI_Barrier(MPI_COMM_WORLD);
    //
    return 0;
  }

  real_t getActionAroundDefect() {
    // calculate the plaquette around the defect
    // this is a placeholder function,
    // TODO: implement the actual action around
    // the defect

    auto action = WilsonAction<DGaugeFieldType>(
        hmc.hamiltonian_field.gauge_field, params.gauge_params.beta);

    return action;
  }

 private:
  int device_id;
  int partner_device_id;
};

// below: Functions used to dispatch the PTBC algorithm

template <typename PTBCType>
int run_PTBC(PTBCType& ptbc, Integrator_Params& int_params,
             GaugeObservableParams& gaugeObsParams,
             PTBCSimulationLoggingParams& ptbcSimLogParams,
             SimulationLoggingParams& simLogParams) {
  Kokkos::Timer timer;
  real_t acc_sum{0.0};
  real_t acc_rate{0.0};

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  for (size_t step = 0; step < int_params.nsteps; ++step) {
    timer.reset();

    // DEBUG_MPI_PRINT("Enter ptbc(hmc) step: %zu", step);

    int accept = ptbc.step();
    MPI_Barrier(MPI_COMM_WORLD);  // synchronize all ranks after step
    // DEBUG_MPI_PRINT("HMC Step %zu: accept = %d", step, accept);

    int ptbc_accept = ptbc.swap();

    // Gauge observables
    ptbc.measure(gaugeObsParams, step);
    ptbc.measure(ptbcSimLogParams, step);

    if (rank == 0) {
      flushAllGaugeObservables(gaugeObsParams, step, true);
      flushPTBCSimulationLogs(ptbcSimLogParams, step, true);
    }
    // PTBC swap/accept

    const real_t time = timer.seconds();
    acc_sum += static_cast<real_t>(accept);
    acc_rate = acc_sum / static_cast<real_t>(step + 1);
    ptbc.measure(simLogParams, step, acc_rate, accept, time);
    if (rank == 0) {
      Kokkos::printf("Step: %zu, accepted: %d, Acceptance rate: %f, Time: %f\n",
                     step, accept, acc_rate, time);
    }
    flushSimulationLogs(simLogParams, step, true);
  }

  MPI_Barrier(MPI_COMM_WORLD);  // synchronize all ranks after the loop
  if (rank == 0) {
    forceflushAllGaugeObservables(gaugeObsParams, true);
    forceflushPTBCSimulationLogs(ptbcSimLogParams, true);
  }
  forceflushSimulationLogs(simLogParams, true);

  return 0;
}

// #define INITIALIZE_PTBCPREPARE(ND, NC, RNG) \
//   template int run_PTBC<DeviceGaugeFieldType<ND, NC, GaugeFieldKind::PTBC>, \
//                         DeviceAdjFieldType<ND, NC>, RNG>( \
//       PTBCParams, RNG &, std::uniform_real_distribution<real_t>, \
//       std::mt19937);
// // INITIALIZE_PTBCPREPARE(2, 1, RNGType)
// // INITIALIZE_PTBCPREPARE(2, 2, RNGType)
// // // INITIALIZE_PTBCPREPARE(2, 3, RNGType)
// // INITIALIZE_PTBCPREPARE(3, 1, RNGType)
// // INITIALIZE_PTBCPREPARE(3, 2, RNGType)
// // INITIALIZE_PTBCPREPARE(3, 3, RNGType)
// INITIALIZE_PTBCPREPARE(4, 1, RNGType)
// INITIALIZE_PTBCPREPARE(4, 2, RNGType)
// // INITIALIZE_PTBCPREPARE(4, 3, RNGType)
// #undef INITIALIZE_PTBCPREPARE
//

}  // namespace klft
