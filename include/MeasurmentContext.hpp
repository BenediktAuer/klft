#pragma once
#include "FieldTypeHelper.hpp"
#include "GLOBAL.hpp"
#include "WilsonFlow.hpp"
namespace klft {
// For now only the gauge field is stored in the context allows for extentsion
// later
struct IMeasurmentContext {
  IMeasurmentContext(const size_t& step) : step(step) {};

  int step;
  void increase_step() { step++; }
  // std::optional<int> rank;
  std::optional<int> measure_rank;
  bool flowed = false;
  void reset_flow() { this->flowed = false; }
  // std::optional<real_t> c_value;

  void set_measure_rank(const int& measure_rank) {
    this->measure_rank.emplace(measure_rank);
  }
  // void set_c_value(const real_t& c_value) { this->c_value = c_value; }
  // int registerMPIRank() {
  //   int local_rank;
  //   int stat = MPI_Comm_rank(MPI_COMM_WORLD, &local_rank);
  //   this->rank.emplace(local_rank);
  //   return stat;
  // }
  // int getRank() {
  //   if (rank.has_value()) {
  //     return rank.value();
  //   }
  //   printf("ERROR: No MPI Rank set!\n, fallback 0");
  //   return 0;
  // }
  int getMeasurmentRank() {
    if (measure_rank.has_value()) {
      return measure_rank.value();
    }
    printf("ERROR: No Measurment Rank set!\n, fallback 0");
    return 0;
  }
};

template <typename DGaugeFieldType>
struct MeasurementContext : IMeasurmentContext {
  constexpr static size_t rank =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank;
  constexpr static size_t Nc = DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Nc;
  using AbstractGaugeFieldType = DGaugeFieldType;  // this important here;
  using GaugeFieldType =
      typename AbstractGaugeFieldType::type;  // this important here
  MeasurementContext(const GaugeFieldType& gauge_field)

      : IMeasurmentContext(0), gauge_field(gauge_field) {

        };
  MeasurementContext(const GaugeFieldType& gauge_field,
                     WilsonFlowParams& wflowparams)

      : IMeasurmentContext(0), gauge_field(gauge_field) {
    printf("%s", wflowparams.to_string().c_str());
    wflow.emplace(
        WilsonFlow<AbstractGaugeFieldType>(this->gauge_field, wflowparams));
    this->wflowparams.emplace(wflowparams);
  };

  GaugeFieldType gauge_field;
  //
  std::optional<WilsonFlow<AbstractGaugeFieldType>> wflow;
  std::optional<WilsonFlowParams> wflowparams;

  const GaugeFieldType& get_flowed_gaugeField() {
    if constexpr (rank == 4) {
      if (wflow.has_value() && !this->flowed) {
        auto& wflow_val = wflow.value();
        auto& wflow_params = wflowparams.value();
        auto new_WFLow =
            WilsonFlow<AbstractGaugeFieldType>(this->gauge_field, wflow_params);
        // wflow_val.flow();
        new_WFLow.flow();
        if (KLFT_VERBOSITY > 1) {
          printf("Performing Wilson flow...\n");
        }
        flowed = true;
        wflow.emplace(new_WFLow);
        return wflow.value().field;
      }
      if (wflow.has_value()) {
        auto& wflow_val = wflow.value();
        if (KLFT_VERBOSITY > 1) {
          printf("Return cached flowed field!\n");
        }
        return wflow_val.field;
      }
    }
    printf(
        "ERROR: No WilsonflowParameters where given to this Context, Fallback "
        "unflowed field is beeing used!");
    return gauge_field;
  }
};
struct MeasuremntIOContext : IMeasurmentContext {
  MeasuremntIOContext(const size_t& step,
                      const real_t& time,
                      const real_t& obs_time,
                      const real_t& accepted,
                      const real_t& deltaH)
      : IMeasurmentContext(step),
        accepted(accepted),
        time(time),
        obs_time(obs_time),
        deltaH(deltaH) {};
  real_t time;
  real_t obs_time;
  real_t accepted;
  real_t deltaH;
};
}  // namespace klft
