#pragma once
#include "FieldTypeHelper.hpp"
#include "GLOBAL.hpp"
namespace klft {
// For now only the gauge field is stored in the context allows for extentsion
// later
struct IMeasurmentContext {
  IMeasurmentContext(const int& step) : step(step) {};

  int step;
  void increase_step() { step++; }
};

template <typename DGaugeFieldType>
struct MeasurementContext : IMeasurmentContext {
  using AbstractGaugeFieldType = DGaugeFieldType;
  using GaugeFieldType = typename DGaugeFieldType::type;
  MeasurementContext(GaugeFieldType& gauge_field)
      : IMeasurmentContext(0), gauge_field(gauge_field) {};

  GaugeFieldType gauge_field;
};
struct MeasuremntIOContext : IMeasurmentContext {
  real_t time;
  real_t obs_time;
  real_t accepted;
  real_t deltaH;
};
}  // namespace klft
