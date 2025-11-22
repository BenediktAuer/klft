#pragma once
#include <iostream>
#include <string>
#include <variant>
#include <vector>
#include "FieldTypeHelper.hpp"
#include "GLOBAL.hpp"
#include "GaugePlaquette.hpp"
#include "MeasurmentContext.hpp"
#include "WilsonLoop.hpp"
namespace klft {

template <typename T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& v) {
  out << "{";
  size_t last = v.size() - 1;
  for (size_t i = 0; i < v.size(); ++i) {
    out << v[i];
    if (i != last)
      out << ", ";
  }
  out << "}\n";
  return out;
}
template <typename T, size_t N>
std::ostream& operator<<(std::ostream& out, const Kokkos::Array<T, N>& v) {
  out << "{";
  size_t last = v.size() - 1;
  for (size_t i = 0; i < v.size(); ++i) {
    out << v[i];
    if (i != last)
      out << ", ";
  }
  out << "}\n";
  return out;
}
template <typename T>
class MeasurementResult {
 public:
  std::string name;
  std::vector<int> trajectory;

  using ResultVariant = std::vector<T>;
  template <typename IndexType>
  T operator()(const IndexType& i) {
    return value[i];
  }
  template <typename IndexType>
  std::pair<int, T> operator()(const IndexType& i, const IndexType j) {
    return {trajectory[i], value[i]};
  }

  ResultVariant value;
  inline void clear() {
    trajectory.clear();
    value.clear();
  }
  inline size_t size() { return trajectory.size() }
};
template <typename ContextT>
class IMeasurementBase {
 public:
  int interval = 1;
  int thermalization_steps;
  virtual ~IMeasurementBase() = default;

  virtual std::string name() const = 0;

  void measure(ContextT& ctx) {
    if (ctx.step < thermalization_steps || (ctx.step % interval != 0)) {
      return;
    }
    measure_impl(ctx);
  }

  virtual void measure_impl(ContextT& ctx) = 0;
  virtual void clear() = 0;
};

template <typename ContextT, typename T>
class IMeasurement : public IMeasurementBase<ContextT> {
 public:
  using Context = ContextT;

  void add_measurement(const int step, const T& value) {
    resultbuffer.trajectory.push_back(step);
    resultbuffer.value.push_back(value);
  }

  void insert_measurement(
      const int step,
      const typename MeasurementResult<T>::ResultVariant& value) {
    resultbuffer.trajectory.push_back(step);
    resultbuffer.value.insert(resultbuffer.value.end(), value.begin(),
                              value.end());
  }

  MeasurementResult<T>& get_result() { return resultbuffer; }
  void clear() override { resultbuffer.clear(); }

 private:
  MeasurementResult<T> resultbuffer;
};

template <typename ContextT>
class PlaquetteMeasurement : public IMeasurement<ContextT, real_t> {
  std::string name() const override { return "Plaquette"; }

  void measure_impl(ContextT& context) override {
    // using GaugeFieldType = typename ContextT::DGaugeFieldType::type;
    constexpr static const size_t Nd = DeviceGaugeFieldTypeTraits<
        typename ContextT::AbstractGaugeFieldType>::Rank;
    constexpr static const size_t Nc = DeviceGaugeFieldTypeTraits<
        typename ContextT::AbstractGaugeFieldType>::Nc;
    real_t plaq = GaugePlaquette<Nd, Nc>(context.gauge_field);
    this->add_measurement(context.step, plaq);
  }
};
template <typename ContextT>
class WilsonLoopTemporalMeasurement
    : public IMeasurement<ContextT, Kokkos::Array<real_t, 3>> {
  WilsonLoopTemporalMeasurement(
      const std::vector<Kokkos::Array<index_t, 2>>& L_T_pairs_)
      : L_T_pairs(L_T_pairs_),
        IMeasurement<ContextT, Kokkos::Array<real_t, 3>>() {}
  std::string name() const override { return "WilsonLoopTemporal"; }
  std::vector<Kokkos::Array<real_t, 3>> measurements;
  const std::vector<Kokkos::Array<index_t, 2>>& L_T_pairs;
  void measure_impl(ContextT& context) override {
    // using GaugeFieldType = typename ContextT::DGaugeFieldType::type;
    constexpr static const size_t Nd = DeviceGaugeFieldTypeTraits<
        typename ContextT::AbstractGaugeFieldType>::Rank;
    constexpr static const size_t Nc = DeviceGaugeFieldTypeTraits<
        typename ContextT::AbstractGaugeFieldType>::Nc;

    WilsonLoop_temporal<Nd, Nc>(context.gauge_field, L_T_pairs, measurements);

    this->insert_measurement(context.step, measurements);
    measurements.clear();
  }
};
template <typename ContextT>
class WilsonLoop_mu_nuMeasuremnt
    : public IMeasurement<ContextT, Kokkos::Array<real_t, 5>> {
  std::vector<Kokkos::Array<real_t, 5>> temp_measurements;
  std::vector<Kokkos::Array<index_t, 2>> W_mu_nu_pairs;
  std::vector<Kokkos::Array<index_t, 2>> W_Lmu_Lnu_pairs;
  std::string name() const override { return "WilsonLoop_mu_nu"; }
  void measure_impl(ContextT& context) override {
    constexpr static const size_t Nd = DeviceGaugeFieldTypeTraits<
        typename ContextT::AbstractGaugeFieldType>::Rank;
    constexpr static const size_t Nc = DeviceGaugeFieldTypeTraits<
        typename ContextT::AbstractGaugeFieldType>::Nc;
    for (const auto& pair_mu_nu : W_mu_nu_pairs) {
      const index_t mu = pair_mu_nu[0];
      const index_t nu = pair_mu_nu[1];
      WilsonLoop_mu_nu<Nd, Nc>(context.gauge_field, mu, nu, W_Lmu_Lnu_pairs,
                               temp_measurements);
      if (KLFT_VERBOSITY > 1) {
        for (const auto& measure : temp_measurements) {
          printf("%d, %d, %d, %d, %11.6f\n", static_cast<index_t>(measure[0]),
                 static_cast<index_t>(measure[1]),
                 static_cast<index_t>(measure[2]),
                 static_cast<index_t>(measure[3]), measure[4]);
        }
      }
    }
    this->insert_measurement(context.step, temp_measurements);
    temp_measurements.clear();
  }
};

class AcceptRateMeasurement
    : public IMeasurement<MeasuremntIOContext, Kokkos::Array<real_t, 2>> {
  std::string name() const override { return "Acceptance,accept"; }
  real_t acc_sum = 0;
  void measure_impl(MeasuremntIOContext& context) override {
    acc_sum += static_cast<real_t>(context.accepted);
    this->add_measurement(
        context.step,
        Kokkos::Array<real_t, 2>{
            acc_sum / static_cast<real_t>(context.step + 1), context.accepted});
  }
};

class TimeMeasurement : public IMeasurement<MeasuremntIOContext, real_t> {
  std::string name() const override { return "LogTime"; }
  void measure_impl(MeasuremntIOContext& context) override {
    this->add_measurement(context.step, context.time);
  }
};
class ObsTimeMeasurement : public IMeasurement<MeasuremntIOContext, real_t> {
  std::string name() const override { return "ObsTime"; }
  void measure_impl(MeasuremntIOContext& context) override {
    this->add_measurement(context.step, context.obs_time);
  }
};
class DeltaHMeasurement : public IMeasurement<MeasuremntIOContext, real_t> {
  std::string name() const override { return "DeltaH"; }
  void measure_impl(MeasuremntIOContext& context) override {
    this->add_measurement(context.step, context.deltaH);
  }
};
}  // namespace klft
