#pragma once
#include <iostream>
#include <string>
#include <variant>
#include <vector>
#include "FieldTypeHelper.hpp"
#include "GLOBAL.hpp"
#include "GaugePlaquette.hpp"
#include "MeasurmentContext.hpp"
#include "TopoCharge.hpp"
#include "WilsonLoop.hpp"
namespace klft {

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
std::ostream& operator<<(std::ostream& out, const std::vector<T>& v) {
  out << "{";
  size_t last = v.size() - 1;
  for (size_t i = 0; i < v.size(); ++i) {
    out << v[i];
    //   if (i != last)
    //     out << ", ";
  }
  out << "}\n";
  return out;
}
typedef enum {
  MPI_GAUGE_OBSERVABLES_PLAQUETTE = 0,
  MPI_GAUGE_OBSERVABLES_WILSON_LOOP_MU_NU = 1,
  MPI_GAUGE_OBSERVABLES_WILSON_LOOP_TEMPORAL = 2,
  MPI_GAUGE_OBSERVABLES_WILSON_LOOP_MU_NU_SIZE = 3,
  MPI_GAUGE_OBSERVABLES_WILSON_LOOP_TEMPORAL_SIZE = 4,
  MPI_GAUGE_OBSERVABLES_TOPOLOGICAL_CHARGE = 5,
  MPI_GAUGE_OBSERVABLES_ACTION_DENSITY = 6,
  MPI_GAUGE_OBSERVABLES_SP_MAX = 7,
  MPI_GAUGE_OBSERVABLES_WILSONFLOW_DETAILS = 8,
  MPI_GAUGE_OBSERVABLES_WILSONFLOW_DETAILS_SIZE = 9,
  MPI_MEASURMENT_NAME = 10
} MPI_GaugeObservableTags;
template <typename T>
class MeasurementResult {
 public:
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
  inline size_t size() { return trajectory.size(); }
};
// Forward definition:
template <typename ContextT>
struct IMeasurementVisitor;

template <typename ContextT>
struct PrintResults;
template <typename ContextT>
class IMeasurementBase {
 public:
  IMeasurementBase(const int& interval, const int& thermalization_steps)
      : interval(interval), thermalization_steps(thermalization_steps) {};
  int interval = 1;
  int thermalization_steps = 0;
  virtual ~IMeasurementBase() = default;

  virtual std::string name() const = 0;
  virtual std::string storageType() const = 0;
  virtual int MPITag() const = 0;
  bool should_measure(const int& step) {
    bool res = !((step < thermalization_steps) || (step % interval != 0));

    return res;
  }
  void measure(ContextT& ctx) {
    if (should_measure(ctx.step)) {
      measure_impl(ctx);
      return;
    }
  }
  bool operator<(const IMeasurementBase& obj) const {
    return name() < obj.name();
  }
  virtual void measure_impl(ContextT& ctx) = 0;
  virtual void clear() = 0;
  virtual void accept(IMeasurementVisitor<ContextT>& v) = 0;
  virtual size_t measurmentSize() = 0;
};
template <typename ContextT, typename T>
class IMeasurement : public IMeasurementBase<ContextT> {
 public:
  using IMeasurementBase<ContextT>::IMeasurementBase;
  using Context = ContextT;

  void add_measurement(const int step, const T& value) {
    resultbuffer.trajectory.push_back(step);
    resultbuffer.value.push_back(value);
  }

  // void insert_measurement(
  //     const int step,
  //     const typename MeasurementResult<T>::ResultVariant& value) {
  //   resultbuffer.trajectory.push_back(step);
  //   resultbuffer.value.insert(resultbuffer.value.end(), value.begin(),
  //                             value.end());
  // }
  // void insert_measurement(
  //     const typename MeasurementResult<T>::ResultVariant& value) {
  //   resultbuffer.value.insert(resultbuffer.value.end(), value.begin(),
  //                             value.end());
  // }
  void insert_step(const int& step) { resultbuffer.trajectory.push_back(step); }
  MeasurementResult<T>& get_result() { return resultbuffer; }
  void clear() override { resultbuffer.clear(); }
  void accept(IMeasurementVisitor<ContextT>& v) override {
    v.visit(*this);  // Dispatch based on T at runtime
  }
  size_t measurmentSize() override { return resultbuffer.size(); }

 private:
  MeasurementResult<T> resultbuffer;
};

template <typename ContextT>
class PlaquetteMeasurement : public IMeasurement<ContextT, real_t> {
  using IMeasurement<ContextT, real_t>::IMeasurement;
  std::string name() const override { return "Plaquette"; }
  std::string storageType() const override { return "real_t"; }
  int MPITag() const override { return MPI_GAUGE_OBSERVABLES_PLAQUETTE; }
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
    : public IMeasurement<ContextT, std::vector<Kokkos::Array<real_t, 3>>> {
 public:
  WilsonLoopTemporalMeasurement(
      const int& interval,
      const int& thermalization_steps,
      const std::vector<Kokkos::Array<index_t, 2>>& L_T_pairs_)
      : L_T_pairs(L_T_pairs_),
        IMeasurement<ContextT, std::vector<Kokkos::Array<real_t, 3>>>(
            interval,
            thermalization_steps) {};
  std::string name() const override { return "WilsonLoopTemporal"; }
  std::string storageType() const override { return "Kokkos_Array_3_real_t"; }
  int MPITag() const override {
    return MPI_GAUGE_OBSERVABLES_WILSON_LOOP_TEMPORAL;
  }

  std::vector<Kokkos::Array<real_t, 3>> measurements;
  const std::vector<Kokkos::Array<index_t, 2>> L_T_pairs;
  void measure_impl(ContextT& context) override {
    // using GaugeFieldType = typename ContextT::DGaugeFieldType::type;
    constexpr static const size_t Nd = DeviceGaugeFieldTypeTraits<
        typename ContextT::AbstractGaugeFieldType>::Rank;
    constexpr static const size_t Nc = DeviceGaugeFieldTypeTraits<
        typename ContextT::AbstractGaugeFieldType>::Nc;

    WilsonLoop_temporal<Nd, Nc>(context.gauge_field, L_T_pairs, measurements);

    this->add_measurement(context.step, measurements);
    measurements.clear();
  }
};
template <typename ContextT>
class WilsonLoop_mu_nuMeasurement
    : public IMeasurement<ContextT, std::vector<Kokkos::Array<real_t, 5>>> {
 public:
  WilsonLoop_mu_nuMeasurement(
      const int& interval,
      const int& thermalization_steps,
      const std::vector<Kokkos::Array<index_t, 2>>& W_mu_nu_pairs,
      const std::vector<Kokkos::Array<index_t, 2>>& W_Lmu_Lnu_pairs)
      : W_mu_nu_pairs(W_mu_nu_pairs),
        W_Lmu_Lnu_pairs(W_Lmu_Lnu_pairs),
        IMeasurement<ContextT, std::vector<Kokkos::Array<real_t, 5>>>(
            interval,
            thermalization_steps) {};
  std::vector<Kokkos::Array<real_t, 5>> temp_measurements;
  std::vector<Kokkos::Array<index_t, 2>> W_mu_nu_pairs;
  std::vector<Kokkos::Array<index_t, 2>> W_Lmu_Lnu_pairs;
  std::string name() const override { return "WilsonLoop_mu_nu"; }
  std::string storageType() const override { return "Kokkos_Array_5_real_t"; }
  int MPITag() const override {
    return MPI_GAUGE_OBSERVABLES_WILSON_LOOP_MU_NU;
  }

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
    }
    this->add_measurement(context.step, temp_measurements);
    temp_measurements.clear();
  }
};

template <typename ContextT>
class TopologicalChargeMeasurment : public IMeasurement<ContextT, real_t> {
  using IMeasurement<ContextT, real_t>::IMeasurement;
  std::string name() const override { return "TopologicalCharge"; }
  std::string storageType() const override { return "real_t"; }
  int MPITag() const override {
    return MPI_GAUGE_OBSERVABLES_TOPOLOGICAL_CHARGE;
  }

  void measure_impl(ContextT& context) override {
    real_t TopologicalCharge;
    TopologicalCharge =
        get_topological_charge<typename ContextT::AbstractGaugeFieldType>(
            context.get_flowed_gaugeField());
    this->add_measurement(context.step, TopologicalCharge);
  }
};
template <typename ContextT>
class SpMaxMeasurment : public IMeasurement<ContextT, real_t> {
  using IMeasurement<ContextT, real_t>::IMeasurement;
  std::string name() const override { return "Sp_max"; }
  std::string storageType() const override { return "real_t"; }
  int MPITag() const override { return MPI_GAUGE_OBSERVABLES_SP_MAX; }

  void measure_impl(ContextT& context) override {
    real_t SP_max;
    SP_max = get_spmax<typename ContextT::AbstractGaugeFieldType>(
        context.get_flowed_gaugeField());
    this->add_measurement(context.step, SP_max);
  }
};
template <typename ContextT>
class ActionDensityMeasurment : public IMeasurement<ContextT, real_t> {
  using IMeasurement<ContextT, real_t>::IMeasurement;
  int MPITag() const override { return MPI_GAUGE_OBSERVABLES_ACTION_DENSITY; }

  std::string name() const override { return "ActionDensity"; }
  std::string storageType() const override { return "real_t"; }
  void measure_impl(ContextT& context) override {
    real_t Density_E;
    Density_E =
        getActionDensity_clover<typename ContextT::AbstractGaugeFieldType>(
            context.get_flowed_gaugeField());
    this->add_measurement(context.step, Density_E);
  }
};

class AcceptRateMeasurement
    : public IMeasurement<MeasuremntIOContext, Kokkos::Array<real_t, 2>> {
  using IMeasurement<MeasuremntIOContext,
                     Kokkos::Array<real_t, 2>>::IMeasurement;
  std::string name() const override { return "Acceptance,accept"; }
  std::string storageType() const override { return "Kokkos_Array_2_real_t"; }

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
  using IMeasurement<MeasuremntIOContext, real_t>::IMeasurement;
  std::string name() const override { return "LogTime"; }
  std::string storageType() const override { return "real_t"; }

  void measure_impl(MeasuremntIOContext& context) override {
    this->add_measurement(context.step, context.time);
  }
};
class ObsTimeMeasurement : public IMeasurement<MeasuremntIOContext, real_t> {
  using IMeasurement<MeasuremntIOContext, real_t>::IMeasurement;
  std::string name() const override { return "ObsTime"; }
  std::string storageType() const override { return "real_t"; }

  void measure_impl(MeasuremntIOContext& context) override {
    this->add_measurement(context.step, context.obs_time);
  }
};
class DeltaHMeasurement : public IMeasurement<MeasuremntIOContext, real_t> {
  using IMeasurement<MeasuremntIOContext, real_t>::IMeasurement;
  std::string name() const override { return "DeltaH"; }
  std::string storageType() const override { return "real_t"; }

  void measure_impl(MeasuremntIOContext& context) override {
    this->add_measurement(context.step, context.deltaH);
  }
};
}  // namespace klft
