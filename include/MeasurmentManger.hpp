#pragma once
#include "FieldTypeHelper.hpp"
#include "GLOBAL.hpp"
#include "MeasurmentContext.hpp"
#include "Measurments.hpp"

namespace klft {
template <typename ContextT>
class MeasurementManager {
 public:
  using MeasPtr = std::shared_ptr<IMeasurementBase<ContextT>>;
  //   void attach_writer(std::shared_ptr<WriterManager<ContextT>>
  //   writer_manager) {
  //     this->writer_manager = writer_manager;

  //     // Register all existing measurements with the writer
  //     if (auto writer_mgr = writer_manager_.lock()) {
  //       for (auto& meas : measurements) {
  //         writer_mgr->register_existing_measurement(meas);
  //       }
  //     }
  //   }
  void register_measurement(std::shared_ptr<IMeasurementBase<ContextT>> m,
                            int write_interval = -1) {  // -1 means use default
    measurements.emplace_back(m);

    // Auto-register with writer if available

    // Store custom interval for when writer is attached
    pending_intervals[m->name()] = write_interval;
    // std::sort(measurements.begin(), measurements.end());
  }

  void measure(ContextT& ctx) {
    for (auto& m : measurements) {
      m->measure(ctx);
    }
    ctx.increase_step();
  }
  // Allow writer to access pending intervals
  const std::map<std::string, int>& get_pending_intervals() const {
    return pending_intervals;
  }

  void clear_pending_intervals() { pending_intervals.clear(); }
  std::vector<MeasPtr> getMeasurments() { return measurements; }

 private:
  std::vector<MeasPtr> measurements;
  //   std::weak_ptr<WriterManager<ContextT>> writer_manager;
  std::map<std::string, int> pending_intervals;
};
class SimLogMeasurmentManager : MeasurementManager<MeasuremntIOContext> {
 private:
  std::string file;
  int interval;
  // Function for flushing to std::cout or done via Writer
 public:
  SimLogMeasurmentManager(const std::string& filename, const int& interval)
      : file(filename), interval(interval) {};
  ~SimLogMeasurmentManager();
};

}  // namespace klft
