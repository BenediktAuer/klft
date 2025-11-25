#pragma once
#include "MeasurmentManger.hpp"
namespace klft {
// Forward declaration:
template <typename ContextT>
struct IMeasurementVisitor {
  virtual ~IMeasurementVisitor() = default;
  virtual void visit(IMeasurement<ContextT, real_t>& m) = 0;
  virtual void visit(IMeasurement<ContextT, Kokkos::Array<real_t, 3>>& m) = 0;
  virtual void visit(IMeasurement<ContextT, Kokkos::Array<real_t, 5>>& m) = 0;
};
template <typename ContextT>
struct PrintResults : IMeasurementVisitor<ContextT> {
  void visit(IMeasurement<ContextT, real_t>& m) { visit_impl<real_t>(m); }
  void visit(IMeasurement<ContextT, Kokkos::Array<real_t, 3>>& m) {
    visit_impl<Kokkos::Array<real_t, 3>>(m);
  }
  void visit(IMeasurement<ContextT, Kokkos::Array<real_t, 5>>& m) {
    visit_impl<Kokkos::Array<real_t, 5>>(m);
  }
  template <typename T>
  void visit_impl(IMeasurement<ContextT, T>& m) {
    auto& res = m.get_result();

    std::cout << "Measurement " << m.name() << ": " << res(0) << "\n";
  }
};
template <typename ContextT>
class WriterManager {
  using MeasPtr = std::shared_ptr<IMeasurementBase<ContextT>>;

 public:
  // Constructor now takes the measurement manager
  WriterManager(const std::string& output_dir, int default_write_interval = 100)
      : output_dir(output_dir), default_write_interval(default_write_interval) {
    std::filesystem::create_directories(output_dir);
  };

  void register_measurments(MeasurementManager<ContextT>& meas_manager) {
    // Apply pending intervals first
    for (const auto& [name, interval] : meas_manager.get_pending_intervals()) {
      custom_intervals[name] =
          interval <= 0 ? this->default_write_interval : interval;
    }
    meas_manager.clear_pending_intervals();
    measurements = meas_manager.getMeasurments();
  }

  void set_default_interval(int interval) { default_write_interval = interval; }

  void set_write_interval(const std::string& name, int interval) {
    custom_intervals[name] = interval;
  }

  // Called when writer is attached to existing measurements

  void register_measurement(std::shared_ptr<IMeasurementBase<ContextT>> meas) {
    int interval = default_write_interval;

    auto it = custom_intervals.find(meas->name());
    if (it != custom_intervals.end()) {
      interval = it->second;
    }

    // Need to handle type erasure here - create generic writer
    create_writer_for_measurement(meas, interval);
  }

  //   // Register with automatic interval selection
  //   template <typename T>
  //   void register_writer_auto(std::shared_ptr<IMeasurement<ContextT, T>>
  //   meas) {
  //     int interval = default_write_interval_;

  //     auto it = custom_intervals_.find(meas->name());
  //     if (it != custom_intervals_.end()) {
  //       interval = it->second;
  //     }

  //     register_writer(meas, interval);
  //   }

  //   template <typename T>
  //   void register_writer(std::shared_ptr<IMeasurement<ContextT, T>>
  //   measurement,
  //                        int write_interval) {
  //     auto writer = std::make_unique<MeasurementWriter<ContextT, T>>(
  //         measurement, output_dir_, write_interval);
  //     writers_.emplace_back(std::move(writer));
  //   }

  void flush(IMeasurementVisitor<ContextT>& visitor, int current_step) {
    for (auto& m : measurements) {
      std::string name = m->name();
      std::string type = m->storageType();
      // std::cout << typeid(*m).name() << "\n";
      if (should_write(name, current_step)) {
        auto file = get_File(name);
        if (!file.is_open()) {
          printf("Error: could not open log file %s\n", name.c_str());
          return;
        }
        m->accept(visitor);
        file.close();
      }
    }
  }

  void flush(IMeasurementVisitor<ContextT>& visitor) {
    for (auto& m : measurements) {
      m->accept(visitor);
    }
  }

 private:
  //   void create_writer_for_measurement(
  //       std::shared_ptr<IMeasurementBase<ContextT>> meas,
  //       int interval) {
  //     // This requires some type introspection or a helper method
  //     // For now, we'll need measurements to provide a way to create their
  //     writers
  //     // See the enhanced version below for a better solution
  //   }
  bool should_write(const std::string& name, const int& step) {
    auto interval = this->custom_intervals[name];
    if (step % interval == 0) {
      return true;
    }
    return false;
  }
  std::ofstream get_File(const std::string& name) {
    return std::ofstream(output_dir + name + ".txt", std::ios::app);
  }
  void write_header(const std::string& name) {
    auto file = get_File(name);
    if (!file.is_open()) {
      printf("Error: could not open log file %s\n", name.c_str());
      return;
    }
    file << "step" << "," << name;
    file.close();
  }

  //   std::weak_ptr<MeasurementManager<ContextT>> meas_manager_;
  std::string output_dir;
  int default_write_interval;
  std::map<std::string, int> custom_intervals;
  std::vector<MeasPtr> measurements;
};

}  // namespace klft
