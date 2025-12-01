#pragma once
#include "MeasurmentManger.hpp"
namespace klft {
// Forward declaration:
template <typename ContextT>
class IMeasurementVisitor {
  std::ostream& out;
  virtual ~IMeasurementVisitor() = default;
  virtual void visit(IMeasurement<ContextT, real_t>& m) = 0;
  virtual void visit(
      IMeasurement<ContextT, std::vector<Kokkos::Array<real_t, 3>>>& m) = 0;
  virtual void visit(
      IMeasurement<ContextT, std::vector<Kokkos::Array<real_t, 5>>>& m) = 0;
  virtual bool writes_to_file() const = 0;
 virtual first_visit()

     protected : void set_output(std::ostream& out) {
       this->out = out
     };
};
template <typename ContextT>
struct PrintResults : IMeasurementVisitor<ContextT> {
  PrintResults() { set_output(std::cout) };
  void visit(IMeasurement<ContextT, real_t>& m) override {
    visit_impl<real_t>(m);
  }
  void visit(IMeasurement<ContextT, std::vector<Kokkos::Array<real_t, 3>>>& m)
      override {
    visit_impl<std::vector<Kokkos::Array<real_t, 3>>>(m);
  }
  void visit(IMeasurement<ContextT, std ::vector<Kokkos::Array<real_t, 5>>>& m)
      override {
    visit_impl<std::vector<Kokkos::Array<real_t, 5>>>(m);
  }
  template <typename T>
  void visit_impl(IMeasurement<ContextT, T>& m) {
    auto& res = m.get_result();

    this->out << "Measurement " << m.name() << ": " << res(0)
              << " Size: " << res.size() << "\n";
  }
  bool writes_to_file() const override { return false; }
};
template <typename ContextT>
struct DumpToFile : IMeasurementVisitor<ContextT> {
  void visit(IMeasurement<ContextT, real_t>& m) override {
    visit_impl<real_t>(m);
  }
  void visit(IMeasurement<ContextT, std::vector<Kokkos::Array<real_t, 3>>>& m)
      override {
    visit_impl<std::vector<Kokkos::Array<real_t, 3>>>(m);
  }
  void visit(IMeasurement<ContextT, std ::vector<Kokkos::Array<real_t, 5>>>& m)
      override {
    visit_impl<std::vector<Kokkos::Array<real_t, 5>>>(m);
  }
  template <typename T>
  void visit_impl(IMeasurement<ContextT, T>& m) {
    auto& res = m.get_result();
    for (size_t i = 0; i < res.size(); i++) {
      auto to_dump = res(i, i);
      out << to_dump.first() << "," << to_dump.second() << "\n";
    }
  }
  void set_output(std::ofstream& out) { this -> out = out };
  bool writes_to_file() const override { return true; }
};

template <typename ContextT>
struct DumpToSingleFile : IMeasurementVisitor<ContextT> {
  DumpToSingleFile<ContextT>(std::ostream& out)
      : IMeasurementVisitor<ContextT>(){set_output(out)};
  void visit(IMeasurement<ContextT, real_t>& m) override {
    visit_impl<real_t>(m);
  }
  void visit(IMeasurement<ContextT, std::vector<Kokkos::Array<real_t, 3>>>& m)
      override {
    visit_impl<std::vector<Kokkos::Array<real_t, 3>>>(m);
  }
  void visit(IMeasurement<ContextT, std ::vector<Kokkos::Array<real_t, 5>>>& m)
      override {
    visit_impl<std::vector<Kokkos::Array<real_t, 5>>>(m);
  }
  template <typename T>
  void visit_impl(IMeasurement<ContextT, T>& m) {
    auto& res = m.get_result();
    for (size_t i = 0; i < res.size(); i++) {
      auto to_dump = res(i, i);
      out << to_dump.first() << "," << to_dump.second();
    }
  }
  void set_output(std::ostream& out) {};  // such that this wont get overwritten
  bool writes_to_file() const override { return false; }  // small hack
};

template <typename ContextT>
class WriterManager {
  using MeasPtr = std::shared_ptr<IMeasurementBase<ContextT>>;

 public:
  enum class FileMode { None, IndividualFiles };
  enum class ConsoleMode { Off, On };

  // Constructor now takes the measurement manager
  WriterManager(FileMode fm,
                ConsoleMode cm,
                const std::string& output_dir,
                const std::string& base_name,
                int default_write_interval = 100)
      : output_dir(output_dir),
        default_write_interval(default_write_interval),
        filemode(fm),
        consolemode(cm),
        base_name(base_name) {
    // std::filesystem::create_directories(output_dir);
    if (filemode == FileMode::IndividualFiles) {
      out = &file;
    } else {
      out = &std::cout;
    }
  };

  void register_measurments(MeasurementManager<ContextT>& meas_manager) {
    // Apply pending intervals first
    for (const auto& [name, interval] : meas_manager.get_pending_intervals()) {
      custom_intervals[name] =
          interval <= 0 ? this->default_write_interval : interval;
    }
    meas_manager.clear_pending_intervals();
    measurements = meas_manager.getMeasurments();
    // write headers
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
      // std::cout << typeid(*m).name() << "\n";
      if (should_write(name, current_step, m->measurmentSize())) {
        reopen(file, name);
        if (!file.is_open()) {
          printf("Error: could not open log file %s\n", name.c_str());
          return;
        }

        visitor.set_output(out) m->accept(visitor);
      }
      file.close();
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
  bool should_write(const std::string& name, const int& step, const int& size) {
    auto interval = this->custom_intervals[name];
    if (step % interval == 0 && size > 0) {
      return true;
    }
    return false;
  }
  template <typename Stream>
  void reopen(Stream& pStream,
              const std::string& name,
              const bool first_touch& = false,
              std::ios::openmode pMode = std::ios::app) {
    if (filemode != FileMode::SingleFile || first_touch) {
      if (pStream.is_open()) {
        pStream.close();
      }
      pStream.clear();
      pStream.open(output_dir + base_name + file + ".txt", pMode);
    }
  }
  void write_header() {
    for (auto& m : measurements) {
      auto file = reopen(file, m->name(), true);
      m->accept(visitor);
      if (!file.is_open()) {
        printf("Error: could not open log file %s\n", name.c_str());
        return;
      }
      file << "step" << "," << m->header();
    }
    file.close()
  }

  //   std::weak_ptr<MeasurementManager<ContextT>> meas_manager_;
  FileMode filemode;
  ConsoleMode consolemode;
  std::string base_name;
  std::ostream* out;
  std::ofstream file;

  std::string output_dir;
  int default_write_interval;
  std::map<std::string, int> custom_intervals;
  std::vector<MeasPtr> measurements;
};

}  // namespace klft
