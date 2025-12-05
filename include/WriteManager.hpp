#pragma once
#include "MeasurmentManger.hpp"
namespace klft {
enum class FileMode { Off, On };
enum class ConsoleMode { Off, On };
struct WriteManagerParams {
  WriteManagerParams(const std ::string& output_dir)
      : output_dir(output_dir) {};
  FileMode fm = FileMode::On;
  ConsoleMode cm = ConsoleMode::On;
  std::string base_name = "";
  std::string output_dir;
  int default_write_interval = 100;
};

template <typename ContextT>
class IMeasurementVisitor {
 public:
  virtual ~IMeasurementVisitor() = default;
  virtual void visit(IMeasurement<ContextT, real_t>& m) = 0;
  virtual void visit(
      IMeasurement<ContextT, std::vector<Kokkos::Array<real_t, 3>>>& m) = 0;
  virtual void visit(
      IMeasurement<ContextT, std::vector<Kokkos::Array<real_t, 5>>>& m) = 0;
  virtual void visit(IMeasurement<ContextT, Kokkos::Array<real_t, 2>>& m) = 0;
};

template <typename ContextT>
class IMeasurementVisitorIO : public IMeasurementVisitor<ContextT> {
 public:
  std::ostream* out;
  IMeasurementVisitorIO(std::ostream& out) : out(&out) {};

  void set_output(std::ostream& out) { this->out = &out; }
};
template <typename ContextT>
struct PrintResults : IMeasurementVisitorIO<ContextT> {
  PrintResults() : IMeasurementVisitorIO<ContextT>(std::cout) {};
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
  void visit(IMeasurement<ContextT, Kokkos::Array<real_t, 2>>& m) override {
    visit_impl<Kokkos::Array<real_t, 2>>(m);
  }
  template <typename T>
  void visit_impl(IMeasurement<ContextT, T>& m) {
    auto& res = m.get_result();

    *(this->out) << "Measurement " << "Step: " << res.get_step(res.size() - 1)
                 << " " << m.name() << ": " << res(res.size() - 1) << "\n";
  }
  void set_output(std::ostream& out) {};
};
template <typename ContextT>
struct DumpToFile : IMeasurementVisitorIO<ContextT> {
  DumpToFile() : IMeasurementVisitorIO<ContextT>(std::cout) {};
  void visit(IMeasurement<ContextT, real_t>& m) override { visit_impl(m); }
  void visit(IMeasurement<ContextT, std::vector<Kokkos::Array<real_t, 3>>>& m)
      override {
    visit_impl<std::vector<Kokkos::Array<real_t, 3>>>(m);
  }
  void visit(IMeasurement<ContextT, std ::vector<Kokkos::Array<real_t, 5>>>& m)
      override {
    visit_impl<std::vector<Kokkos::Array<real_t, 5>>>(m);
  }
  void visit(IMeasurement<ContextT, Kokkos::Array<real_t, 2>>& m) override {
    visit_impl<Kokkos::Array<real_t, 2>>(m);
  }
  void visit_impl(IMeasurement<ContextT, real_t>& m) {
    auto& res = m.get_result();
    for (size_t i = 0; i < res.size(); i++) {
      auto to_dump = res(i, i);

      *(this->out) << to_dump.first << "," << to_dump.second << "\n";
    }
  }
  template <typename T>
  void visit_impl(IMeasurement<ContextT, T>& m) {
    auto& res = m.get_result();
    for (size_t i = 0; i < res.size(); i++) {
      auto to_dump = res(i, i);
      for (size_t j = 0; j < to_dump.second.size(); j++) {
        *(this->out) << to_dump.first << "," << to_dump.second[j];
      }
    }
  }
};

template <typename ContextT>
struct DumpToSingleFile : IMeasurementVisitorIO<ContextT> {
  int index = 0;
  DumpToSingleFile<ContextT>() : IMeasurementVisitorIO<ContextT>(std::cout) {}
  void visit(IMeasurement<ContextT, real_t>& m) override { visit_impl(m); }
  void visit(IMeasurement<ContextT, std::vector<Kokkos::Array<real_t, 3>>>& m)
      override {
    visit_impl<std::vector<Kokkos::Array<real_t, 3>>>(m);
  }
  void visit(IMeasurement<ContextT, std ::vector<Kokkos::Array<real_t, 5>>>& m)
      override {
    visit_impl<std::vector<Kokkos::Array<real_t, 5>>>(m);
  }
  void visit(IMeasurement<ContextT, Kokkos::Array<real_t, 2>>& m) override {
    visit_impl<Kokkos::Array<real_t, 2>>(m);
  }
  void visit_impl(IMeasurement<ContextT, real_t>& m) {
    auto& res = m.get_result();

    auto to_dump = res(index);

    *(this->out) << "," << to_dump;
  }
  template <typename T>
  void visit_impl(IMeasurement<ContextT, T>& m) {
    auto& res = m.get_result();

    auto to_dump = res(index);
    for (size_t j = 0; j < to_dump.size(); j++) {
      *(this->out) << "," << to_dump[j];
    }
  }

  void increase_step() { index++; }
  void reset_step() { index = 0; }
};
template <typename ContextT>
struct DumpStepToFile : IMeasurementVisitorIO<ContextT> {
  int index = 0;
  DumpStepToFile<ContextT>() : IMeasurementVisitorIO<ContextT>(std::cout) {}
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
  void visit(IMeasurement<ContextT, Kokkos::Array<real_t, 2>>& m) override {
    visit_impl<Kokkos::Array<real_t, 2>>(m);
  }
  template <typename T>
  void visit_impl(IMeasurement<ContextT, T>& m) {
    auto& res = m.get_result();

    *(this->out) << m.get_step(index);
  }

  void increase_step() { index++; }
  void reset_step() { index = 0; }
};

template <typename ContextT>
class WriteManager {
  using MeasPtr = std::shared_ptr<IMeasurementBase<ContextT>>;

 public:
  // Constructor now takes the measurement manager
  WriteManager(FileMode fm,
               ConsoleMode cm,
               const std::string& output_dir,
               const std::string& base_name,
               int default_write_interval = 100)
      : output_dir(output_dir),
        default_write_interval(default_write_interval),
        fm(fm),
        consolemode(cm),
        base_name(base_name) {
          // std::filesystem::create_directories(output_dir);

        };
  WriteManager(const WriteManagerParams& params)
      : fm(params.fm),
        consolemode(params.cm),
        output_dir(params.output_dir),
        default_write_interval(params.default_write_interval),
        base_name(params.base_name) {};
  WriteManager() = default;

  void register_measurments(
      std::shared_ptr<IMeasurementManager<ContextT>> meas_manager,
      const int mpiTag = 0) {
    // Apply pending intervals first
    for (const auto& [name, interval] : meas_manager->get_pending_intervals()) {
      custom_intervals[name] =
          interval <= 0 ? this->default_write_interval : interval;
    }
    meas_manager->clear_pending_intervals();
    measurements = meas_manager->getMeasurments();
    if (1 -
        mpiTag) {  // it dosnt matter wich of the mpi ranks will write the
                   // header, only important that it is done once and only once

      write_header();
    }
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

  void flush(int current_step) {
    for (auto& m : measurements) {
      std::string name = m->name();
      // std::cout << typeid(*m).name() << "\n";
      if (should_print(consolemode, current_step, m->interval,
                       m->measurmentSize())) {
        m->accept(stdOutPrinter);
      }
      if (should_write(name, current_step, m->measurmentSize())) {
        if (fm == FileMode::On) {
          /* code */
          reopen(file, name);

          visitor.set_output(file);
          m->accept(visitor);
          m->clear();
        }
      }
    }
    file.close();
  }

  void flush() {
    for (auto& m : measurements) {
      if (m->measurmentSize() > 0) {
        /* code */
        if (fm == FileMode::On) {
          /* code */
          reopen(file, m->name());

          visitor.set_output(file);
          m->accept(visitor);
          m->clear();
        }
      }
    }
    file.close();
  }

 protected:
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
  virtual bool should_print(const ConsoleMode& mode,
                            const int& current_step,
                            const int& interval,
                            const int& size) {
    return mode == ConsoleMode::On && current_step % interval == 0 &&
           size > 0 && KLFT_VERBOSITY > 1;
  }
  template <typename Stream>
  void reopen(Stream& pStream,
              const std::string& name,

              std::ios::openmode pMode = std::ios::app) {
    if (pStream.is_open()) {
      pStream.close();
    }
    pStream.clear();
    pStream.open(output_dir + base_name + name + ".txt", pMode);
    if (!file.is_open()) {
      printf("Error: could not open log file %s\n", name.c_str());
      return;
    }
  }
  virtual void write_header() {
    for (auto& m : measurements) {
      reopen(file, m->name());
      if (!file.is_open()) {
        printf("Error: could not open log file %s\n", m->name().c_str());
        return;
      }
      file << "step" << "," << m->header() << "\n";
    }
    file.close();
  }

  //   std::weak_ptr<MeasurementManager<ContextT>> meas_manager_;
  FileMode fm;
  ConsoleMode consolemode;
  std::string base_name;

  std::ofstream file;

  std::string output_dir;
  int default_write_interval;
  std::map<std::string, int> custom_intervals;
  std::vector<MeasPtr> measurements;
  DumpToFile<ContextT> visitor;
  PrintResults<ContextT> stdOutPrinter;
};

struct WriteManagerSimLog : public WriteManager<MeasuremntIOContext> {
  WriteManagerSimLog(FileMode fm,
                     ConsoleMode cm,
                     const std::string& output_dir,
                     const std::string& base_name,
                     int default_write_interval = 100)
      : WriteManager<MeasuremntIOContext>(fm,
                                          cm,
                                          output_dir,
                                          base_name,
                                          default_write_interval) {}
  bool should_write(const std::string& name, const int& step, const int& size) {
    auto interval = this->custom_intervals[name];
    if (step % interval == 0 && size > 0) {
      return true;
    }
    return false;
  }
  bool should_print(const ConsoleMode& mode,
                    const int& current_step,
                    const int& interval,
                    const int& size) {
    return (mode == ConsoleMode::On) && (current_step % interval == 0) &&
           (size > 0) && (KLFT_VERBOSITY > 1);
  }
  void write_header() {
    /* code */
    this->reopen(this->file, "");
    this->file << "step";
    for (auto& m : this->measurements) {
      if (!this->file.is_open()) {
        printf("Error: could not open log file %s\n", m->name().c_str());
        return;
      }
      this->file << "," << m->header();
    }
    this->file << "\n";
    this->file.close();
  }
  void flush(int current_step) {
    if (this->measurements.size() == 0) {
      return;
    }
    reopen(file, "");
    singlevisitor.set_output(file);
    stepDump.set_output(file);
    for (int i = 0; i < this->measurements[0]->measurmentSize(); i++) {
      // write first step
      this->measurements[0]->accept(stepDump);

      for (auto& m : measurements) {
        std::string name = m->name();
        // std::cout << typeid(*m).name() << "\n";
        if (should_print(consolemode, current_step, m->interval,
                         m->measurmentSize())) {
          m->accept(stdOutPrinter);
        }
        if (should_write(name, current_step, m->measurmentSize())) {
          if (fm == FileMode::On) {
            /* code */
            // ;

            m->accept(singlevisitor);
            // m->clear();
          }
        }
      }
      (this->file) << "\n";
      stepDump.increase_step();
      singlevisitor.increase_step();
    }
    this->file.close();
    for (auto& m : measurements) {
      std::string name = m->name();
      if (should_write(name, current_step, m->measurmentSize())) {
        if (fm == FileMode::On) {
          m->clear();
          singlevisitor.reset_step();
          stepDump.reset_step();
        }
      }
    }
  }

  DumpToSingleFile<MeasuremntIOContext> singlevisitor;
  DumpStepToFile<MeasuremntIOContext> stepDump;
};

}  // namespace klft
