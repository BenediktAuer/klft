#pragma once
#include <exception>
#include "FieldTypeHelper.hpp"
#include "GLOBAL.hpp"
#include "MeasurmentContext.hpp"
#include "Measurments.hpp"

namespace klft {
// forward declaration
template <typename ContextT>
struct IMeasurementVisitor;
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
                            int write_interval = -1) {
    // -1 means use default
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

  std::vector<MeasPtr> measurements;
  //   std::weak_ptr<WriterManager<ContextT>> writer_manager;
  std::map<std::string, int> pending_intervals;
};

class SimLogMeasurmentManager : public MeasurementManager<MeasuremntIOContext> {
 private:
  std::string file;
  int interval;
  // Function for flushing to std::cout or done via Writer
 public:
  SimLogMeasurmentManager(const std::string& filename, const int& interval)
      : file(filename), interval(interval) {};
};

class MPI_Measuremnt_Mismatch : public std::exception {
 private:
  std::string received_name;
  std::string own_name;
  mutable std::string message;  // <-- add this to cache the message
 public:
  MPI_Measuremnt_Mismatch(const std::string& received_name,
                          const std::string& own_name)
      : received_name(received_name), own_name(own_name) {
    message = "between the Receiving(" + own_name + ") and send name (" +
              received_name + ") Measurements Name is a miss match!";
  };
  // Override what() method
  const char* what() const noexcept override {
    return message.c_str();  // <-- safe: message is a member
  }
  std::string get_received_name() { return received_name; }
  std::string get_own_name() { return own_name; }
};
template <typename ContextT>
struct MPISendVisitor : IMeasurementVisitor<ContextT> {
  // MPISendVisitor() : IMeasurementVisitor<ContextT>() {
  //   int rank;
  //   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  //   this->rank = rank;
  // };
  // int rank;
  int receiving_rank = 0;
  int step = 0;
  void set_step(const int& step) { this->step = step; };

  void visit(IMeasurement<ContextT, real_t>& m) override { visit_impl(m); }
  void visit(IMeasurement<ContextT, std::vector<Kokkos::Array<real_t, 3>>>& m)
      override {
    visit_impl<Kokkos::Array<real_t, 3>>(m);
  }
  void visit(IMeasurement<ContextT, std::vector<Kokkos::Array<real_t, 5>>>& m)
      override {
    visit_impl<Kokkos::Array<real_t, 5>>(m);
  }
  void visit(IMeasurement<ContextT, Kokkos::Array<real_t, 2>>& m) override {
    visit_impl<std::vector<Kokkos::Array<real_t, 2>>>(m);
  }
  template <typename T>
  void visit_impl(IMeasurement<ContextT, std::vector<T>>& m) {
    if (m.should_measure(step)) {
      auto& res = m.get_result();
      MPI_Send(m.name().c_str(), m.name().length(), MPI::CHAR, receiving_rank,
               MPI_MEASURMENT_NAME, MPI_COMM_WORLD);
      auto last_measuremnt =
          res.value.back();  // get last element of total vector, it that case
                             // this is a vector itself
      MPI_Send(last_measuremnt.data(), last_measuremnt.size() * sizeof(T),
               MPI_BYTE, receiving_rank, m.MPITag(), MPI_COMM_WORLD);
      // If rank/= 0 clear measurment result
      // if (rank != 0) {
      // m->clear()
      // }
    }
  }
  void visit_impl(IMeasurement<ContextT, real_t>& m) {
    if (m.should_measure(step)) {
      /* code */

      auto& res = m.get_result();
      MPI_Send(m.name().c_str(), m.name().length(), MPI::CHAR, receiving_rank,
               MPI_MEASURMENT_NAME, MPI_COMM_WORLD);
      auto last_measuremnt =
          res.value.back();  // get last element of total vector
      MPI_Send(&last_measuremnt, 1, mpi_real_t(), receiving_rank, m.MPITag(),
               MPI_COMM_WORLD);
      // If rank/= 0 clear measurment result
      // if (rank != 0) {
      // m->clear()
      // }
    }
  }
};
template <typename ContextT>
struct MPIReceiveVisitor : IMeasurementVisitor<ContextT> {
  int sending_rank = 0;
  int step = 0;
  void set_sending_rank(const int& sender) { sending_rank = sender; }

  // make local fields for send and recive rank
  void visit(IMeasurement<ContextT, real_t>& m) override { visit_impl(m); }
  void visit(IMeasurement<ContextT, std::vector<Kokkos::Array<real_t, 3>>>& m)
      override {
    visit_impl<Kokkos::Array<real_t, 3>>(m);
  }
  void visit(IMeasurement<ContextT, std::vector<Kokkos::Array<real_t, 5>>>& m)
      override {
    visit_impl<Kokkos::Array<real_t, 5>>(m);
  }
  void visit(IMeasurement<ContextT, Kokkos::Array<real_t, 2>>& m) override {
    visit_impl<std::vector<Kokkos::Array<real_t, 2>>>(m);
  }
  void set_step(const int& step) { this->step = step; }
  template <typename T>
  void visit_impl(IMeasurement<ContextT, std::vector<T>>& m) {
    // TODO if abfrage ob m gemasuret hat
    if (m.should_measure(step)) {
      auto& res = m.get_result();
      MPI_Status status;
      MPI_Probe(sending_rank, MPI_MEASURMENT_NAME, MPI_COMM_WORLD, &status);
      int l;
      MPI_Get_count(&status, MPI_CHAR, &l);
      char* buf = new char[l];
      MPI_Recv(buf, l, MPI_CHAR, sending_rank, MPI_MEASURMENT_NAME,
               MPI_COMM_WORLD, &status);
      std::string rec_name(buf, l);

      delete[] buf;
      if (rec_name != m.name()) {
        throw MPI_Measuremnt_Mismatch(rec_name, m.name());
      }
      std::vector<T> measure;
      int count = 0;
      MPI_Probe(sending_rank, m.MPITag(), MPI_COMM_WORLD, &status);
      MPI_Get_count(&status, MPI_CHAR, &count);
      count /= sizeof(T);
      measure.resize(count);
      MPI_Recv(measure.data(), count * sizeof(T), MPI_BYTE, sending_rank,
               m.MPITag(), MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      m.add_measurement(this->step, measure);
    }
  }
  void visit_impl(IMeasurement<ContextT, real_t>& m) {
    // TODO if abfrage ob m gemasuret hat

    if (m.should_measure(step)) {
      auto& res = m.get_result();
      MPI_Status status;
      MPI_Probe(sending_rank, MPI_MEASURMENT_NAME, MPI_COMM_WORLD, &status);
      int l;
      MPI_Get_count(&status, MPI_CHAR, &l);
      char* buf = new char[l];
      MPI_Recv(buf, l, MPI_CHAR, sending_rank, MPI_MEASURMENT_NAME,
               MPI_COMM_WORLD, &status);
      std::string rec_name(buf, l);

      delete[] buf;
      if (rec_name != m.name()) {
        throw MPI_Measuremnt_Mismatch(rec_name, m.name());
      }
      real_t measure;
      MPI_Recv(&measure, 1, mpi_real_t(), sending_rank, m.MPITag(),
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      m.add_measurement(this->step, measure);
    }
  }
};

template <typename ContextT>
class MeasurementManagerMPI : public MeasurementManager<ContextT> {
 public:
  MeasurementManagerMPI<ContextT>() : MeasurementManager<ContextT>() {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    this->rank = rank;
  }
  void measure(ContextT& ctx) {
    rec_visitor.set_sending_rank(ctx.getMeasurmentRank());
    rec_visitor.set_step(ctx.step);
    send_visitor.set_step(ctx.step);

    if (rank == ctx.getMeasurmentRank() &&
        rank != 0) {  // this determindes the mpirank, so only
                      // one rank will access the if block
      for (auto& m : this->measurements) {
        m->measure(ctx);
        // printf("Rank %d: Measured %s!\n", rank, m->name().c_str());
        m->accept(send_visitor);
        // printf("Rank %d: Send!\n", rank);
        m->clear();
      }
    }
    if (rank == 0) {
      if (ctx.getMeasurmentRank() != 0) {
        for (auto& m : this->measurements) {
          m->accept(rec_visitor);
        }
        /* code */
      } else {
        for (auto& m : this->measurements) {
          m->measure(ctx);
          // printf("Rank %d: Measured %s!\n", rank, m->name().c_str());
        }
      }
    }

    ctx.increase_step();  // all will update step of context
  }

 private:
  int rank;
  MPIReceiveVisitor<ContextT> rec_visitor;
  MPISendVisitor<ContextT> send_visitor;
};

}  // namespace klft
