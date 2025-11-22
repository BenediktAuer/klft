#include "FieldTypeHelper.hpp"
#include "GLOBAL.hpp"
#include "MeasurmentContext.hpp"
#include "Measurments.hpp"

namespace klft {
template <typename ContextT>
class MeasurementManager {
 public:
  using MeasPtr = std::unique_ptr<IMeasurementBase<ContextT>>;

  void register_measurment(MeasPtr m) {
    measurements.emplace_back(std::move(m));
  }

  void measure(ContextT& ctx) {
    for (auto& m : measurements) {
      m->measure(ctx);
    }
    ctx.increase_step();
  }
  //   void flush(Writer& writer) { writer.flush(); }

 private:
  std::vector<MeasPtr> measurements;
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
