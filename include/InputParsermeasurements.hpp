#pragma once
#include <yaml-cpp/yaml.h>
#include "MeasurmentManger.hpp"
#include "Measurments.hpp"
#include "WriteManager.hpp"

namespace klft {
template <typename ContextT>
inline int parseInputFile(const std::string& filename,
                          const std::string& output_dir,
                          MeasurementManager<ContextT>& measur,
                          WriteManagerParams& params) {
  // Handler to find default subnode:
  params.output_dir = output_dir;
  try {
    YAML::Node config = YAML::LoadFile(filename);
    YAML::Node list = config["GaugeMeasurements"];
    // Find default values:
    auto fallback_flush = 30;
    auto fallback_thermalization = 100;
    auto fallback_interval = 5;
    for (const auto& node : list) {
      if (node["name"].as<std::string>() == "default") {
        /* code */
        fallback_flush = list["flush"].as<index_t>(30);
        fallback_thermalization = list["thermalization"].as<index_t>(100);
        fallback_interval = list["interval"].as<index_t>(5);
      }
      if (node["name"].as<std::string>() == "IO") {
        params.fm = list["FIleMode"].as<FileMode>(FileMode::On);
        params.cm = list["ConsoleMode"].as<ConsoleMode>(ConsoleMode::On);
        params.base_name = list["base_name"].as<std::string>("");
        params.default_write_interval =
            list["default_write_interval"].as<int>(100);
      }
    }

    // auto fallback_interval = 1;
    // auto fallback_thermalization = 0;
    // auto fallback_flush = 1;
    if (!list || !list.IsSequence())
      throw std::runtime_error("GaugeMeasurements must be a sequence");
    bool flag_wilson_flow = false;
    for (const auto& node : list) {
      auto name = node["name"].as<std::string>();
      auto interval = node["interval"].as<index_t>(fallback_interval);
      auto thermalization = node["thermalization"].as<index_t>(fallback_flush);
      auto flush = node["flush"].as<index_t>(fallback_flush);
      if (name == "plaquette") {
        measur.register_measurement(
            std::make_unique<PlaquetteMeasurement<ContextT>>(interval,
                                                             thermalization),
            flush);
        printf("Registerd %s\n", name.c_str());
      }
      if (name == "topological_charge") {
        measur.register_measurement(
            std::make_unique<TopologicalChargeMeasurment<ContextT>>(
                interval, thermalization),
            flush);
        printf("Registerd %s\n", name.c_str());
        flag_wilson_flow = true;
      }
      if (name == "measure_action_density") {
        measur.register_measurement(
            std::make_unique<ActionDensityMeasurment<ContextT>>(interval,
                                                                thermalization),
            flush);
        printf("Registerd %s\n", name.c_str());
        flag_wilson_flow = true;
      }
      if (name == "measure_sp_max") {
        measur.register_measurement(std::make_unique<SpMaxMeasurment<ContextT>>(
                                        interval, thermalization),
                                    flush);
        printf("Registerd %s\n", name.c_str());
        flag_wilson_flow = true;
      }
      if (name == "wilson_loop_temporal") {
        YAML::Node arr = node["W_temp_L_T_pairs"];
        std::vector<Kokkos::Array<index_t, 2>> p;
        if (arr && arr.IsSequence()) {
          for (const auto& pairNode : arr) {
            if (!pairNode.IsSequence() || pairNode.size() != 2) {
              throw std::runtime_error(
                  "Each W_temp_L_T_pairs entry must be [L, T].");
            }
            p.emplace_back(IndexArray<2>(
                {pairNode[0].as<index_t>(), pairNode[1].as<index_t>()}));
          }
        }
        measur.register_measurement(
            std::make_unique<WilsonLoopTemporalMeasurement<ContextT>>(
                interval, thermalization, p),
            flush);
        // std::cout << p;
        printf("Registerd %s\n", name.c_str());
      }
      if (name == "wilson_loop_mu_nu") {
        YAML::Node arr = node["W_mu_nu_pairs"];
        std::vector<Kokkos::Array<index_t, 2>> W_mu_nu_pairs;
        if (arr && arr.IsSequence()) {
          for (const auto& pairNode : arr) {
            if (!pairNode.IsSequence() || pairNode.size() != 2) {
              throw std::runtime_error(
                  "Each W_mu_nu_pairs entry must be [L, T].");
            }
            W_mu_nu_pairs.emplace_back(IndexArray<2>(
                {pairNode[0].as<index_t>(), pairNode[1].as<index_t>()}));
          }
        }
        YAML::Node arr1 = node["W_Lmu_Lnu_pairs"];
        std::vector<Kokkos::Array<index_t, 2>> W_Lmu_Lnu_pairs;
        if (arr1 && arr1.IsSequence()) {
          for (const auto& pairNode : arr1) {
            if (!pairNode.IsSequence() || pairNode.size() != 2) {
              throw std::runtime_error(
                  "Each W_Lmu_Lnu_pairs entry must be [L, T].");
            }
            W_Lmu_Lnu_pairs.emplace_back(IndexArray<2>(
                {pairNode[0].as<index_t>(), pairNode[1].as<index_t>()}));
          }
        }
        measur.register_measurement(
            std::make_unique<WilsonLoop_mu_nuMeasurement<ContextT>>(
                interval, thermalization, W_mu_nu_pairs, W_Lmu_Lnu_pairs),
            flush);
        // std::cout << p;
        printf("Registerd %s\n", name.c_str());
      }
    }
  } catch (const std::exception& e) {
    printf("(NewGaugeObservableParams) Error parsing input file: %s\n",
           e.what());
    return 0;
  }
  return 1;
}
}  // namespace klft

namespace YAML {
template <>
struct convert<klft::FileMode> {
  static Node encode(const klft::FileMode& rhs) {
    switch (rhs) {
      case klft::FileMode::Off:
        return Node("Off");
      case klft::FileMode::On:
        return Node("On");
    }
    return Node();
  }

  static bool decode(const Node& node, klft::FileMode& rhs) {
    if (!node.IsScalar())
      return false;

    const std::string s = node.Scalar();
    if (s == "Off") {
      rhs = klft::FileMode::Off;
      return true;
    }
    if (s == "On") {
      rhs = klft::FileMode::On;
      return true;
    }

    return false;
  }
};
template <>
struct convert<klft::ConsoleMode> {
  static Node encode(const klft::ConsoleMode& rhs) {
    switch (rhs) {
      case klft::ConsoleMode::Off:
        return Node("Off");
      case klft::ConsoleMode::On:
        return Node("On");
    }
    return Node();
  }

  static bool decode(const Node& node, klft::ConsoleMode& rhs) {
    if (!node.IsScalar())
      return false;

    const std::string s = node.Scalar();
    if (s == "Off") {
      rhs = klft::ConsoleMode::Off;
      return true;
    }
    if (s == "On") {
      rhs = klft::ConsoleMode::On;
      return true;
    }

    return false;
  }
};
}  // namespace YAML
