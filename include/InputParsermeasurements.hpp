#pragma once
#include <yaml-cpp/yaml.h>
#include "MeasurmentManger.hpp"
#include "Measurments.hpp"

namespace klft {
template <typename ContextT>
inline int parseInputFile(const std::string& filename,
                          MeasurementManager<ContextT>& measur) {
  // Handler to find default subnode:

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
