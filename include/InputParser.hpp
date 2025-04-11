//******************************************************************************/
//
// This file is part of the Kokkos Lattice Field Theory (KLFT) library.
//
// KLFT is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// KLFT is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with KLFT.  If not, see <http://www.gnu.org/licenses/>.
//
//******************************************************************************/

// this file defines input parser for running different simulations
// we are going to overload the the fuction parseInputFile for all
// different Param structs

#pragma once
#include <yaml-cpp/yaml.h>
#include "Metropolis_Params.hpp"
#include "GaugeObservable.hpp"
#include <fstream>

namespace klft {

  // get MetropolisParams from input file
  bool parseInputFile(const std::string &filename, MetropolisParams &metropolisParams) {
    try {
      YAML::Node config = YAML::LoadFile(filename);

      // Parse MetropolisParams
      if (config["MetropolisParams"]) {
        const auto &mp = config["MetropolisParams"];
        // general parameters
        metropolisParams.Ndims = mp["Ndims"].as<index_t>(4);
        metropolisParams.L0 = mp["L0"].as<index_t>(32);
        metropolisParams.L1 = mp["L1"].as<index_t>(32);
        metropolisParams.L2 = mp["L2"].as<index_t>(32);
        metropolisParams.L3 = mp["L3"].as<index_t>(32);
        metropolisParams.nHits = mp["nHits"].as<index_t>(10);
        // parameters specific to the GaugeField
        metropolisParams.Nd = mp["Nd"].as<size_t>(4);
        metropolisParams.Nc = mp["Nc"].as<size_t>(2);
        // parameters specific to the Wilson action
        metropolisParams.beta = mp["beta"].as<double>(1.0);
        metropolisParams.delta = mp["delta"].as<double>(0.1);
        // ...
        // add more parameters above this as needed
      } else {
        printf("Error: MetropolisParams not found in input file\n");
        return false;
      }
      return true;
    } catch (const YAML::Exception &e) {
      printf("Error parsing input file: %s\n", e.what());
      return false;
    }
  }

  // get GaugeObservableParams from input file
  bool parseInputFile(const std::string &filename, GaugeObservableParams &gaugeParams) {
    try {
      YAML::Node config = YAML::LoadFile(filename);

      // Parse GaugeObservableParams
      if (config["GaugeObservableParams"]) {
        const auto &gp = config["GaugeObservableParams"];
        // interval between measurements
        gaugeObservableParams.measurement_interval = gp["measurement_interval"].as<size_t>(0);
        // whether to measure the plaquette
        gaugeObservableParams.measure_plaquette = gp["measure_plaquette"].as<bool>(false);
        // whether to measure the temporal Wilson loop
        gaugeObservableParams.measure_wilson_loop_temporal = gp["measure_wilson_loop_temporal"].as<bool>(false);
        // whether to measure the mu-nu Wilson loop
        gaugeObservableParams.measure_wilson_loop_mu_nu = gp["measure_wilson_loop_mu_nu"].as<bool>(false);

        // pairs of (L,T) for the temporal Wilson loop
        if (gp["W_temp_L_T_pairs"]) {
          for (const auto &pair : gp["W_temp_L_T_pairs"]) {
            gaugeObservableParams.W_temp_L_T_pairs.push_back({pair[0].as<size_t>(), pair[1].as<size_t>()});
          }
        }

        // pairs of (mu,nu) for the mu-nu Wilson loop
        if (gp["W_mu_nu_pairs"]) {
          for (const auto &pair : gp["W_mu_nu_pairs"]) {
            gaugeObservableParams.W_mu_nu_pairs.push_back({pair[0].as<size_t>(), pair[1].as<size_t>()});
          }
        }

        // pairs of (Lmu,Lnu) for the Wilson loop
        if (gp["W_Lmu_Lnu_pairs"]) {
          for (const auto &pair : gp["W_Lmu_Lnu_pairs"]) {
            gaugeObservableParams.W_Lmu_Lnu_pairs.push_back({pair[0].as<size_t>(), pair[1].as<size_t>()});
          }
        }

        // filenames for the measurements
        gaugeObservableParams.plaquette_filename = gp["plaquette_filename"].as<std::string>("");
        gaugeObservableParams.W_temp_filename = gp["W_temp_filename"].as<std::string>("");
        gaugeObservableParams.W_mu_nu_filename = gp["W_mu_nu_filename"].as<std::string>("");
        // ...
        // add more parameters above this line as needed
      } else {
        printf("Error: GaugeObservableParams not found in input file\n");
        return false;
      }
      return true;
    } catch (const YAML::Exception &e) {
      printf("Error parsing input file: %s\n", e.what());
      return false;
    }
  }
}