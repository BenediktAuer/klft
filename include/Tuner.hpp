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

// A very rudimentary tuning setup for tiling
// in Kokkos MDRangePolicy
// This is a work in progress, currently only works well
// for 4D fields and really does some kind of tuning only
// in cudaspace

#pragma once
#include <fstream>
#include "GLOBAL.hpp"

// define how many times the kernel is run to tune
#ifndef STREAM_NTIMES
#define STREAM_NTIMES 20
#endif

namespace klft {

// need to hash the functor to get a unique id
// to store tuned tiling so that tuning is not repeated
// over multiple kernel calls
template <class FunctorType>
size_t get_Functor_hash(const FunctorType& functor) {
  // this is a very naive hash function
  // should be replaced with a better one
  return std::hash<const void*>{}(&functor);
}

// define a hash table to look up the tuned tiling
// this is a very naive hash table
// should be replaced with a better one
template <size_t rank>
struct TuningHashTable {
  std::unordered_map<std::string, IndexArray<rank>> table;
  void insert(const std::string key, const IndexArray<rank>& value) {
    table[key] = value;
  }
  IndexArray<rank> get(const std::string key) { return table[key]; }
  bool contains(const std::string key) {
    return table.find(key) != table.end();
  }
  void clear() { table.clear(); }
};

// create global 4D, 3D and 2D hash tables
inline TuningHashTable<4> tuning_hash_table_4D;
inline TuningHashTable<3> tuning_hash_table_3D;
inline TuningHashTable<2> tuning_hash_table_2D;

template <size_t rank, class FunctorType, class WorkTag = void>
void tune_and_launch_for(std::string functor_id,
                         const IndexArray<rank>& start,
                         const IndexArray<rank>& end,
                         const FunctorType& functor) {
  // launch kernel if tuning is disabled
  if (!KLFT_TUNING) {
    const auto policy = Policy<rank, WorkTag>(start, end);
    Kokkos::parallel_for(policy, functor);
    return;
  }
  // create a unique string for the kernel
  std::string start_uid = "";
  std::string end_uid = "";
  for (index_t i = 0; i < rank; i++) {
    start_uid += std::to_string(start[i]) + "_";
    end_uid += std::to_string(end[i]) + "_";
  }
  const std::string functor_uid = functor_id + "_rank_" + std::to_string(rank) +
                                  "_start_" + start_uid + "end_" + end_uid;
  if constexpr (rank == 4) {
    if (tuning_hash_table_4D.contains(functor_uid)) {
      auto tiling = tuning_hash_table_4D.get(functor_uid);
      if (KLFT_VERBOSITY > 2) {
        printf("Tuning found for kernel %s, tiling: %d %d %d %d\n",
               functor_uid.c_str(), tiling[0], tiling[1], tiling[2], tiling[3]);
      }
      auto policy = Policy<rank, WorkTag>(start, end, tiling);
      Kokkos::parallel_for(policy, functor);
      return;
    }
  } else if constexpr (rank == 3) {
    if (tuning_hash_table_3D.contains(functor_uid)) {
      auto tiling = tuning_hash_table_3D.get(functor_uid);
      if (KLFT_VERBOSITY > 2) {
        printf("Tuning found for kernel %s, tiling: %d %d %d %d\n",
               functor_uid.c_str(), tiling[0], tiling[1], tiling[2], tiling[3]);
      }
      auto policy = Policy<rank, WorkTag>(start, end, tiling);
      Kokkos::parallel_for(policy, functor);
      return;
    }
  } else if constexpr (rank == 2) {
    if (tuning_hash_table_2D.contains(functor_uid)) {
      auto tiling = tuning_hash_table_2D.get(functor_uid);
      if (KLFT_VERBOSITY > 2) {
        printf("Tuning found for kernel %s, tiling: %d %d %d %d\n",
               functor_uid.c_str(), tiling[0], tiling[1], tiling[2], tiling[3]);
      }
      auto policy = Policy<rank, WorkTag>(start, end, tiling);
      Kokkos::parallel_for(policy, functor);
      return;
    }
  } else {
    // unsupported rank
    printf("Error: unsupported rank %zu\n", rank);
    return;
  }
  if (KLFT_VERBOSITY > 2) {
    printf("Start tuning for kernel %s\n", functor_uid.c_str());
  }
  // if not tuned, tune the functor
  const auto policy = Policy<rank, WorkTag>(start, end);
  IndexArray<rank> best_tiling;
  for (index_t i = 0; i < rank; i++) {
    best_tiling[i] = 1;
  }
  // timer for tuning
  Kokkos::Timer timer;
  double best_time = std::numeric_limits<double>::max();
  // first for hostspace
  // there is no tuning
  if constexpr (std::is_same_v<typename Kokkos::DefaultExecutionSpace,
                               Kokkos::DefaultHostExecutionSpace>) {
    // for OpenMP we parallelise over the two outermost (leftmost) dimensions
    // and so the chunk size for the innermost dimensions corresponds to the
    // view extents
    best_tiling[rank - 1] = end[rank - 1] - start[rank - 1];
    best_tiling[rank - 2] = end[rank - 2] - start[rank - 2];
  } else {
    // for Cuda we need to tune the tiling
    const auto max_tile = policy.max_total_tile_size() / 2;
    IndexArray<rank> current_tiling;
    IndexArray<rank> tile_one;
    for (index_t i = 0; i < rank; i++) {
      current_tiling[i] = 1;
      best_tiling[i] = 1;
      tile_one[i] = 1;
    }
    std::vector<index_t> fast_ind_tiles;
    index_t fast_ind = max_tile;
    while (fast_ind > 2) {
      fast_ind = fast_ind / 2;
      fast_ind_tiles.push_back(fast_ind);
    }
    for (auto& tile : fast_ind_tiles) {
      current_tiling = tile_one;
      current_tiling[0] = tile;
      index_t second_tile = max_tile / tile;
      while (second_tile > 1) {
        current_tiling[1] = second_tile;
        if (max_tile / tile / second_tile >= 4) {
          for (index_t i : {2, 1}) {
            current_tiling[2] = i;
            current_tiling[3] = i;
            auto tune_policy =
                Policy<rank, WorkTag>(start, end, current_tiling);
            double min_time = std::numeric_limits<double>::max();
            for (int ii = 0; ii < STREAM_NTIMES; ii++) {
              timer.reset();
              Kokkos::parallel_for(tune_policy, functor);
              Kokkos::fence();
              min_time = std::min(min_time, timer.seconds());
            }
            if (min_time < best_time) {
              best_time = min_time;
              best_tiling = current_tiling;
            }
            if (KLFT_VERBOSITY > 2) {
              printf("Current Tile size: %d %d %d %d, time: %11.4e\n",
                     current_tiling[0], current_tiling[1], current_tiling[2],
                     current_tiling[3], min_time);
            }
          }
        } else if (max_tile / tile / second_tile == 2) {
          for (int64_t i : {2, 1}) {
            current_tiling[2] = i;
            current_tiling[3] = 1;
            auto tune_policy =
                Policy<rank, WorkTag>(start, end, current_tiling);
            double min_time = std::numeric_limits<double>::max();
            for (int ii = 0; ii < STREAM_NTIMES; ii++) {
              timer.reset();
              Kokkos::parallel_for(tune_policy, functor);
              Kokkos::fence();
              min_time = std::min(min_time, timer.seconds());
            }
            if (min_time < best_time) {
              best_time = min_time;
              best_tiling = current_tiling;
            }
            if (KLFT_VERBOSITY > 2) {
              printf("Current Tile size: %d %d %d %d, time: %11.4e\n",
                     current_tiling[0], current_tiling[1], current_tiling[2],
                     current_tiling[3], min_time);
            }
          }
        } else {
          current_tiling[2] = 1;
          current_tiling[3] = 1;
          auto tune_policy = Policy<rank, WorkTag>(start, end, current_tiling);
          double min_time = std::numeric_limits<double>::max();
          for (int ii = 0; ii < STREAM_NTIMES; ii++) {
            timer.reset();
            Kokkos::parallel_for(tune_policy, functor);
            Kokkos::fence();
            min_time = std::min(min_time, timer.seconds());
          }
          if (min_time < best_time) {
            best_time = min_time;
            best_tiling = current_tiling;
          }
          if (KLFT_VERBOSITY > 2) {
            printf("Current Tile size: %d %d %d %d, time: %11.4e\n",
                   current_tiling[0], current_tiling[1], current_tiling[2],
                   current_tiling[3], min_time);
          }
        }
        second_tile = second_tile / 2;
      }
    }
  }
  if (KLFT_VERBOSITY > 2) {
    printf("Best Tile size: %d %d %d %d\n", best_tiling[0], best_tiling[1],
           best_tiling[2], best_tiling[3]);
    printf("Best Time: %11.4e s\n", best_time);
  }
  // store the best tiling in the hash table
  if constexpr (rank == 4) {
    tuning_hash_table_4D.insert(functor_uid, best_tiling);
  } else if constexpr (rank == 3) {
    tuning_hash_table_3D.insert(functor_uid, best_tiling);
  } else if constexpr (rank == 2) {
    tuning_hash_table_2D.insert(functor_uid, best_tiling);
  }
  if (KLFT_VERBOSITY > 3) {
    double time_rec = std::numeric_limits<double>::max();
    auto tune_policy = Policy<rank, WorkTag>(start, end);
    for (int ii = 0; ii < STREAM_NTIMES; ii++) {
      timer.reset();
      Kokkos::parallel_for(tune_policy, functor);
      Kokkos::fence();
      time_rec = std::min(time_rec, timer.seconds());
    }
    printf("Time with default tile size: %11.4e s\n", time_rec);
    printf("Speedup: %f\n", time_rec / best_time);
  }
  // run the kernel with the best tiling
  auto tune_policy = Policy<rank, WorkTag>(start, end, best_tiling);
  Kokkos::parallel_for(tune_policy, functor);
  return;
};

// write tune hash table to file
inline void writeTuneCache(std::string cache_file_name) {
  // open file in write mode
  std::ofstream cache_file(cache_file_name, std::ios::out);
  if (!cache_file.is_open()) {
    printf("Error: could not open cache file %s\n", cache_file_name.c_str());
    return;
  }
  // write the hash tables to the file
  for (const auto& entry : tuning_hash_table_4D.table) {
    cache_file << 4 << " " << entry.first << " ";
    for (const auto& value : entry.second) {
      cache_file << value << " ";
    }
    cache_file << "\n";
  }
  for (const auto& entry : tuning_hash_table_3D.table) {
    cache_file << 3 << " " << entry.first << " ";
    for (const auto& value : entry.second) {
      cache_file << value << " ";
    }
    cache_file << "\n";
  }
  for (const auto& entry : tuning_hash_table_2D.table) {
    cache_file << 2 << " " << entry.first << " ";
    for (const auto& value : entry.second) {
      cache_file << value << " ";
    }
    cache_file << "\n";
  }
  // close the file
  cache_file.close();
  if (KLFT_VERBOSITY > 0) {
    printf("Tuning hash table written to %s\n", cache_file_name.c_str());
  }
}

// read tune hash table from file
inline void readTuneCache(std::string cache_file_name) {
  // open file in read mode
  std::ifstream cache_file(cache_file_name);
  if (!cache_file.is_open()) {
    printf("Could not open cache file %s\n", cache_file_name.c_str());
    return;
  }
  // read the hash tables from the file
  std::string line;
  while (std::getline(cache_file, line)) {
    std::istringstream iss(line);
    size_t rank;
    std::string functor_id;
    iss >> rank;
    if (rank == 4) {
      iss >> functor_id;
      IndexArray<4> tiling;
      for (index_t i = 0; i < 4; i++) {
        iss >> tiling[i];
      }
      if (KLFT_VERBOSITY > 2) {
        printf("Tuning found for kernel %s, tiling: %d %d %d %d\n",
               functor_id.c_str(), tiling[0], tiling[1], tiling[2], tiling[3]);
      }
      tuning_hash_table_4D.insert(functor_id, tiling);
    } else if (rank == 3) {
      iss >> functor_id;
      IndexArray<3> tiling;
      for (index_t i = 0; i < 3; i++) {
        iss >> tiling[i];
      }
      if (KLFT_VERBOSITY > 2) {
        printf("Tuning found for kernel %s, tiling: %d %d %d\n",
               functor_id.c_str(), tiling[0], tiling[1], tiling[2]);
      }
      tuning_hash_table_3D.insert(functor_id, tiling);
    } else if (rank == 2) {
      iss >> functor_id;
      IndexArray<2> tiling;
      for (index_t i = 0; i < 2; i++) {
        iss >> tiling[i];
      }
      if (KLFT_VERBOSITY > 2) {
        printf("Tuning found for kernel %s, tiling: %d %d\n",
               functor_id.c_str(), tiling[0], tiling[1]);
      }
      tuning_hash_table_2D.insert(functor_id, tiling);
    } else {
      printf("Error: unsupported rank %zu\n", rank);
    }
  }
  // close the file
  cache_file.close();
  if (KLFT_VERBOSITY > 0) {
    printf("Tuning hash table read from %s\n", cache_file_name.c_str());
  }
}

}  // namespace klft