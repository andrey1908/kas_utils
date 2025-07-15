#pragma once

#include "kas_utils/collection.hpp"

#include <map>
#include <utility>
#include <type_traits>
#include <sstream>
#include <iomanip>

namespace kas_utils {

struct DefaultTimeMeasurerIndexType {};

template<typename INDEX_TYPE>
struct CollectionIndexType {
  typedef typename std::conditional<std::is_same<INDEX_TYPE, DefaultTimeMeasurerIndexType>::value, double, INDEX_TYPE>::type type;
};

template<typename INDEX_TYPE = DefaultTimeMeasurerIndexType>
class TimeMeasurer : Collection<std::pair<typename CollectionIndexType<INDEX_TYPE>::type, double>> /* index, time */ {
public:
  TimeMeasurer(const std::string& name,
      bool print_results_on_destruction = false) :
          Collection<std::pair<typename CollectionIndexType<INDEX_TYPE>::type, double>>(name, "time_measurers", "TM",
              std::bind(&TimeMeasurer<INDEX_TYPE>::printSummary, this, std::placeholders::_1, std::placeholders::_2),
              nullptr, observationToOut),
          print_results_on_destruction_(print_results_on_destruction) {}

  void start() {
    static_assert(std::is_same<INDEX_TYPE, DefaultTimeMeasurerIndexType>::value);
    start(DefaultTimeMeasurerIndexType());
  }

  void start(INDEX_TYPE index) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& [start_index, start_time] = start_indices_times_[pthread_self()];
    if constexpr(std::is_same<INDEX_TYPE, DefaultTimeMeasurerIndexType>::value) {
      start_index = toSeconds(std::chrono::system_clock::now().time_since_epoch());
    } else {
      start_index = index;
    }
    start_time = std::chrono::steady_clock::now();
  }

  void stop() {
    std::lock_guard<std::mutex> lock(mutex_);
    auto stop_time = std::chrono::steady_clock::now();
    const auto& [start_index, start_time] = start_indices_times_.at(pthread_self());
    double time = toSeconds(stop_time - start_time);
    this->add(std::make_pair(start_index, time));
  }

private:
  static void observationToOut(std::ostream& out,
      const std::pair<typename CollectionIndexType<INDEX_TYPE>::type, double>& index_time) {
    auto index = index_time.first;
    double time = index_time.second;
    out << std::fixed << std::setprecision(6) << index << ' ' << time;
  }

  void printSummary(const std::string& name,
      const std::vector<std::pair<typename CollectionIndexType<INDEX_TYPE>::type, double>>& observations) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!print_results_on_destruction_) {
      return;
    }
    if (observations.size() > 0) {
      double total_measured_time = 0.;
      double avarage_time = 0.;
      double max_time = 0;
      for (const auto& observation : observations) {
        double time = observation.second;
        total_measured_time += time;
        max_time = std::max(max_time, time);
      }
      avarage_time = total_measured_time / observations.size();

      int number_of_threads = start_indices_times_.size();

      std::string log_string;
      log_string += name + ":\n";
      log_string += "    Number of measurements: " + std::to_string(observations.size()) + "\n";
      log_string += "    Total measured time: " + std::to_string(total_measured_time) + "\n";
      log_string += "    Average time: " + std::to_string(avarage_time) + "\n";
      log_string += "    Max time: " + std::to_string(max_time) + "\n";
      log_string += "    Number of threads: " + std::to_string(number_of_threads) + "\n";

      std::cout << log_string;
    }
  }

private:
  std::mutex mutex_;
  bool print_results_on_destruction_;

  std::map<pthread_t, std::pair<
      typename CollectionIndexType<INDEX_TYPE>::type,
      std::chrono::time_point<std::chrono::steady_clock>>> start_indices_times_;
};

}

#define EXPAND(x) x
#define GET_MACRO(_1, _2, name, ...) name
#define MEASURE_TIME_FROM_HERE(...) \
  EXPAND( GET_MACRO(__VA_ARGS__, MEASURE_TIME_FROM_HERE_WITH_INDEX, MEASURE_TIME_FROM_HERE_DEFAULT)(__VA_ARGS__) )
#define MEASURE_BLOCK_TIME(...) \
  EXPAND( GET_MACRO(__VA_ARGS__, MEASURE_BLOCK_TIME_WITH_INDEX, MEASURE_BLOCK_TIME_DEFAULT)(__VA_ARGS__) )

#define MEASURE_TIME_FROM_HERE_DEFAULT(name) \
  static kas_utils::TimeMeasurer (time_measurer_ ## name)(#name, true); \
  (time_measurer_ ## name).start()

#define MEASURE_TIME_FROM_HERE_WITH_INDEX(name, index) \
  static kas_utils::TimeMeasurer<decltype(index)> (time_measurer_ ## name)(#name, true); \
  (time_measurer_ ## name).start(index)

#define STOP_TIME_MEASUREMENT(name) \
  (time_measurer_ ## name).stop()

#define MEASURE_BLOCK_TIME_DEFAULT(name) \
  static kas_utils::TimeMeasurer (time_measurer_ ## name)(#name, true); \
  class time_measurer_stop_trigger_class_ ## name { \
  public: \
    (time_measurer_stop_trigger_class_ ## name)() {}; \
    (~time_measurer_stop_trigger_class_ ## name)() {(time_measurer_ ## name).stop();}; \
  }; \
  time_measurer_stop_trigger_class_ ## name    time_measurer_stop_trigger_ ## name; \
  (time_measurer_ ## name).start()

#define MEASURE_BLOCK_TIME_WITH_INDEX(name, index) \
  static kas_utils::TimeMeasurer<decltype(index)> (time_measurer_ ## name)(#name, true); \
  class time_measurer_stop_trigger_class_ ## name { \
  public: \
    (time_measurer_stop_trigger_class_ ## name)() {}; \
    (~time_measurer_stop_trigger_class_ ## name)() {(time_measurer_ ## name).stop();}; \
  }; \
  time_measurer_stop_trigger_class_ ## name    time_measurer_stop_trigger_ ## name; \
  (time_measurer_ ## name).start(index)
