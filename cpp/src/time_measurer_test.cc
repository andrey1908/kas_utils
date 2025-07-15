#include "kas_utils/time_measurer.hpp"

#include <unistd.h>
#include <iostream>

void long_operation(unsigned int n) {
  usleep(n);
}

int main() {
  int num = 10;
  unsigned int operation_duration = 10000;

  kas_utils::TimeMeasurer long_operation_time_measurer_default("long_operation_default", true);
  for (int i = 0; i < num; i++) {
    long_operation_time_measurer_default.start();
    long_operation(operation_duration);
    long_operation_time_measurer_default.stop();
  }

  kas_utils::TimeMeasurer<int> long_operation_time_measurer_int("long_operation_int", true);
  for (int i = 0; i < num; i++) {
    long_operation_time_measurer_int.start(i);
    long_operation(operation_duration);
    long_operation_time_measurer_int.stop();
  }

  for (int i = 0; i < num; i++) {
    MEASURE_BLOCK_TIME(long_operation_macro_default);
    long_operation(operation_duration);
  }

  for (int i = 0; i < num; i++) {
    MEASURE_BLOCK_TIME(long_operation_macro_int, i);
    long_operation(operation_duration);
  }
}
