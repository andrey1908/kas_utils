#include "kas_utils/collection.hpp"

#include <unistd.h>
#include <iostream>
#include <iomanip>

int main() {
  kas_utils::Collection<double> coll("my_col", "my_group", nullptr, nullptr,
      [](std::ostream& out, double d) {
        out << std::fixed << std::setprecision(6) << d;
      });
  coll.add(1);
  coll.add(12.34);
}
