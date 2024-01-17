#pragma once

#include <yaml-cpp/yaml.h>

namespace kas_utils {

bool mergeYaml(YAML::Node& target, const YAML::Node& source);

}
