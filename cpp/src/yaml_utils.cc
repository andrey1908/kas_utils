#include "kas_utils/yaml_utils.h"

namespace kas_utils {

bool mergeYaml(YAML::Node& target, const YAML::Node& source) {
    if (target.IsNull()) {
        target = YAML::Clone(source);
        return true;
    }

    if (target.Type() != source.Type()) {
        return false;
    }
    switch (target.Type()) {
    case YAML::NodeType::Map:
        for (auto it = source.begin(); it != source.end(); ++it)
        {
            const std::string& key = it->first.as<std::string>();
            const YAML::Node& val = it->second;
            if (!target[key])
            {
                target[key] = YAML::Clone(val);
            }
            else
            {
                YAML::Node next_target = target[key];
                bool ret = mergeYaml(next_target, val);
                if (ret == false) {
                    return false;
                }
            }
        }
        break;
    case YAML::NodeType::Scalar:
    case YAML::NodeType::Sequence:
        target = YAML::Clone(source);
        break;
    default:
        return false;
    }
    return true;
}

}