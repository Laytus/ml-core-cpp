#pragma once

#include <stdexcept>
#include <string>

namespace ml {

enum class RegularizationType {
    None,
    Ridge,
    Lasso
};

struct RegularizationConfig {
    RegularizationType type{RegularizationType::None};
    double lambda{0.0};

    static RegularizationConfig none() {
        return RegularizationConfig{
            RegularizationType::None,
            0.0
        };
    }

    static RegularizationConfig ridge(double lambda_value) {
        if (lambda_value < 0.0) {
            throw std::invalid_argument(
                "RegularizationConfig::ridge: lambda must be non-negative"
            );
        }

        return RegularizationConfig{
            RegularizationType::Ridge,
            lambda_value
        };
    }

    static RegularizationConfig lasso(double lambda_value) {
        if (lambda_value < 0.0) {
            throw std::invalid_argument(
                "RegularizationConfig::lasso: lambda must be non-negative"
            );
        }

        return RegularizationConfig{
            RegularizationType::Lasso,
            lambda_value
        };
    }

    bool is_enabled() const {
        return type != RegularizationType::None && lambda > 0.0;
    }

    bool is_ridge() const {
        return type == RegularizationType::Ridge && lambda > 0.0;
    }

    bool is_lasso() const {
        return type == RegularizationType::Lasso && lambda > 0.0;
    }
};

inline std::string regularization_type_name(RegularizationType type) {
    switch (type) {
        case RegularizationType::None:
            return "none";
        case RegularizationType::Ridge:
            return "ridge";
        case RegularizationType::Lasso:
            return "lasso";
    }

    return "unknown";
}

}  // namespace ml