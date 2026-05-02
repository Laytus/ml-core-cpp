#include "phase1_math_sanity.hpp"
#include "phase2_evaluation_sanity.hpp"
#include "phase3_linear_models_sanity.hpp"

int main() {
    ml::experiments::run_phase1_math_sanity();
    ml::experiments::run_phase2_evaluation_sanity();
    ml::experiments::run_phase3_linear_models_sanity();

    return 0;
}