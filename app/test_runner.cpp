#include "phase1_math_sanity.hpp"
#include "phase2_evaluation_sanity.hpp"

int main() {
    ml::experiments::run_phase1_math_sanity();
    ml::experiments::run_phase2_evaluation_sanity();

    return 0;
}