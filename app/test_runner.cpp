#include <Eigen/Dense>
#include <iostream>

namespace {

void run_eigen_sanity_check() {
    Eigen::VectorXd v(3);
    v << 1.0, 2.0, 3.0;

    Eigen::MatrixXd m(3, 3);
    m << 1.0, 0.0, 0.0,
         0.0, 1.0, 0.0,
         0.0, 0.0, 1.0;

    Eigen::VectorXd result = m * v;

    std::cout << "Eigen sanity check:\n";
    std::cout << result << "\n";
}

}  // namespace

int main() {
    run_eigen_sanity_check();
    return 0;
}