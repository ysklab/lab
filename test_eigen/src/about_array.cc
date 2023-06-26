#include <Eigen/Core>
#include <iostream>

int main() {
    Eigen::ArrayXd arr(4);
    arr << 1, 3, -1, -1;
    Eigen::ArrayXd zeros(4);
    zeros.setZero();
    arr = arr.max(zeros); // clap min to zero
    std::cout << arr.transpose() << std::endl;
    std::cout << " count=" <<arr.count() << " sum=" << arr.sum() << std::endl;
}