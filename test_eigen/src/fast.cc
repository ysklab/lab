#include <Eigen/Core>
#include <vector>
#include <iostream>
#include <chrono>
#include <random>

void Mult(const std::vector<std::vector<double>>& mat1, const std::vector<std::vector<double>>& mat2, std::vector<std::vector<double>>* res) {
    assert(mat1.front().size() == mat2.size());
    for (int r = 0; r < mat1.size(); ++r) {
        for (int c = 0; c < mat2.front().size(); ++c) {
            double sum = 0.0;
            for (int k = 0; k < mat1.front().size(); ++k) {
                sum += mat1[r][k] * mat2[k][c];
            }
            (*res)[r][c] = sum;
        }
    }
    double ss = 0.0;
    for (int i = 0; i< res->size(); ++i) {
        for (int j = 0; j < res->front().size(); ++j) {
            ss+= (*res)[i][j];
        }
    }
    std::cout << ss << std::endl;
}
void Mult(const Eigen::MatrixXd& mat1, const Eigen::MatrixXd& mat2, Eigen::MatrixXd* res) {
    (*res) = mat1 * mat2;
    double ss = 0.0;
    for (int i = 0; i< res->rows(); ++i) {
        for (int j = 0; j < res->cols(); ++j) {
            ss+= (*res)(i, j);
        }
    }
    std::cout << ss << std::endl;
}
int main() {
    int N = 1000;
    using namespace std::chrono;
    std::chrono::high_resolution_clock cc;
    std::default_random_engine e;
    std::uniform_real_distribution<double> rand;

    std::vector<std::vector<double>> m1(N, std::vector<double>(N, rand(e)));
    auto m2 = m1, m3 = m1;

    Eigen::MatrixXd mm1(N,N), mm2(N,N), mm3(N,N);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            mm1(i,j) = m1[i][j];
            mm2(i,j) = m2[i][j];
        }
    }

    auto tic = cc.now();
    Mult(m1, m2, &m3);
    auto toc = cc.now();
    std::cout << duration_cast<std::chrono::microseconds>(toc-tic).count() << " micro" << std::endl;
    tic = cc.now();
    Mult(mm1, mm2, &mm3);
    toc = cc.now();
    std::cout << duration_cast<std::chrono::microseconds>(toc-tic).count() << " micro" << std::endl;
}

