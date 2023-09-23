#include <cmath>
#include <iostream>
#include <random>
enum class A {
  AA = 0,
  BB = 1,
};
int main() {
  auto n = 100000LL;
  std::default_random_engine e;
  std::uniform_real_distribution<float> ff(-1, 1);
  while (--n) {
    float x = ff(e), y = ff(e);
    float th = std::atan2(y, x);
    if (th <= -M_PI || th >= M_PI) {
      std::cout << "bad" << th << std::endl;
    }
    // std::cout << th << std::endl;
  }

  std::cout << std::floor(0.00000000000001) << std::endl;
  std::cout << static_cast<unsigned int>(std::floor(0.00000000000001))
            << std::endl;
  {
    std::cout << "tset " << std::endl;
    auto aa = static_cast<A>(0.00f);
    auto bb = static_cast<A>(1.00f);
    std::cout << (aa == A::AA) << " " << (bb == A::BB) << std::endl;
  }
}