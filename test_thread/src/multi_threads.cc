#include <thread>
#include <iostream>
#include <ctime>

void F() {
  using namespace std::chrono_literals;
  for (int i = 0; i< 10; ++i) {
    std::cout << " print " << i << std::endl;
    std::this_thread::sleep_for(100ms);
  }
}
inline void Test1() {
  std::thread x(F);
  x.join();
  std::cout << "ready return" << std::endl;
}
// -- //
class A {
 public:
  A() {
    thread_ = std::thread(F);
  }
  ~A() {
    if (thread_.joinable()) thread_.join();
  }
 private:
  std::thread thread_;
};
inline void Test2() {
  A a;
  std::cout << "ready return" << std::endl;
}
inline void Test3() {
  std::thread x(F);
x.detach();
  std::cout << "joinable =" << x.joinable() << std::endl;


}
int main() {
  Test2();
}

