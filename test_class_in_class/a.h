#include <memory>
class A {
private:
  class B;

private:
  std::unique_ptr<B> p_;
};

class A::B {
  int x;
};