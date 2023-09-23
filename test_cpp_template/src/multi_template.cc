#include <iostream>
#include <vector>
#include <sstream>
#include <string>
class A {
    public:
    A(int a, double b, const std::string& s) {
        std::cout << a << " " << b << " " << s << std::endl;
    }
};
template<typename... Args>
void Push(std::vector<A>& arr, Args&&... args) {
    arr.emplace_back(std::forward<Args>(args)...);
}

void Test_Stream_Status() {
  using namespace std;
  istringstream iss(string("12 def "));
  string x, y;
  iss >> x >> y;
  std::cout << x << y << std::endl;
  cout << "as_bool =" << (!!iss) << ", fail=" << iss.fail() << " isgood=" << iss.good() << ", eof=" << iss.eof();
  // good: no error and not meet eof
  // eof: meet eof, nothing to read
}




int main() {
  std::vector<A> arr;
  Push(arr, 10, 3.4, "good");

}