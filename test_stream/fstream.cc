#include <fstream>
#include <iostream>
int main(int argc, char **argv) {

  using namespace std;
  string str;
  std::ifstream ifs(argv[1]);

  ifs.seekg(0, ios::end);
  cout << ifs.good() << str << endl; // 无输出
  getline(ifs, str);
  cout << ifs.good() << str << endl; // 无输出
  ifs.seekg(0, ios::beg);
  getline(ifs, str);                 // 无输出
  cout << ifs.good() << str << endl; // 无输出:

  {
    std::ifstream ifs(argv[1], std::ios::binary);
    char c;
    ifs >> c;
    std::cout << "get c: inchar=" << c << " in16=" << std::hex << (int)c
              << " , in10=" << std::dec << (int)c << std::endl;
    uint16_t s;
    ifs >> s;
    std::cout << "get s: in10=" << std::dec << s << " in16=" << std::hex << s
              << std::endl;
    ifs >> s;
    std::cout << "get s: in10=" << std::dec << s << " in16=" << std::hex << s
              << std::endl;
    ifs >> s;
    std::cout << "get s: in10=" << std::dec << s << " in16=" << std::hex << s
              << std::endl;
  }
}