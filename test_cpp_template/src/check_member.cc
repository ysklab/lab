#include <type_traits>
#include <iostream>

// primary template handles types that have no nested ::type member:
template<class, class = void>
struct has_type_member : std::false_type {};
 
// specialization recognizes types that do have a nested ::type member:
template<class T>
struct has_type_member<T, std::void_t<typename T::type>> : std::true_type {};


struct MM { int x; };
int F() {
    MM m;
    if constexpr (std::is_member_function_pointer_v<decltype(&MM::time)>) {
        std::cout << m.time() << std::endl;
    }
}

int main() {
    std::cout << has_type_member<int>::value << std::endl;
}