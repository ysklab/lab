#include <stdio.h>
#include <iostream>

class P2D {
public:
// P2D() : P2D {20.0, 30.0} {
//
//    printf("none param %p \n", this);
// }

   P2D() {
      P2D(20.0, 30.0).show();
      printf("none param %p \n", this);
   }

   P2D(float _x, float _y) : x {_x}, y {_y} {
      
   }

   void show() {
      printf(" has param %p \n", this);
      std::cout << " x: " << x << "  y: " << y << std::endl;
   }

public:
   float x;
   float y;
};

int main(int argc, char** argv) {
   P2D p;
   p.show();

// printf(" main  %p \n", &p);

   return 0;
}