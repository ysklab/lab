test in ubuntu20.04

build-essentials cmake

```bash
apt install liblua5.2-dev
apt install libgoogle-glog-dev
```

相关的知识：
luaL_newstate 创建新lua栈
luaL_openlibs 加载lua库
luaL_loadstring 加载lua code作为一个Function对象
lua_gettop 获得栈顶index，也就是栈size。栈index从栈底开始，栈底是1
lua_type 或者栈元素的类型