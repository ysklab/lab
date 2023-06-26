首先当心代码中因为release优化和eigen中懒惰求值导致的某步骤不运行。

没有任何参数，也没有设置CMAKE——BUILD——TYPE，估计是按照DEBUG的：
16730886 micro
 5785785 micro
然后，设定Realese
999169 micro
161275 micro
然后，CMAKE_CXX_FLAGS_RELEASE中启用O3
897877 micro
161479 micro
然后,关闭O3,启用Ofast
892931 micro
156541 micro
然后，关闭Ofast，启用O3和-march=native
926486 micro
 37927 micro
至于有人说MKL可以上百倍加速，目前未成功验证