project(test_scope)

add_executable(main  main1.cc main2.cc main.cc)

# each .a has same xx
add_library(main1 STATIC main1.cc)
add_library(main2 STATIC main2.cc)
add_executable(mainkk main.cc)
target_link_libraries(mainkk main1 main2)
