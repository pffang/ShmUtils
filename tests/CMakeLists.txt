cmake_minimum_required(VERSION 3.10)

add_executable(Test Test.cpp)
add_executable(Server Server.cpp)
add_executable(Client Client.cpp)
#target_link_libraries(Test ShmUtils)