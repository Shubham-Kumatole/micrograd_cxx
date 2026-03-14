cmake_minimum_required(VERSION 3.20)
project(micrograd VERSION 1.0 LANGUAGES CXX)
set(Python3_ROOT_DIR "/opt/homebrew/opt/python@3.14")
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Source files - list explicitly to avoid GLOB pitfalls
add_executable(${PROJECT_NAME}
    src/main.cpp
    src/Value.cpp
    src/Neuron.cpp
    src/Layer.cpp
    src/MultiLayerPerceptron.cpp
)
# Your own headers in include/
target_include_directories(${PROJECT_NAME} PRIVATE
${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(micrograd PRIVATE Python3::Python Python3::NumPy)

target_compile_options(${PROJECT_NAME} PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -O2
)
