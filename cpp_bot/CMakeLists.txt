cmake_minimum_required(VERSION 3.20)
project(trading_bot_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.3
)

FetchContent_Declare(
  cpr
  GIT_REPOSITORY https://github.com/libcpr/cpr.git
  GIT_TAG 1.10.5
)

FetchContent_MakeAvailable(nlohmann_json cpr)

find_package(OpenSSL REQUIRED)

add_executable(trading_bot
  src/main.cpp
  src/config.cpp
  src/model.cpp
  src/features.cpp
  src/binance_client.cpp
  src/csv_logger.cpp
)

target_include_directories(trading_bot PRIVATE include)
target_link_libraries(trading_bot PRIVATE cpr::cpr nlohmann_json::nlohmann_json OpenSSL::Crypto)
