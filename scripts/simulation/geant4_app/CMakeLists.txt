cmake_minimum_required(VERSION 3.16)
project(muography_sim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Batch-only production app: no UI/vis drivers required.
find_package(Geant4 REQUIRED)
include(${Geant4_USE_FILE})

# Vendored third-party headers (offline builds: HPC compute nodes
# usually have no internet access).
set(THIRD_PARTY_DIR ${PROJECT_SOURCE_DIR}/../third_party)
if(NOT EXISTS ${THIRD_PARTY_DIR}/ecomug/EcoMug.h)
  message(FATAL_ERROR
    "Vendored EcoMug.h not found at ${THIRD_PARTY_DIR}/ecomug/. "
    "This build is offline-only by design; both third_party/ecomug and "
    "third_party/json must be present.")
endif()

file(GLOB sources ${PROJECT_SOURCE_DIR}/src/*.cc)

add_executable(muography_sim muography_sim.cc ${sources})
target_include_directories(muography_sim PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${THIRD_PARTY_DIR}/ecomug
  ${THIRD_PARTY_DIR}/json
)
target_link_libraries(muography_sim PRIVATE ${Geant4_LIBRARIES})
