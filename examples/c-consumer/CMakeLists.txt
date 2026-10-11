# A CMake project that consumes the TAPP surface, the way a C or C++ codebase
# actually would.
#
#   cmake -S examples/c-consumer -B build/c-consumer -DCMAKE_BUILD_TYPE=Release
#   cmake --build build/c-consumer
#   ./build/c-consumer/tapp-c-consumer
#
# Two ways to get the library are supported, because they suit different sites:
#
#   TAPP_USE_CORROSION=ON  (default)
#     Corrosion drives cargo from CMake and imports the crate as a normal CMake
#     target, so `target_link_libraries` is all the consumer writes. It works
#     out the platform link line — on Linux a Rust staticlib still needs
#     -lpthread -ldl -lm, and corrosion knows that so you do not have to.
#     Requires a Rust toolchain at *build* time (not at run time), and network
#     access on the first configure to fetch corrosion itself.
#
#   TAPP_USE_CORROSION=OFF
#     Link a library that is already built, via TAPP_LIBRARY_DIR. This is the
#     path for a site that builds the Rust half separately — or offline, or
#     from a vendored tarball — and for anyone who would rather not have CMake
#     invoke cargo at all.
#
#   TAPP_PREFIX=<dir>
#     Consume an *installed* prefix — the layout `install.sh` produces, which is
#     also the layout a JLL or a system package presents. Takes precedence over
#     both of the above. This is the only mode in which the header is found
#     somewhere other than the source tree, so it is the one that would catch a
#     header the install forgot to copy.
#
# Nothing here needs a Rust toolchain at run time, and nothing crosses the C++
# ABI: the boundary is C, so there is no name mangling, no exception
# propagation and no libstdc++ version coupling.

cmake_minimum_required(VERSION 3.22)
project(tapp-c-consumer C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

option(TAPP_USE_CORROSION "Build the Rust library via corrosion" ON)

# The crate root, i.e. the repository root two levels up from this file.
get_filename_component(TAPP_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(TAPP_CRATE_DIR "${TAPP_REPO_ROOT}/crates/tcontract-capi")

add_executable(tapp-c-consumer main.c)

# This program's own dependency, not the library's: `main.c` calls `cabs` from
# <complex.h> to compare results. Stated once, before the branch, because all
# three link modes need it and only one of them used to say so.
if(UNIX)
    target_link_libraries(tapp-c-consumer PRIVATE m)
endif()

if(DEFINED TAPP_PREFIX)
    # An installed prefix. Prefer pkg-config, since that is what the install
    # ships and what a real consumer would reach for; fall back to the layout
    # directly if pkg-config is not available.
    find_package(PkgConfig)
    if(PkgConfig_FOUND)
        set(ENV{PKG_CONFIG_PATH} "${TAPP_PREFIX}/lib/pkgconfig")
        pkg_check_modules(TAPP REQUIRED IMPORTED_TARGET tcontract_capi)
        message(STATUS "Using installed TAPP ${TAPP_VERSION} via pkg-config from ${TAPP_PREFIX}")
        target_link_libraries(tapp-c-consumer PRIVATE PkgConfig::TAPP)
        set(TAPP_RUNTIME_DIR "${TAPP_LIBRARY_DIRS}")
    else()
        message(STATUS "Using installed TAPP from ${TAPP_PREFIX} (no pkg-config)")
        find_library(TAPP_LIBRARY
            NAMES tcontract_capi
            HINTS "${TAPP_PREFIX}/lib" "${TAPP_PREFIX}/bin"
            REQUIRED
        )
        target_include_directories(tapp-c-consumer PRIVATE "${TAPP_PREFIX}/include")
        target_link_libraries(tapp-c-consumer PRIVATE "${TAPP_LIBRARY}")
        get_filename_component(TAPP_RUNTIME_DIR "${TAPP_LIBRARY}" DIRECTORY)
    endif()

    # An installed cdylib carries a SONAME but is not on the default search path,
    # so the consumer needs the rpath just as in the prebuilt mode.
    set_target_properties(tapp-c-consumer PROPERTIES BUILD_RPATH "${TAPP_RUNTIME_DIR}")

    enable_testing()
    add_test(NAME c-consumer COMMAND tapp-c-consumer)
    return()
endif()

target_include_directories(tapp-c-consumer PRIVATE "${TAPP_CRATE_DIR}/include")

if(TAPP_USE_CORROSION)
    include(FetchContent)
    FetchContent_Declare(
        Corrosion
        GIT_REPOSITORY https://github.com/corrosion-rs/corrosion.git
        GIT_TAG v0.5.1
    )
    FetchContent_MakeAvailable(Corrosion)

    # `cargo metadata` on the workspace manifest, then import just the crate
    # that exports the C ABI. CRATES= keeps the unpublished bench harness out
    # of the build entirely.
    corrosion_import_crate(
        MANIFEST_PATH "${TAPP_REPO_ROOT}/Cargo.toml"
        CRATES tcontract-capi
    )

    target_link_libraries(tapp-c-consumer PRIVATE tcontract_capi)
else()
    if(NOT DEFINED TAPP_LIBRARY_DIR)
        set(TAPP_LIBRARY_DIR "${TAPP_REPO_ROOT}/target/release")
    endif()

    find_library(TAPP_LIBRARY
        NAMES tcontract_capi
        HINTS "${TAPP_LIBRARY_DIR}"
        REQUIRED
        NO_DEFAULT_PATH
    )
    message(STATUS "Using prebuilt TAPP library: ${TAPP_LIBRARY}")

    target_link_libraries(tapp-c-consumer PRIVATE "${TAPP_LIBRARY}")

    # A Rust *staticlib* bundles the standard library and needs these; a cdylib
    # does not, but asking for them is harmless either way. Threads are needed
    # regardless, since the engine's optional threading uses std::thread.
    find_package(Threads REQUIRED)
    target_link_libraries(tapp-c-consumer PRIVATE Threads::Threads ${CMAKE_DL_LIBS})

    # Let the built binary find a cdylib without LD_LIBRARY_PATH being set.
    set_target_properties(tapp-c-consumer PROPERTIES
        BUILD_RPATH "${TAPP_LIBRARY_DIR}"
    )
endif()

enable_testing()
add_test(NAME c-consumer COMMAND tapp-c-consumer)
