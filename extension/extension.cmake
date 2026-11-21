# Inkling 2 Mini (inkling2_loop) at native weight precision, for GUFO on Strix Halo.

# Host code: probe, loader and the CPU reference forward pass. No GPU.
gufo_extension_target(
  NAME gufo_inkling2_native
  SOURCES
    src/chat.cpp
    src/config.cpp
    src/model.cpp
    src/reference.cpp
    src/tokenizer.cpp
    src/extension.cpp
  LINK_LIBRARIES
    gufo_ext_runtime
    Threads::Threads
)

if(TARGET gufo_inkling2_native)
  target_compile_options(gufo_inkling2_native PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>)
endif()

# The provider and the HIP backend.
gufo_extension_target(
  NAME gufo_inkling2_native_gpu
  SOURCES
    src/decoder.cpp
    src/drafter.cpp
    src/gpu.cpp
    src/provider.cpp
  LINK_LIBRARIES
    gufo_inkling2_native
    gufo_ext_runtime
    hip::host
    Threads::Threads
)

if(TARGET gufo_inkling2_native_gpu)
  gufo_extension_hip_sources(gufo_inkling2_native_gpu
    SOURCES
      src/hip/decode.hip
      src/hip/ops.hip
  )
  target_compile_options(gufo_inkling2_native_gpu PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:HIP>:-O3>
    $<$<COMPILE_LANGUAGE:HIP>:-Wno-unused-command-line-argument>
  )
endif()

install(FILES "${GUFO_EXTENSION_BUILD_ROOT}/LICENSE"
  DESTINATION share/licenses/gufo/extensions/inkling2_native)
