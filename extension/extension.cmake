# Inkling 2 Mini (inkling2_loop) at native weight precision, for GUFO on Strix Halo.
# Milestone 1: probe and loader only. No GPU code yet.

gufo_extension_target(
  NAME gufo_inkling2_native
  SOURCES
    src/config.cpp
    src/provider.cpp
  LINK_LIBRARIES
    gufo_ext_runtime
)

install(FILES "${GUFO_EXTENSION_BUILD_ROOT}/LICENSE"
  DESTINATION share/licenses/gufo/extensions/inkling2_native)
