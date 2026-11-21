#include <gufo/ext/registry.hpp>

// The provider registers itself from the HIP target (provider.cpp); builds
// without HIP have nothing to add yet.
void inkling2_native_register(gufo::ext::Registry& registry) {
  (void)registry;
}
