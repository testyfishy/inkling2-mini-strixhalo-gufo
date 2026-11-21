#include <gufo/ext/registry.hpp>

#include "tools.hpp"

// The provider registers itself from the HIP target (provider.cpp); what
// needs no GPU is registered here: the syntax of the model's tool calls.
void inkling2_native_register(gufo::ext::Registry& registry) {
  (void)registry;
  inkling2::RegisterToolDialect();
}
