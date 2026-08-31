#include <glr/provider.hpp>

#include <cstdint>
#include <iostream>
#include <vector>

int main() {
  static_assert(glr::host_schema == "glr.host.v1");
  const glr::tensor_buffer tensor{{1}, glr::dtype::int64, std::vector<std::uint8_t>(8)};
  if (tensor.shape.size() != 1 || tensor.data.size() != 8) {
    return 1;
  }
  std::cout << glr::host_schema << " provider-sdk-ok\n";
  return 0;
}
