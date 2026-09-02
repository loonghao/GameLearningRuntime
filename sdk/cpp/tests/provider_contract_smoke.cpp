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
  const glr::resume_request request{
      "episode-1", 0, std::optional<std::string>{"runtime-1"}};
  const glr::action_reconciliation reconciliation{
      request.episode_id, 1, glr::reconciliation_outcome::unknown, 0, 42, false};
  if (reconciliation.authoritative_step_id != request.last_committed_step_id) {
    return 1;
  }
  const glr::realtime_timing_contract timing{1, 10, 100, 5, "monotonic"};
  timing.validate();
  const glr::realtime_step_timing step_timing{50, 5, 2};
  step_timing.validate(timing);
  const glr::input_lease_token lease{"session.one.lease", "session.one", "target.game"};
  lease.validate();
  std::cout << glr::host_schema << " provider-sdk-ok\n";
  return 0;
}
