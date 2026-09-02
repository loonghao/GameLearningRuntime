#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace glr {

inline constexpr std::string_view host_schema = "glr.host.v1";
inline constexpr std::string_view environment_protocol_version = "1.0";

enum class dtype { boolean, uint8, int32, int64, float32, float64 };
enum class space_kind { continuous, discrete, multi_discrete, binary };
enum class action_outcome { accepted, rejected, unknown, no_effect, partial, blocked };

struct tensor_buffer final {
  std::vector<std::uint64_t> shape;
  dtype element_type;
  std::vector<std::uint8_t> data;
};

struct tensor_spec final {
  std::string path;
  std::vector<std::int64_t> shape;
  dtype element_type;
  space_kind kind;
  std::optional<double> minimum;
  std::optional<double> maximum;
  std::string description;
};

struct provider_descriptor final {
  std::string environment_id;
  std::vector<tensor_spec> observations;
  std::vector<tensor_spec> actions;
  std::vector<tensor_spec> action_masks;
  tensor_spec reward;
  tensor_spec done;
  std::vector<std::string> capabilities;
  std::map<std::string, std::string> metadata;
};

struct provider_event final {
  std::string name;
  std::uint64_t timestamp_ns;
  std::vector<std::uint8_t> payload_json_utf8;
};

struct action_receipt final {
  std::string action_id;
  std::string episode_id;
  std::uint64_t step_id;
  action_outcome outcome;
  std::uint64_t issued_timestamp_ns;
  std::uint64_t observed_timestamp_ns;
  std::string postcondition;
  std::optional<double> progress_delta;
  std::optional<std::uint64_t> authoritative_observation_sequence;
  bool retryable;
};

struct provider_time_step final {
  std::string episode_id;
  std::uint64_t step_id;
  std::uint64_t timestamp_ns;
  std::map<std::string, tensor_buffer> observation;
  tensor_buffer reward;
  tensor_buffer terminated;
  tensor_buffer truncated;
  std::map<std::string, tensor_buffer> action_mask;
  std::vector<provider_event> events;
  std::vector<std::uint8_t> info_json_utf8;
  std::optional<glr::action_receipt> action_receipt;
};

struct reset_request final {
  std::optional<std::uint64_t> seed;
  std::map<std::string, std::string> options;
};

struct attach_request final {
  std::map<std::string, std::string> options;
};

struct step_request final {
  std::string episode_id;
  std::uint64_t expected_step_id;
  std::map<std::string, tensor_buffer> action;
};

class runtime_provider {
 public:
  runtime_provider() = default;
  runtime_provider(const runtime_provider&) = delete;
  runtime_provider& operator=(const runtime_provider&) = delete;
  runtime_provider(runtime_provider&&) = delete;
  runtime_provider& operator=(runtime_provider&&) = delete;
  virtual ~runtime_provider() = default;

  [[nodiscard]] virtual provider_descriptor describe() const = 0;
  [[nodiscard]] virtual provider_time_step reset(const reset_request& request) = 0;
  [[nodiscard]] virtual provider_time_step attach(const attach_request& request) = 0;
  [[nodiscard]] virtual provider_time_step step(const step_request& request) = 0;
  virtual void close() noexcept = 0;
};

}  // namespace glr
