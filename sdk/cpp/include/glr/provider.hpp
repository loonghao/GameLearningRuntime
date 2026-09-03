#pragma once

#include <cstdint>
#include <stdexcept>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace glr {

inline constexpr std::string_view host_schema = "glr.host.v1";
inline constexpr std::string_view environment_protocol_version = "1.0";
inline constexpr std::string_view realtime_control_schema = "glr.realtime-control.v1";
inline constexpr std::string_view checkpoint_contract_schema = "glr.checkpoint-contract.v1";
inline constexpr std::string_view checkpoint_manifest_schema = "glr.checkpoint-manifest.v1";
inline constexpr std::string_view runtime_health_schema = "glr.runtime-health.v1";

enum class dtype { boolean, uint8, int32, int64, float32, float64 };
enum class space_kind { continuous, discrete, multi_discrete, binary };
enum class action_outcome { accepted, rejected, unknown, no_effect, partial, blocked };
enum class refusal_reason_class { transient, structural };
enum class reconciliation_outcome { applied, not_applied, unknown };
enum class realtime_action_status { consumed, expired, cancelled, rejected };
enum class input_lease_operation { acquire, renew, release, preempt };
enum class input_lease_status { acquired, renewed, released, preempted, rejected };
enum class runtime_health_status { starting, ready, draining, unhealthy, stopped };

inline bool ascii_letter_or_digit(unsigned char value) {
  return (value >= 'A' && value <= 'Z') || (value >= 'a' && value <= 'z')
      || (value >= '0' && value <= '9');
}

inline bool valid_runtime_identifier(std::string_view value) {
  if (value.empty() || value.size() > 128
      || !ascii_letter_or_digit(static_cast<unsigned char>(value.front()))) {
    return false;
  }
  for (const unsigned char character : value.substr(1)) {
    if (!(ascii_letter_or_digit(character) || character == '_' || character == '.'
          || character == ':' || character == '-')) {
      return false;
    }
  }
  return true;
}

inline bool valid_sha256(std::string_view value) {
  if (value.size() != 64) {
    return false;
  }
  for (const char character : value) {
    if (!((character >= '0' && character <= '9')
          || (character >= 'a' && character <= 'f'))) {
      return false;
    }
  }
  return true;
}

inline bool valid_checkpoint_path(std::string_view value) {
  if (value.empty() || value.front() == '/' || value.find('\\') != std::string_view::npos
      || value.find(':') != std::string_view::npos) {
    return false;
  }
  std::size_t start = 0;
  while (start <= value.size()) {
    const std::size_t end = value.find('/', start);
    const std::string_view part = value.substr(start, end == std::string_view::npos
                                                         ? value.size() - start
                                                         : end - start);
    if (part == "." || part == "..") {
      return false;
    }
    if (end == std::string_view::npos) {
      break;
    }
    start = end + 1;
  }
  return true;
}

struct checkpoint_contract final {
  std::string schema_version = std::string(checkpoint_contract_schema);
  std::string protocol_version;
  std::string observation_sha256;
  std::string action_sha256;
  std::string reward_sha256;
  std::optional<std::string> knowledge_sha256;

  void validate() const {
    if (schema_version != checkpoint_contract_schema || protocol_version.empty()
        || !valid_sha256(observation_sha256) || !valid_sha256(action_sha256)
        || !valid_sha256(reward_sha256)
        || (knowledge_sha256.has_value() && !valid_sha256(*knowledge_sha256))) {
      throw std::invalid_argument("invalid checkpoint contract");
    }
  }
};

struct checkpoint_manifest final {
  std::string schema_version = std::string(checkpoint_manifest_schema);
  std::string checkpoint_path;
  std::string checkpoint_sha256;
  std::uint64_t checkpoint_size_bytes;
  checkpoint_contract contract;
  std::string contract_sha256;
  std::map<std::string, std::string> metadata;

  void validate() const {
    if (schema_version != checkpoint_manifest_schema || !valid_checkpoint_path(checkpoint_path)
        || !valid_sha256(checkpoint_sha256)
        || !valid_sha256(contract_sha256)) {
      throw std::invalid_argument("invalid checkpoint manifest");
    }
    contract.validate();
  }
};

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

struct realtime_timing_contract final {
  std::uint64_t minimum_hold_ns;
  std::uint64_t maximum_hold_ns;
  std::uint64_t settle_deadline_ns;
  std::uint64_t simulation_quantum_ns;
  std::string clock_source;
  std::string schema_version = std::string(realtime_control_schema);

  void validate() const {
    if (schema_version != realtime_control_schema || minimum_hold_ns == 0
        || maximum_hold_ns == 0 || settle_deadline_ns == 0
        || simulation_quantum_ns == 0 || minimum_hold_ns > maximum_hold_ns
        || maximum_hold_ns > settle_deadline_ns || clock_source.empty()) {
      throw std::invalid_argument("invalid realtime timing contract");
    }
  }
};

struct realtime_step_timing final {
  std::uint64_t deadline_ns;
  std::uint64_t quantum_ns;
  std::optional<std::uint64_t> hold_ns;

  void validate(const realtime_timing_contract& contract) const {
    if (deadline_ns == 0 || quantum_ns == 0 || deadline_ns > contract.settle_deadline_ns
        || quantum_ns > contract.simulation_quantum_ns
        || (hold_ns.has_value() && (*hold_ns == 0 || *hold_ns < contract.minimum_hold_ns
                                    || *hold_ns > contract.maximum_hold_ns))) {
      throw std::invalid_argument("realtime step timing is outside descriptor bounds");
    }
  }
};

struct realtime_action_receipt final {
  std::string action_id;
  realtime_action_status status;
  std::uint64_t deadline_ns;
  std::uint64_t quantum_ns;
  std::uint64_t issued_at_ns;
  std::optional<std::uint64_t> consumed_at_ns;
  std::optional<std::uint64_t> settled_at_ns;
  std::optional<std::string> cancellation_token;

  void validate() const {
    if (action_id.empty() || deadline_ns == 0 || quantum_ns == 0
        || (consumed_at_ns.has_value() && *consumed_at_ns < issued_at_ns)
        || (settled_at_ns.has_value() && *settled_at_ns < issued_at_ns)
        || (consumed_at_ns.has_value() && *consumed_at_ns >= issued_at_ns
            && *consumed_at_ns - issued_at_ns > deadline_ns)) {
      throw std::invalid_argument("invalid realtime action receipt");
    }
    if (settled_at_ns.has_value() && consumed_at_ns.has_value()
        && *settled_at_ns < *consumed_at_ns) {
      throw std::invalid_argument("settled timestamp precedes consumed timestamp");
    }
  }
};

struct input_lease_token final {
  std::string lease_id;
  std::string session_id;
  std::string target_id;

  void validate() const {
    if (lease_id.empty() || session_id.empty() || target_id.empty()) {
      throw std::invalid_argument("input lease token IDs are required");
    }
  }
};

struct input_lease_request final {
  input_lease_operation operation;
  std::string session_id;
  std::string target_id;
  std::optional<std::string> lease_id;
  std::optional<std::uint64_t> expires_at_ns;

  void validate() const {
    if (session_id.empty() || target_id.empty()
        || (operation == input_lease_operation::acquire && lease_id.has_value())
        || (operation != input_lease_operation::acquire && !lease_id.has_value())) {
      throw std::invalid_argument("invalid input lease request");
    }
  }
};

struct input_lease_receipt final {
  input_lease_status status;
  std::optional<input_lease_token> token;
  std::uint64_t observed_at_ns;
  std::optional<std::uint64_t> expires_at_ns;
  std::optional<std::string> reason;

  void validate() const {
    if (token.has_value()) {
      token->validate();
    }
    if (expires_at_ns.has_value() && *expires_at_ns <= observed_at_ns) {
      throw std::invalid_argument("input lease receipt expiry must be in the future");
    }
  }
};

struct runtime_identity final {
  std::string runtime_id;
  std::string runtime_version;

  void validate() const {
    if (!valid_runtime_identifier(runtime_id) || !valid_runtime_identifier(runtime_version)) {
      throw std::invalid_argument("invalid runtime identity");
    }
  }
};

struct runtime_lease final {
  std::string lease_id;
  std::string owner_id;
  std::uint64_t expires_at_ns;

  void validate(std::uint64_t observed_at_ns) const {
    if (!valid_runtime_identifier(lease_id) || !valid_runtime_identifier(owner_id)
        || expires_at_ns <= observed_at_ns) {
      throw std::invalid_argument("invalid runtime lease");
    }
  }
};

struct runtime_health final {
  std::string schema_version = std::string(runtime_health_schema);
  runtime_identity identity;
  runtime_health_status status;
  std::uint64_t observed_at_ns;
  bool accepting_new_sessions;
  std::uint32_t active_sessions;
  std::optional<runtime_lease> lease;

  void validate() const {
    if (schema_version != runtime_health_schema) {
      throw std::invalid_argument("invalid runtime health schema");
    }
    identity.validate();
    if (lease.has_value()) {
      lease->validate(observed_at_ns);
    }
  }
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
  std::optional<realtime_timing_contract> realtime_timing;
  std::optional<glr::runtime_identity> runtime_identity;
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
  std::optional<glr::realtime_action_receipt> realtime;
  std::optional<std::string> target_id;
  std::optional<glr::refusal_reason_class> reason_class;
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

struct action_reconciliation final {
  std::string episode_id;
  std::uint64_t expected_step_id;
  reconciliation_outcome outcome;
  std::uint64_t authoritative_step_id;
  std::uint64_t timestamp_ns;
  bool retryable{false};
};

struct provider_resume_result final {
  provider_time_step timestep;
  std::uint64_t committed_step_id;
  std::optional<action_reconciliation> reconciliation;
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
  std::optional<std::string> action_id;
  std::optional<std::uint64_t> issued_at_ns;
  std::optional<realtime_step_timing> timing;
  std::optional<input_lease_token> lease;
  std::optional<std::string> cancellation_token;
};

struct resume_request final {
  std::string episode_id;
  std::uint64_t last_committed_step_id;
  std::optional<std::string> target_id;
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
  [[nodiscard]] virtual runtime_health health() const {
    throw std::logic_error("provider does not support runtime-health-v1");
  }
  [[nodiscard]] virtual provider_time_step reset(const reset_request& request) = 0;
  [[nodiscard]] virtual provider_time_step attach(const attach_request& request) = 0;
  [[nodiscard]] virtual provider_time_step step(const step_request& request) = 0;
  [[nodiscard]] virtual provider_resume_result resume(
      const resume_request& /*request*/) {
    throw std::logic_error("provider does not support reconnect-resume-v1");
  }

  [[nodiscard]] virtual input_lease_receipt lease(const input_lease_request& request) {
    (void)request;
    return {input_lease_status::rejected, std::nullopt, 0, std::nullopt,
            std::optional<std::string>("provider does not support input leases")};
  }
  virtual void close() noexcept = 0;
};

}  // namespace glr
