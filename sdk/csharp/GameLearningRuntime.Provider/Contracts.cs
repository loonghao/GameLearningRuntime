using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;

namespace GameLearningRuntime.Provider
{
    /// <summary>Versioned descriptor-level realtime timing bounds.</summary>
    public sealed class RealtimeTimingContract
    {
        /// <summary>Wire schema identifier for the realtime-control contract.</summary>
        public const string SchemaVersion = "glr.realtime-control.v1";

        /// <summary>Create descriptor-level realtime timing bounds.</summary>
        public RealtimeTimingContract(
            ulong minimumHoldNanoseconds,
            ulong maximumHoldNanoseconds,
            ulong settleDeadlineNanoseconds,
            ulong simulationQuantumNanoseconds,
            string clockSource = "monotonic")
        {
            if (minimumHoldNanoseconds == 0 || maximumHoldNanoseconds == 0
                || settleDeadlineNanoseconds == 0 || simulationQuantumNanoseconds == 0)
            {
                throw new ArgumentOutOfRangeException(nameof(minimumHoldNanoseconds), "Timing bounds must be positive.");
            }

            if (minimumHoldNanoseconds > maximumHoldNanoseconds)
            {
                throw new ArgumentException("Minimum hold cannot exceed maximum hold.", nameof(minimumHoldNanoseconds));
            }

            if (maximumHoldNanoseconds > settleDeadlineNanoseconds)
            {
                throw new ArgumentException("Maximum hold cannot exceed settle deadline.", nameof(maximumHoldNanoseconds));
            }

            if (string.IsNullOrWhiteSpace(clockSource))
            {
                throw new ArgumentException("Clock source cannot be empty.", nameof(clockSource));
            }

            MinimumHoldNanoseconds = minimumHoldNanoseconds;
            MaximumHoldNanoseconds = maximumHoldNanoseconds;
            SettleDeadlineNanoseconds = settleDeadlineNanoseconds;
            SimulationQuantumNanoseconds = simulationQuantumNanoseconds;
            ClockSource = clockSource;
        }

        /// <summary>Minimum input hold duration.</summary>
        public ulong MinimumHoldNanoseconds { get; }
        /// <summary>Maximum input hold duration.</summary>
        public ulong MaximumHoldNanoseconds { get; }
        /// <summary>Maximum settle deadline.</summary>
        public ulong SettleDeadlineNanoseconds { get; }
        /// <summary>Maximum simulation quantum.</summary>
        public ulong SimulationQuantumNanoseconds { get; }
        /// <summary>Clock source used for timing values.</summary>
        public string ClockSource { get; }
    }

    /// <summary>Bounded per-step timing values.</summary>
    public sealed class RealtimeStepTiming
    {
        /// <summary>Create bounded per-step timing values.</summary>
        public RealtimeStepTiming(ulong deadlineNanoseconds, ulong quantumNanoseconds, ulong? holdNanoseconds = null)
        {
            if (deadlineNanoseconds == 0 || quantumNanoseconds == 0)
            {
                throw new ArgumentOutOfRangeException(nameof(deadlineNanoseconds), "Step timing values must be positive.");
            }

            if (holdNanoseconds.HasValue && holdNanoseconds.Value == 0)
            {
                throw new ArgumentOutOfRangeException(nameof(holdNanoseconds), "Hold duration must be positive.");
            }

            DeadlineNanoseconds = deadlineNanoseconds;
            QuantumNanoseconds = quantumNanoseconds;
            HoldNanoseconds = holdNanoseconds;
        }

        /// <summary>Action deadline duration.</summary>
        public ulong DeadlineNanoseconds { get; }
        /// <summary>Simulation quantum duration.</summary>
        public ulong QuantumNanoseconds { get; }
        /// <summary>Optional input hold duration.</summary>
        public ulong? HoldNanoseconds { get; }

        /// <summary>Validate these values against descriptor bounds.</summary>
        public void ValidateAgainst(RealtimeTimingContract contract)
        {
            if (DeadlineNanoseconds > contract.SettleDeadlineNanoseconds
                || QuantumNanoseconds > contract.SimulationQuantumNanoseconds)
            {
                throw new ArgumentException("Step timing exceeds the descriptor bounds.", nameof(contract));
            }

            if (HoldNanoseconds.HasValue && (HoldNanoseconds.Value < contract.MinimumHoldNanoseconds
                || HoldNanoseconds.Value > contract.MaximumHoldNanoseconds))
            {
                throw new ArgumentException("Step hold is outside the descriptor bounds.", nameof(contract));
            }
        }
    }

    /// <summary>Typed outcome of a realtime action dispatch.</summary>
    public enum RealtimeActionStatus
    {
        /// <summary>The provider consumed the action.</summary>
        Consumed,
        /// <summary>The action deadline elapsed before consumption.</summary>
        Expired,
        /// <summary>The action was cancelled before consumption.</summary>
        Cancelled,
        /// <summary>The provider rejected the action.</summary>
        Rejected,
    }

    /// <summary>Typed timing receipt linked to an action post-state.</summary>
    public sealed class RealtimeActionReceipt
    {
        /// <summary>Create one realtime action receipt.</summary>
        public RealtimeActionReceipt(
            string actionId,
            RealtimeActionStatus status,
            ulong deadlineNanoseconds,
            ulong quantumNanoseconds,
            ulong issuedAtNanoseconds,
            ulong? consumedAtNanoseconds = null,
            ulong? settledAtNanoseconds = null,
            string? cancellationToken = null)
        {
            if (string.IsNullOrWhiteSpace(actionId))
            {
                throw new ArgumentException("Action ID cannot be empty.", nameof(actionId));
            }

            if (deadlineNanoseconds == 0 || quantumNanoseconds == 0)
            {
                throw new ArgumentOutOfRangeException(nameof(deadlineNanoseconds), "Receipt timing values must be positive.");
            }

            if (consumedAtNanoseconds.HasValue && consumedAtNanoseconds.Value >= issuedAtNanoseconds
                && consumedAtNanoseconds.Value - issuedAtNanoseconds > deadlineNanoseconds)
            {
                throw new ArgumentException("Consumed timestamp exceeds the action deadline.", nameof(consumedAtNanoseconds));
            }

            if (consumedAtNanoseconds.HasValue && consumedAtNanoseconds.Value < issuedAtNanoseconds)
            {
                throw new ArgumentException("Consumed timestamp cannot precede issue time.", nameof(consumedAtNanoseconds));
            }

            if (settledAtNanoseconds.HasValue && settledAtNanoseconds.Value < issuedAtNanoseconds)
            {
                throw new ArgumentException("Settled timestamp cannot precede issue time.", nameof(settledAtNanoseconds));
            }

            if (settledAtNanoseconds.HasValue && consumedAtNanoseconds.HasValue
                && settledAtNanoseconds.Value < consumedAtNanoseconds.Value)
            {
                throw new ArgumentException("Settled timestamp cannot precede consumed time.", nameof(settledAtNanoseconds));
            }

            ActionId = actionId;
            Status = status;
            DeadlineNanoseconds = deadlineNanoseconds;
            QuantumNanoseconds = quantumNanoseconds;
            IssuedAtNanoseconds = issuedAtNanoseconds;
            ConsumedAtNanoseconds = consumedAtNanoseconds;
            SettledAtNanoseconds = settledAtNanoseconds;
            CancellationToken = cancellationToken;
        }

        /// <summary>Stable action identity.</summary>
        public string ActionId { get; }
        /// <summary>Typed dispatch outcome.</summary>
        public RealtimeActionStatus Status { get; }
        /// <summary>Action deadline duration.</summary>
        public ulong DeadlineNanoseconds { get; }
        /// <summary>Simulation quantum duration.</summary>
        public ulong QuantumNanoseconds { get; }
        /// <summary>Provider timestamp when the action was issued.</summary>
        public ulong IssuedAtNanoseconds { get; }
        /// <summary>Optional timestamp when the provider consumed the action.</summary>
        public ulong? ConsumedAtNanoseconds { get; }
        /// <summary>Optional timestamp when post-state settled.</summary>
        public ulong? SettledAtNanoseconds { get; }
        /// <summary>Optional cancellation fencing token.</summary>
        public string? CancellationToken { get; }
    }

    /// <summary>Lifecycle operation for one target-bound input lease.</summary>
    public enum InputLeaseOperation
    {
        /// <summary>Acquire a new lease.</summary>
        Acquire,
        /// <summary>Renew an existing lease.</summary>
        Renew,
        /// <summary>Release an existing lease.</summary>
        Release,
        /// <summary>Preempt an existing lease.</summary>
        Preempt,
    }

    /// <summary>Typed result of a lease operation.</summary>
    public enum InputLeaseStatus
    {
        /// <summary>A lease was acquired.</summary>
        Acquired,
        /// <summary>A lease was renewed.</summary>
        Renewed,
        /// <summary>A lease was released.</summary>
        Released,
        /// <summary>A lease was preempted.</summary>
        Preempted,
        /// <summary>The operation was rejected.</summary>
        Rejected,
    }

    /// <summary>Opaque lease binding for one target and logical session.</summary>
    public sealed class InputLeaseToken
    {
        /// <summary>Create a target- and session-bound lease token.</summary>
        public InputLeaseToken(string leaseId, string sessionId, string targetId)
        {
            if (string.IsNullOrWhiteSpace(leaseId) || string.IsNullOrWhiteSpace(sessionId)
                || string.IsNullOrWhiteSpace(targetId))
            {
                throw new ArgumentException("Lease, session, and target IDs are required.");
            }

            LeaseId = leaseId;
            SessionId = sessionId;
            TargetId = targetId;
        }

        /// <summary>Opaque lease identity.</summary>
        public string LeaseId { get; }
        /// <summary>Logical session identity.</summary>
        public string SessionId { get; }
        /// <summary>Target identity.</summary>
        public string TargetId { get; }
    }

    /// <summary>One explicit lease lifecycle request.</summary>
    public sealed class InputLeaseRequest
    {
        /// <summary>Create one explicit lease lifecycle request.</summary>
        public InputLeaseRequest(
            InputLeaseOperation operation,
            string sessionId,
            string targetId,
            string? leaseId = null,
            ulong? expiresAtNanoseconds = null)
        {
            if (string.IsNullOrWhiteSpace(sessionId) || string.IsNullOrWhiteSpace(targetId))
            {
                throw new ArgumentException("Session and target IDs are required.");
            }

            if (operation == InputLeaseOperation.Acquire && leaseId != null)
            {
                throw new ArgumentException("Acquire cannot provide an existing lease ID.", nameof(leaseId));
            }

            if (operation != InputLeaseOperation.Acquire && string.IsNullOrWhiteSpace(leaseId))
            {
                throw new ArgumentException("An existing lease ID is required.", nameof(leaseId));
            }

            if (expiresAtNanoseconds.HasValue && expiresAtNanoseconds.Value == 0)
            {
                throw new ArgumentOutOfRangeException(nameof(expiresAtNanoseconds), "Lease expiry must be positive.");
            }

            Operation = operation;
            SessionId = sessionId;
            TargetId = targetId;
            LeaseId = leaseId;
            ExpiresAtNanoseconds = expiresAtNanoseconds;
        }

        /// <summary>Requested lifecycle operation.</summary>
        public InputLeaseOperation Operation { get; }
        /// <summary>Logical session identity.</summary>
        public string SessionId { get; }
        /// <summary>Target identity.</summary>
        public string TargetId { get; }
        /// <summary>Existing lease identity for renew/release/preempt.</summary>
        public string? LeaseId { get; }
        /// <summary>Optional absolute expiry timestamp.</summary>
        public ulong? ExpiresAtNanoseconds { get; }
    }

    /// <summary>Typed result of a lease lifecycle operation.</summary>
    public sealed class InputLeaseReceipt
    {
        /// <summary>Create a typed lease operation result.</summary>
        public InputLeaseReceipt(
            InputLeaseStatus status,
            InputLeaseToken? token,
            ulong observedAtNanoseconds,
            ulong? expiresAtNanoseconds = null,
            string? reason = null)
        {
            if (expiresAtNanoseconds.HasValue && expiresAtNanoseconds.Value <= observedAtNanoseconds)
            {
                throw new ArgumentException("Lease expiry must be after the observation time.", nameof(expiresAtNanoseconds));
            }

            Status = status;
            Token = token;
            ObservedAtNanoseconds = observedAtNanoseconds;
            ExpiresAtNanoseconds = expiresAtNanoseconds;
            Reason = reason;
        }

        /// <summary>Typed operation result.</summary>
        public InputLeaseStatus Status { get; }
        /// <summary>Exact target/session-bound token, when present.</summary>
        public InputLeaseToken? Token { get; }
        /// <summary>Provider observation timestamp.</summary>
        public ulong ObservedAtNanoseconds { get; }
        /// <summary>Optional lease expiry timestamp.</summary>
        public ulong? ExpiresAtNanoseconds { get; }
        /// <summary>Optional bounded rejection reason.</summary>
        public string? Reason { get; }
    }

    /// <summary>Typed outcome of one mutating realtime action.</summary>
    public enum ActionOutcome
    {
        /// <summary>The provider consumed the action and observed its postcondition.</summary>
        Accepted,
        /// <summary>The provider rejected the action before mutation.</summary>
        Rejected,
        /// <summary>The provider cannot determine whether mutation occurred.</summary>
        Unknown,
        /// <summary>The action was consumed but produced no observed progress.</summary>
        NoEffect,
        /// <summary>The action produced only part of its requested postcondition.</summary>
        Partial,
        /// <summary>The action was prevented by a known runtime blocker.</summary>
        Blocked,
    }

    /// <summary>Classifies why a provider refused a command.</summary>
    public enum RefusalReasonClass
    {
        /// <summary>The command may succeed after a bounded retry.</summary>
        Transient,
        /// <summary>The command cannot succeed without changing the request or state.</summary>
        Structural,
    }

    /// <summary>Authoritative receipt tied to a provider post-state.</summary>
    public sealed class ActionReceipt
    {
        /// <summary>Create a bounded action receipt.</summary>
        public ActionReceipt(
            string actionId,
            Guid episodeId,
            ulong stepId,
            ActionOutcome outcome,
            ulong issuedTimestampNanoseconds,
            ulong observedTimestampNanoseconds,
            string postcondition = "unknown",
            double? progressDelta = null,
            ulong? authoritativeObservationSequence = null,
            bool retryable = false,
            RealtimeActionReceipt? realtime = null,
            string? targetId = null,
            RefusalReasonClass? reasonClass = null)
        {
            if (string.IsNullOrWhiteSpace(actionId) || actionId.Length > 128)
            {
                throw new ArgumentException("Action ID must contain 1-128 characters.", nameof(actionId));
            }

            if (stepId == 0)
            {
                throw new ArgumentOutOfRangeException(nameof(stepId), "Step ID must be positive.");
            }

            if (observedTimestampNanoseconds < issuedTimestampNanoseconds)
            {
                throw new ArgumentException(
                    "Observed timestamp cannot precede issued timestamp.",
                    nameof(observedTimestampNanoseconds));
            }

            if (string.IsNullOrWhiteSpace(postcondition) || postcondition.Length > 128)
            {
                throw new ArgumentException(
                    "Postcondition must contain 1-128 characters.",
                    nameof(postcondition));
            }

            if (progressDelta.HasValue && (double.IsNaN(progressDelta.Value) || double.IsInfinity(progressDelta.Value)))
            {
                throw new ArgumentException("Progress delta must be finite.", nameof(progressDelta));
            }

            if (targetId != null && (string.IsNullOrWhiteSpace(targetId) || targetId.Length > 128))
            {
                throw new ArgumentException("Target ID must contain 1-128 characters when present.", nameof(targetId));
            }

            if (realtime != null && realtime.ActionId != actionId)
            {
                throw new ArgumentException("Realtime receipt action ID must match the outer action ID.", nameof(realtime));
            }

            ActionId = actionId;
            EpisodeId = episodeId;
            StepId = stepId;
            Outcome = outcome;
            IssuedTimestampNanoseconds = issuedTimestampNanoseconds;
            ObservedTimestampNanoseconds = observedTimestampNanoseconds;
            Postcondition = postcondition;
            ProgressDelta = progressDelta;
            AuthoritativeObservationSequence = authoritativeObservationSequence;
            Retryable = retryable;
            Realtime = realtime;
            TargetId = targetId;
            ReasonClass = reasonClass;
        }

        /// <summary>Provider action identity.</summary>
        public string ActionId { get; }

        /// <summary>Logical episode identity.</summary>
        public Guid EpisodeId { get; }

        /// <summary>Authoritative post-state step identity.</summary>
        public ulong StepId { get; }

        /// <summary>Typed action outcome.</summary>
        public ActionOutcome Outcome { get; }

        /// <summary>Provider action issue timestamp.</summary>
        public ulong IssuedTimestampNanoseconds { get; }

        /// <summary>Provider post-state observation timestamp.</summary>
        public ulong ObservedTimestampNanoseconds { get; }

        /// <summary>Bounded postcondition status.</summary>
        public string Postcondition { get; }

        /// <summary>Optional bounded progress delta.</summary>
        public double? ProgressDelta { get; }

        /// <summary>Optional authoritative observation sequence.</summary>
        public ulong? AuthoritativeObservationSequence { get; }

        /// <summary>Whether a provider explicitly permits retry.</summary>
        public bool Retryable { get; }

        /// <summary>Optional typed realtime timing receipt.</summary>
        public RealtimeActionReceipt? Realtime { get; }

        /// <summary>Optional provider target identity associated with a refusal.</summary>
        public string? TargetId { get; }

        /// <summary>Optional refusal reason class.</summary>
        public RefusalReasonClass? ReasonClass { get; }
    }

    /// <summary>An immutable tensor payload using the GLR little-endian wire layout.</summary>
    public sealed class TensorBuffer
    {
        private readonly byte[] data;

        /// <summary>Create a copied tensor payload.</summary>
        public TensorBuffer(IEnumerable<long> shape, DType dtype, byte[] data)
        {
            if (shape == null)
            {
                throw new ArgumentNullException(nameof(shape));
            }

            var copiedShape = shape.ToArray();
            if (copiedShape.Any(dimension => dimension < 0))
            {
                throw new ArgumentOutOfRangeException(nameof(shape), "Dimensions cannot be negative.");
            }

            Shape = Array.AsReadOnly(copiedShape);
            DType = dtype;
            this.data = data?.ToArray() ?? throw new ArgumentNullException(nameof(data));
        }

        /// <summary>Tensor dimensions.</summary>
        public IReadOnlyList<long> Shape { get; }

        /// <summary>Tensor element type.</summary>
        public DType DType { get; }

        /// <summary>Copied contiguous tensor bytes.</summary>
        public byte[] Data => data.ToArray();
    }

    /// <summary>One flattened tensor specification.</summary>
    public sealed class TensorSpec
    {
        /// <summary>Create an immutable flattened tensor specification.</summary>
        public TensorSpec(
            string path,
            IEnumerable<long> shape,
            DType dtype,
            SpaceKind kind,
            double? minimum = null,
            double? maximum = null,
            string description = "")
        {
            if (string.IsNullOrWhiteSpace(path))
            {
                throw new ArgumentException("Path cannot be empty.", nameof(path));
            }

            if (shape == null)
            {
                throw new ArgumentNullException(nameof(shape));
            }

            var copiedShape = shape.ToArray();
            if (copiedShape.Any(dimension => dimension < -1))
            {
                throw new ArgumentOutOfRangeException(nameof(shape), "Only -1 may denote a dynamic dimension.");
            }

            if (minimum.HasValue && maximum.HasValue && minimum.Value > maximum.Value)
            {
                throw new ArgumentException("Minimum cannot exceed maximum.", nameof(minimum));
            }

            Path = path;
            Shape = Array.AsReadOnly(copiedShape);
            DType = dtype;
            Kind = kind;
            Minimum = minimum;
            Maximum = maximum;
            Description = description ?? throw new ArgumentNullException(nameof(description));
        }

        /// <summary>Dot-separated tensor-tree path.</summary>
        public string Path { get; }

        /// <summary>Tensor dimensions, where -1 denotes a dynamic dimension.</summary>
        public IReadOnlyList<long> Shape { get; }

        /// <summary>Tensor element type.</summary>
        public DType DType { get; }

        /// <summary>Tensor space semantics.</summary>
        public SpaceKind Kind { get; }

        /// <summary>Optional inclusive minimum.</summary>
        public double? Minimum { get; }

        /// <summary>Optional inclusive maximum.</summary>
        public double? Maximum { get; }

        /// <summary>Human-readable semantic description.</summary>
        public string Description { get; }
    }

    /// <summary>Immutable environment descriptor returned by a provider.</summary>
    public sealed class ProviderDescriptor
    {
        /// <summary>Create a provider descriptor.</summary>
        public ProviderDescriptor(
            string environmentId,
            IEnumerable<TensorSpec> observations,
            IEnumerable<TensorSpec> actions,
            IEnumerable<TensorSpec> actionMasks,
            TensorSpec reward,
            TensorSpec done,
            IEnumerable<string> capabilities,
            IReadOnlyDictionary<string, string>? metadata = null,
            RealtimeTimingContract? realtimeTiming = null)
        {
            if (string.IsNullOrWhiteSpace(environmentId))
            {
                throw new ArgumentException("Environment ID cannot be empty.", nameof(environmentId));
            }

            EnvironmentId = environmentId;
            Observations = CopyList(observations, nameof(observations));
            Actions = CopyList(actions, nameof(actions));
            ActionMasks = CopyList(actionMasks, nameof(actionMasks));
            Reward = reward ?? throw new ArgumentNullException(nameof(reward));
            Done = done ?? throw new ArgumentNullException(nameof(done));
            Capabilities = CopyList(capabilities, nameof(capabilities));
            Metadata = CopyDictionary(metadata ?? new Dictionary<string, string>());
            RealtimeTiming = realtimeTiming;
        }

        /// <summary>Stable public environment identity.</summary>
        public string EnvironmentId { get; }

        /// <summary>GLR environment protocol version.</summary>
        public string ProtocolVersion => HostProtocol.EnvironmentProtocolVersion;

        /// <summary>Flattened observation specifications.</summary>
        public IReadOnlyList<TensorSpec> Observations { get; }

        /// <summary>Flattened action specifications.</summary>
        public IReadOnlyList<TensorSpec> Actions { get; }

        /// <summary>Flattened action-mask specifications.</summary>
        public IReadOnlyList<TensorSpec> ActionMasks { get; }

        /// <summary>Reward tensor specification.</summary>
        public TensorSpec Reward { get; }

        /// <summary>Termination/truncation tensor specification.</summary>
        public TensorSpec Done { get; }

        /// <summary>Truthful capabilities proved by the provider.</summary>
        public IReadOnlyList<string> Capabilities { get; }

        /// <summary>Reviewed non-sensitive metadata.</summary>
        public IReadOnlyDictionary<string, string> Metadata { get; }

        /// <summary>Optional descriptor-level realtime timing bounds.</summary>
        public RealtimeTimingContract? RealtimeTiming { get; }

        private static IReadOnlyList<T> CopyList<T>(IEnumerable<T> source, string name)
        {
            if (source == null)
            {
                throw new ArgumentNullException(name);
            }

            return Array.AsReadOnly(source.ToArray());
        }

        private static IReadOnlyDictionary<string, string> CopyDictionary(
            IReadOnlyDictionary<string, string> source)
        {
            return new ReadOnlyDictionary<string, string>(
                source.ToDictionary(pair => pair.Key, pair => pair.Value));
        }
    }

    /// <summary>One semantic runtime event with strict JSON payload bytes.</summary>
    public sealed class ProviderEvent
    {
        private readonly byte[] payloadJsonUtf8;

        /// <summary>Create a copied event.</summary>
        public ProviderEvent(string name, ulong timestampNanoseconds, byte[] payloadJsonUtf8)
        {
            if (string.IsNullOrWhiteSpace(name))
            {
                throw new ArgumentException("Event name cannot be empty.", nameof(name));
            }

            Name = name;
            TimestampNanoseconds = timestampNanoseconds;
            this.payloadJsonUtf8 = payloadJsonUtf8?.ToArray()
                ?? throw new ArgumentNullException(nameof(payloadJsonUtf8));
        }

        /// <summary>Stable event name.</summary>
        public string Name { get; }

        /// <summary>Runtime event timestamp.</summary>
        public ulong TimestampNanoseconds { get; }

        /// <summary>Strict UTF-8 JSON object bytes.</summary>
        public byte[] PayloadJsonUtf8 => payloadJsonUtf8.ToArray();
    }

    /// <summary>Authoritative post-state returned by a provider.</summary>
    public sealed class ProviderTimeStep
    {
        private readonly byte[] infoJsonUtf8;

        /// <summary>Create a copied provider time step.</summary>
        public ProviderTimeStep(
            Guid episodeId,
            ulong stepId,
            ulong timestampNanoseconds,
            IReadOnlyDictionary<string, TensorBuffer> observation,
            TensorBuffer reward,
            TensorBuffer terminated,
            TensorBuffer truncated,
            IReadOnlyDictionary<string, TensorBuffer>? actionMask = null,
            IEnumerable<ProviderEvent>? events = null,
            byte[]? infoJsonUtf8 = null,
            ActionReceipt? actionReceipt = null)
        {
            EpisodeId = episodeId;
            StepId = stepId;
            TimestampNanoseconds = timestampNanoseconds;
            Observation = CopyTensorMap(observation, nameof(observation));
            Reward = reward ?? throw new ArgumentNullException(nameof(reward));
            Terminated = terminated ?? throw new ArgumentNullException(nameof(terminated));
            Truncated = truncated ?? throw new ArgumentNullException(nameof(truncated));
            ActionMask = CopyTensorMap(
                actionMask ?? new Dictionary<string, TensorBuffer>(),
                nameof(actionMask));
            Events = Array.AsReadOnly((events ?? Array.Empty<ProviderEvent>()).ToArray());
            this.infoJsonUtf8 = (infoJsonUtf8 ?? Array.Empty<byte>()).ToArray();
            if (actionReceipt != null && (actionReceipt.EpisodeId != episodeId || actionReceipt.StepId != stepId))
            {
                throw new ArgumentException(
                    "Action receipt must match the provider time step identity.",
                    nameof(actionReceipt));
            }

            ActionReceipt = actionReceipt;
        }

        /// <summary>Logical GLR episode identity.</summary>
        public Guid EpisodeId { get; }

        /// <summary>Monotonic step identity within the episode.</summary>
        public ulong StepId { get; }

        /// <summary>Authoritative post-state timestamp.</summary>
        public ulong TimestampNanoseconds { get; }

        /// <summary>Flattened observation tensors.</summary>
        public IReadOnlyDictionary<string, TensorBuffer> Observation { get; }

        /// <summary>Reward tensor.</summary>
        public TensorBuffer Reward { get; }

        /// <summary>Termination tensor.</summary>
        public TensorBuffer Terminated { get; }

        /// <summary>Truncation tensor.</summary>
        public TensorBuffer Truncated { get; }

        /// <summary>Flattened action-mask tensors.</summary>
        public IReadOnlyDictionary<string, TensorBuffer> ActionMask { get; }

        /// <summary>Semantic events.</summary>
        public IReadOnlyList<ProviderEvent> Events { get; }

        /// <summary>Optional typed receipt for the mutating action producing this state.</summary>
        public ActionReceipt? ActionReceipt { get; }

        /// <summary>Strict UTF-8 JSON object bytes for reviewed auxiliary info.</summary>
        public byte[] InfoJsonUtf8 => infoJsonUtf8.ToArray();

        private static IReadOnlyDictionary<string, TensorBuffer> CopyTensorMap(
            IReadOnlyDictionary<string, TensorBuffer> source,
            string name)
        {
            if (source == null)
            {
                throw new ArgumentNullException(name);
            }

            return new ReadOnlyDictionary<string, TensorBuffer>(
                source.ToDictionary(pair => pair.Key, pair => pair.Value));
        }
    }

    /// <summary>Authoritative result for reconnecting an interrupted logical episode.</summary>
    public sealed class ProviderResumeResult
    {
        /// <summary>Create a validated reconnect result.</summary>
        public ProviderResumeResult(
            ProviderTimeStep timestep,
            ulong committedStepId,
            ActionReconciliation? reconciliation = null)
        {
            Timestep = timestep ?? throw new ArgumentNullException(nameof(timestep));
            if (committedStepId != timestep.StepId)
            {
                throw new ArgumentException(
                    "Committed step ID must match the returned time step.",
                    nameof(committedStepId));
            }

            if (reconciliation != null
                && (reconciliation.EpisodeId != timestep.EpisodeId
                    || reconciliation.AuthoritativeStepId != committedStepId))
            {
                throw new ArgumentException(
                    "Reconciliation must match the returned authoritative cursor.",
                    nameof(reconciliation));
            }

            CommittedStepId = committedStepId;
            Reconciliation = reconciliation;
        }

        /// <summary>Authoritative post-state after reconnect.</summary>
        public ProviderTimeStep Timestep { get; }

        /// <summary>Last committed step known by the runtime.</summary>
        public ulong CommittedStepId { get; }

        /// <summary>Optional result for the in-flight action being reconciled.</summary>
        public ActionReconciliation? Reconciliation { get; }
    }

    /// <summary>Authoritative outcome for one action submitted before reconnect.</summary>
    public sealed class ActionReconciliation
    {
        /// <summary>Create an immutable action reconciliation record.</summary>
        public ActionReconciliation(
            Guid episodeId,
            ulong expectedStepId,
            ReconciliationOutcome outcome,
            ulong authoritativeStepId,
            ulong timestampNanoseconds,
            bool retryable = false)
        {
            if (expectedStepId == 0)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(expectedStepId),
                    "Expected step ID must be non-zero.");
            }

            EpisodeId = episodeId;
            ExpectedStepId = expectedStepId;
            Outcome = outcome;
            AuthoritativeStepId = authoritativeStepId;
            TimestampNanoseconds = timestampNanoseconds;
            Retryable = retryable;
        }

        /// <summary>Logical GLR episode identity.</summary>
        public Guid EpisodeId { get; }

        /// <summary>Step ID of the action submitted before reconnect.</summary>
        public ulong ExpectedStepId { get; }

        /// <summary>Authoritative action application outcome.</summary>
        public ReconciliationOutcome Outcome { get; }

        /// <summary>Authoritative post-state cursor.</summary>
        public ulong AuthoritativeStepId { get; }

        /// <summary>Runtime timestamp for the reconciliation decision.</summary>
        public ulong TimestampNanoseconds { get; }

        /// <summary>Whether the caller may safely retry the action.</summary>
        public bool Retryable { get; }
    }
}
