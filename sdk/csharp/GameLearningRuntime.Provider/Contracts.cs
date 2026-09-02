using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;

namespace GameLearningRuntime.Provider
{
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
            bool retryable = false)
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
            IReadOnlyDictionary<string, string>? metadata = null)
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
}
