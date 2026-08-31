using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;

namespace GameLearningRuntime.Provider
{
    /// <summary>A physical reset request. Providers must reject it when reset is not truthful.</summary>
    public sealed class ResetRequest
    {
        /// <summary>Create an immutable reset request.</summary>
        public ResetRequest(ulong? seed, IReadOnlyDictionary<string, string>? options = null)
        {
            Seed = seed;
            Options = CopyOptions(options);
        }

        /// <summary>Optional deterministic seed.</summary>
        public ulong? Seed { get; }

        /// <summary>Reviewed string options.</summary>
        public IReadOnlyDictionary<string, string> Options { get; }

        internal static IReadOnlyDictionary<string, string> CopyOptions(
            IReadOnlyDictionary<string, string>? options)
        {
            return new ReadOnlyDictionary<string, string>(
                (options ?? new Dictionary<string, string>())
                    .ToDictionary(pair => pair.Key, pair => pair.Value));
        }
    }

    /// <summary>A truthful live-attach request that makes no physical reset claim.</summary>
    public sealed class AttachRequest
    {
        /// <summary>Create an immutable attach request.</summary>
        public AttachRequest(IReadOnlyDictionary<string, string>? options = null)
        {
            Options = ResetRequest.CopyOptions(options);
        }

        /// <summary>Reviewed string options.</summary>
        public IReadOnlyDictionary<string, string> Options { get; }
    }

    /// <summary>One episode- and step-fenced semantic action.</summary>
    public sealed class StepRequest
    {
        /// <summary>Create an immutable step request.</summary>
        public StepRequest(
            Guid episodeId,
            ulong expectedStepId,
            IReadOnlyDictionary<string, TensorBuffer> action)
        {
            if (expectedStepId == 0)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(expectedStepId),
                    "Expected step ID must be non-zero.");
            }

            if (action == null)
            {
                throw new ArgumentNullException(nameof(action));
            }

            EpisodeId = episodeId;
            ExpectedStepId = expectedStepId;
            Action = new ReadOnlyDictionary<string, TensorBuffer>(
                action.ToDictionary(pair => pair.Key, pair => pair.Value));
        }

        /// <summary>Current logical episode identity.</summary>
        public Guid EpisodeId { get; }

        /// <summary>Expected next step identity.</summary>
        public ulong ExpectedStepId { get; }

        /// <summary>Flattened semantic action tensors.</summary>
        public IReadOnlyDictionary<string, TensorBuffer> Action { get; }
    }

    /// <summary>
    /// Game-semantic provider implemented by a Unity plugin or another reviewed .NET host.
    /// A thin engine bootstrap must invoke mutations on the engine main thread.
    /// </summary>
    public interface IRuntimeProvider : IDisposable
    {
        /// <summary>Return immutable observation/action and capability metadata.</summary>
        ProviderDescriptor Describe();

        /// <summary>Perform a truthful physical reset and return step zero.</summary>
        ProviderTimeStep Reset(ResetRequest request);

        /// <summary>Attach to an existing world as a fresh logical episode.</summary>
        ProviderTimeStep Attach(AttachRequest request);

        /// <summary>Apply one fenced semantic action and return authoritative post-state.</summary>
        ProviderTimeStep Step(StepRequest request);
    }

    /// <summary>Engine-owned dispatcher used by a thin bootstrap to marshal provider calls.</summary>
    public interface IEngineThreadDispatcher
    {
        /// <summary>Execute one bounded call on the engine thread and return its result.</summary>
        T Invoke<T>(Func<T> operation);
    }
}
