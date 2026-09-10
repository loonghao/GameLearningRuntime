using System;
using System.Collections.Generic;
using System.Threading;
using GameLearningRuntime.Provider;
using UnityEngine;

namespace GameLearningRuntime.Unity
{
    // Source-integrated sample: only the explicitly supplied Transform is owned.
    // Each decision moves one unit; this is not a physics/frame-step provider.
    public sealed class TransformProvider : IRuntimeProvider
    {
        private readonly Transform target;
        private readonly int threadId = Thread.CurrentThread.ManagedThreadId;
        private Guid episode;
        private ulong cursor;
        private bool closed;

        public TransformProvider(Transform target)
        {
            this.target = target != null ? target : throw new ArgumentNullException(nameof(target));
        }

        private void Check()
        {
            if (closed) throw new ObjectDisposedException(nameof(TransformProvider));
            if (Thread.CurrentThread.ManagedThreadId != threadId)
                throw new InvalidOperationException("Provider requires its engine thread");
            if (target == null) throw new InvalidOperationException("Target was destroyed");
        }

        public ProviderDescriptor Describe()
        {
            Check();
            return new ProviderDescriptor("glr.unity.transform-v1",
                new[] { new TensorSpec("position", new long[] { 1 }, DType.Float32, SpaceKind.Continuous) },
                new[] { new TensorSpec("move", new long[] { 1 }, DType.Int32, SpaceKind.Discrete, -1, 1) },
                Array.Empty<TensorSpec>(),
                new TensorSpec("reward", new long[] { 1 }, DType.Float32, SpaceKind.Continuous),
                new TensorSpec("done", new long[] { 1 }, DType.Bool, SpaceKind.Binary),
                new[] { "reset", "live-attach", "step", "semantic-observation", "native-action" });
        }

        public ProviderTimeStep Reset(ResetRequest request)
        {
            Check();
            if (request.Options.Count != 0) throw new ArgumentException("Unknown reset options");
            target.position = Vector3.zero;
            episode = Guid.NewGuid();
            cursor = 0;
            return Observe();
        }

        public ProviderTimeStep Attach(AttachRequest request)
        {
            Check();
            if (request.Options.Count != 0) throw new ArgumentException("Unknown attach options");
            if (Math.Abs(target.position.x) >= 3) throw new InvalidOperationException("Reset the terminal sample first");
            episode = Guid.NewGuid();
            cursor = 0;
            return Observe();
        }

        public ProviderTimeStep Step(StepRequest request)
        {
            Check();
            if (episode == Guid.Empty || request.EpisodeId != episode || request.ExpectedStepId != cursor + 1)
                throw new InvalidOperationException("Stale episode or step");
            if (Math.Abs(target.position.x) >= 3) throw new InvalidOperationException("Episode is done");
            if (request.Timing != null || request.Lease != null || request.CancellationToken != null)
                throw new ArgumentException("Realtime scheduling is not supported");
            if (request.Action.Count != 1 || !request.Action.TryGetValue("move", out var action)
                || action.DType != DType.Int32 || action.Shape.Count != 1 || action.Shape[0] != 1
                || action.Data.Length != 4)
                throw new ArgumentException("Expected one int32 move tensor");
            var bytes = action.Data;
            if (!BitConverter.IsLittleEndian) Array.Reverse(bytes);
            int move = BitConverter.ToInt32(bytes, 0);
            if (move < -1 || move > 1) throw new ArgumentOutOfRangeException(nameof(request));
            target.position += Vector3.right * move;
            cursor++;
            return Observe();
        }

        private ProviderTimeStep Observe()
        {
            float position = target.position.x;
            bool done = Math.Abs(position) >= 3;
            return new ProviderTimeStep(episode, cursor, (ulong)(DateTime.UtcNow.Ticks - 621355968000000000L) * 100,
                new Dictionary<string, TensorBuffer> { { "position", Float(position) } },
                Float(done ? 1 : 0),
                new TensorBuffer(new long[] { 1 }, DType.Bool, new byte[] { (byte)(done ? 1 : 0) }),
                new TensorBuffer(new long[] { 1 }, DType.Bool, new byte[] { 0 }));
        }

        private static TensorBuffer Float(float value)
        {
            var bytes = BitConverter.GetBytes(value);
            if (!BitConverter.IsLittleEndian) Array.Reverse(bytes);
            return new TensorBuffer(new long[] { 1 }, DType.Float32, bytes);
        }

        public InputLeaseReceipt Lease(InputLeaseRequest request)
        {
            throw new NotSupportedException("Semantic actions do not expose input leases");
        }

        public void Dispose() { closed = true; }
    }
}
