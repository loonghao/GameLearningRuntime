namespace GameLearningRuntime.Provider
{
    /// <summary>Stable identifiers shared with the Rust Runtime Host.</summary>
    public static class HostProtocol
    {
        /// <summary>The first Runtime Host envelope schema.</summary>
        public const string Schema = "glr.host.v1";

        /// <summary>The learner-facing environment protocol carried by the host.</summary>
        public const string EnvironmentProtocolVersion = "1.0";
    }

    /// <summary>Tensor element types supported by the GLR v1 wire contract.</summary>
    public enum DType
    {
        /// <summary>One-byte boolean with only zero and one allowed.</summary>
        Bool,
        /// <summary>Unsigned eight-bit integer.</summary>
        UInt8,
        /// <summary>Little-endian signed 32-bit integer.</summary>
        Int32,
        /// <summary>Little-endian signed 64-bit integer.</summary>
        Int64,
        /// <summary>Little-endian IEEE 754 32-bit float.</summary>
        Float32,
        /// <summary>Little-endian IEEE 754 64-bit float.</summary>
        Float64,
    }

    /// <summary>Semantic meaning of a tensor space.</summary>
    public enum SpaceKind
    {
        /// <summary>Continuous numeric values.</summary>
        Continuous,
        /// <summary>One discrete choice.</summary>
        Discrete,
        /// <summary>Multiple discrete choices.</summary>
        MultiDiscrete,
        /// <summary>Boolean values.</summary>
        Binary,
    }

    /// <summary>Authoritative result used to reconcile an action after reconnect.</summary>
    public enum ReconciliationOutcome
    {
        /// <summary>The runtime confirms that the action was applied.</summary>
        Applied,
        /// <summary>The runtime confirms that the action was not applied.</summary>
        NotApplied,
        /// <summary>The runtime cannot determine whether the action was applied.</summary>
        Unknown,
    }
}
