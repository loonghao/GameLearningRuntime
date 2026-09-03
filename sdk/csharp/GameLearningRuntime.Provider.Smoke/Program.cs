using System;
using GameLearningRuntime.Provider;

if (HostProtocol.Schema != "glr.host.v1")
{
    throw new InvalidOperationException("C# provider SDK schema drifted");
}

var tensor = new TensorBuffer(new long[] { 1 }, DType.Int64, new byte[8]);
if (tensor.Shape.Count != 1 || tensor.Data.Length != 8)
{
    throw new InvalidOperationException("C# tensor contract failed");
}

var checkpoint = new CheckpointContract(
    "glr.v1",
    new string('1', 64),
    new string('2', 64),
    new string('3', 64));
var manifest = new CheckpointManifest(
    "policy.ckpt",
    new string('4', 64),
    8,
    checkpoint,
    new string('5', 64));
if (manifest.Contract.ProtocolVersion != "glr.v1" || manifest.CheckpointSizeBytes != 8)
{
    throw new InvalidOperationException("C# checkpoint contract failed");
}

var identity = new RuntimeIdentity("synthetic-counter", "0.10.0");
var health = new RuntimeHealth(
    identity,
    RuntimeHealthStatus.Ready,
    10,
    acceptingNewSessions: true,
    activeSessions: 1);
if (health.Identity.RuntimeId != identity.RuntimeId || !health.AcceptingNewSessions)
{
    throw new InvalidOperationException("C# runtime health contract failed");
}

Console.WriteLine($"{HostProtocol.Schema} provider-sdk-ok");
