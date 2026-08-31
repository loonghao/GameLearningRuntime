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

Console.WriteLine($"{HostProtocol.Schema} provider-sdk-ok");
