using System;
using System.Collections.Generic;
using System.IO;
using GameLearningRuntime.Provider;
using GameLearningRuntime.Unity;
using UnityEngine;

public static class PlayerSmoke
{
    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    private static void Run()
    {
        if (Array.IndexOf(Environment.GetCommandLineArgs(), "--glr-smoke") < 0) return;
        var target = new GameObject("GLR owned player sample");
        try
        {
            using (var provider = new TransformProvider(target.transform))
            {
                var initial = provider.Reset(new ResetRequest(7));
                var action = new Dictionary<string, TensorBuffer> {
                    { "move", new TensorBuffer(new long[] { 1 }, DType.Int32, new byte[] { 1, 0, 0, 0 }) }
                };
                for (ulong step = 1; step <= 3; step++)
                {
                    var state = provider.Step(new StepRequest(initial.EpisodeId, step, action));
                    if (target.transform.position.x != step || state.StepId != step)
                        throw new Exception("Engine post-state mismatch");
                    bool refused = false;
                    try { provider.Step(new StepRequest(initial.EpisodeId, step, action)); }
                    catch (InvalidOperationException) { refused = true; }
                    if (!refused) throw new Exception("Stale step was accepted");
                }
                provider.Reset(new ResetRequest(7));
                if (target.transform.position != Vector3.zero) throw new Exception("Reset failed");
            }
#if ENABLE_IL2CPP
            const string backend = "il2cpp";
#else
            const string backend = "mono";
#endif
            File.WriteAllText(Path.Combine(Application.dataPath, "../glr-player-smoke.json"),
                "{\"engine\":\"unity\",\"backend\":\"" + backend + "\",\"steps\":3,\"reset\":true,\"stale_rejected\":true}");
            Application.Quit(0);
        }
        catch (Exception error) { Debug.LogException(error); Application.Quit(1); }
        finally { UnityEngine.Object.Destroy(target); }
    }
}
