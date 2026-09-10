using System;
using System.Collections.Generic;
using System.IO;
using GameLearningRuntime.Provider;
using GameLearningRuntime.Unity;
using UnityEditor;
using UnityEngine;

public static class ProviderSmoke
{
    public static void BuildMono() { Build(ScriptingImplementation.Mono2x); }
    public static void BuildIl2Cpp() { Build(ScriptingImplementation.IL2CPP); }

    private static void Build(ScriptingImplementation backend)
    {
        var scene = UnityEditor.SceneManagement.EditorSceneManager.NewScene(
            UnityEditor.SceneManagement.NewSceneSetup.EmptyScene);
        UnityEditor.SceneManagement.EditorSceneManager.SaveScene(scene, "Assets/Smoke.unity");
        PlayerSettings.SetScriptingBackend(BuildTargetGroup.Standalone, backend);
        var report = BuildPipeline.BuildPlayer(new BuildPlayerOptions {
            scenes = new[] { "Assets/Smoke.unity" },
            locationPathName = "Builds/" + backend + "/GlrSmoke.exe",
            target = BuildTarget.StandaloneWindows64,
            options = BuildOptions.None
        });
        EditorApplication.Exit(report.summary.result == UnityEditor.Build.Reporting.BuildResult.Succeeded ? 0 : 1);
    }

    public static void Run()
    {
        var target = new GameObject("GLR owned sample");
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
                var reset = provider.Reset(new ResetRequest(7));
                if (target.transform.position != Vector3.zero || reset.EpisodeId == initial.EpisodeId)
                    throw new Exception("Physical reset failed");
            }
            File.WriteAllText("glr-unity-smoke.json", "{\"engine\":\"unity\",\"steps\":3,\"reset\":true,\"stale_rejected\":true}");
            EditorApplication.Exit(0);
        }
        catch (Exception error)
        {
            Debug.LogException(error);
            EditorApplication.Exit(1);
        }
        finally { UnityEngine.Object.DestroyImmediate(target); }
    }
}
