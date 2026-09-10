using System;
using System.IO;
using UnityEngine;

public sealed class ExternalSmoke : MonoBehaviour
{
    private int count;

    private void Start() { WriteState(); }

    private void WriteState()
    {
        File.WriteAllText(Path.Combine(Application.dataPath, "../glr-external-state.json"),
            "{\"count\":" + count + "}");
    }

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    private static void StartSample()
    {
        if (Array.IndexOf(Environment.GetCommandLineArgs(), "--glr-external") < 0) return;
        new GameObject("GLR external input sample").AddComponent<ExternalSmoke>();
    }

    private void OnGUI()
    {
        GUI.Label(new Rect(20, 20, 400, 40), "GLR external sample: " + count);
        if (GUI.Button(new Rect(Screen.width / 2 - 80, Screen.height / 2 - 30, 160, 60), "Advance"))
        {
            count++;
            WriteState();
        }
    }
}
