using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Mathematics;

public class CommonUtils
{
    public static string FashionMnistDataPath = "/NN/Fashion-mnist/Datasets";
    public static string CartPoleAgentDataPath = "/RL/Env/Cartpole/Datasets";

    public static Unity.Mathematics.Random random = new Unity.Mathematics.Random();

    public static void InitRandom(int seed)
    {
        random.InitState((uint)seed);
    }

    public static float RandomValue()
    {
        return random.NextFloat();
    }



#if UNITY_EDITOR
    public static UnityEditor.EditorWindow editorWindow;
    public static UnityEditor.EditorWindow GetMainGameView()
    {
        var assembly = typeof(UnityEditor.EditorWindow).Assembly;
        var type = assembly.GetType("UnityEditor.GameView");
        var gameview = UnityEditor.EditorWindow.GetWindow(type);
        return gameview;
    }
    public static void BeginCaptureRenderDoc()
    {
        editorWindow = GetMainGameView();

        UnityEditorInternal.RenderDoc.BeginCaptureRenderDoc(editorWindow);
    }
    public static void EndCaptureRenderDoc()
    {
        UnityEditorInternal.RenderDoc.EndCaptureRenderDoc(editorWindow);
    }
#else
    public static void BeginCaptureRenderDoc() { }
    public static void EndCaptureRenderDoc() { }
#endif
}
