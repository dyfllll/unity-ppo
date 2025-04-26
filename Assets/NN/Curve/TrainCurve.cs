using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityNN;

public class TrainCurve : MonoBehaviour
{
    public NetArgs args;
    public NeuralNet neuralNet;
    public Dataset dataset;
    public ActivationFunc.Type activationFunc = ActivationFunc.Type.ReLU;
    public int hiddenLayerNodeCount = 64;

    public int paramCount = 10;
    public float[] paramArray;
    public int[] paramType;
    public float[] offsetArray;

    public float range = 1;


    public bool drawReal = true;
    public bool drawPredict = true;

    public int batch = 32;
    public int pointCount = 100;


    private float[] inputArray;
    private float[] outputArray;
    private float[] lossArray;
    private float curError = 1e10f;
    public float targetError = 0.001f;
    public int curEpochs = 0;
    public float learningRate = 0.01f;


    private float[] drawXArray;
    private float[] drawRealYArray;
    private float[] drawPredYArray;

    public void Start()
    {
        paramArray = new float[paramCount];
        paramType = new int[paramCount];
        offsetArray = new float[paramCount];
        for (int i = 0; i < paramCount; i++)
        {
            paramArray[i] = Random.Range(-1.0f, 1.0f);
            paramType[i] = Random.Range(0, 3);
            offsetArray[i] = Random.Range(-1.0f, 1.0f) * Mathf.PI;
        }

        int maxBatch = pointCount;

        inputArray = new float[batch];
        outputArray = new float[batch];
        lossArray = new float[batch];


        dataset = new Dataset();
        dataset.batchSize = maxBatch;
        dataset.inputCount = 1;
        dataset.outputCount = 1;
        dataset.inputBuffer = new ComputeBuffer(1 * maxBatch, sizeof(float));
        dataset.outputBuffer = new ComputeBuffer(1 * maxBatch, sizeof(float));

        neuralNet = new NeuralNet(args, new List<Layer>() {
            new Linear(hiddenLayerNodeCount),
            new ActivationFunc(activationFunc),
            new Linear(hiddenLayerNodeCount),
            new ActivationFunc(activationFunc),
            new Linear(1),
        }, dataset);
        neuralNet.Init();

        InitDrawArray();

    }
    private void OnDestroy()
    {
        dataset?.Release();
        neuralNet?.Release();
    }

    public void Update()
    {
        if (curError < targetError) return;

        float minX = -Mathf.PI * range;
        float maxX = Mathf.PI * range;
        for (int i = 0; i < batch; i++)
        {
            float x = Random.Range(minX, maxX);
            float y = GetValue(x);
            inputArray[i] = x;
            outputArray[i] = y;
        }

        Optimizer.UpdateParam(curEpochs + 1, learningRate);

        dataset.batchSize = batch;
        dataset.inputBuffer.SetData(inputArray);
        dataset.outputBuffer.SetData(outputArray);

        neuralNet.Forward();

        if (args.lossType == LossFunc.Type.None)
        {
            float[] tempArray = new float[batch * 1];
            neuralNet.layers[^1].dataBuffer.GetData(tempArray);
            for (int b = 0; b < batch; b++)
            {
                float diff = tempArray[b] - outputArray[b];
                lossArray[b] = diff * diff;
            }
            float[] dOutputArray = new float[batch * 1];
            for (int b = 0; b < batch; b++)
            {
                dOutputArray[b] = tempArray[b] - outputArray[b];
            }
            neuralNet.swapBuffer.dOutputBuffer.SetData(dOutputArray);
        }
        else
        {
            neuralNet.loss.dataBuffer.GetData(lossArray);
        }

        neuralNet.Backward();




        curError = 0;
        foreach (var item in lossArray)
        {
            curError += item;
        }
        curError /= lossArray.Length;

        Debug.Log(curError);
        curEpochs++;

        Predict();
    }





    private void Predict()
    {
        dataset.batchSize = pointCount;
        dataset.inputBuffer.SetData(drawXArray);
        neuralNet.Predict();
        neuralNet.layers[^1].dataBuffer.GetData(drawPredYArray);
    }


    private float GetValue(float x)
    {
        float sum = 0;

        for (int i = 0; i < paramCount; i++)
        {
            switch (paramType[i])
            {
                case 0:
                    sum += paramArray[i] * Mathf.Sin(x);
                    break;
                case 1:
                    sum += paramArray[i] * Mathf.Cos(x);
                    break;
                case 2:
                    sum = Mathf.Max(0, sum);
                    break;
                default:
                    break;
            }

        }
        return sum;

    }


    private void InitDrawArray()
    {
        float minX = -Mathf.PI * range;
        float maxX = Mathf.PI * range;

        float step = (maxX - minX) / (pointCount - 1);

        drawXArray = new float[pointCount];
        drawRealYArray = new float[pointCount];
        drawPredYArray = new float[pointCount];
        for (int i = 0; i < pointCount; i++)
        {
            float x = i * step + minX;
            float y = GetValue(x);

            drawXArray[i] = x;
            drawRealYArray[i] = y;
            drawPredYArray[i] = 0;
        }
    }


    private void OnDrawGizmos()
    {
        if (!Application.isPlaying) return;

        if (drawReal)
        {

            for (int i = 1; i < pointCount; i++)
            {
                float lastX = drawXArray[i - 1];
                float lastY = drawRealYArray[i - 1];

                float x = drawXArray[i];
                float y = drawRealYArray[i];

                Gizmos.color = Color.white;
                Gizmos.DrawLine(new Vector3(lastX, lastY, 0), new Vector3(x, y, 0));
            }

        }

        if (drawPredict)
        {

            for (int i = 1; i < pointCount; i++)
            {
                float lastX = drawXArray[i - 1];
                float lastY = drawPredYArray[i - 1];

                float x = drawXArray[i];
                float y = drawPredYArray[i];

                Gizmos.color = Color.green;
                Gizmos.DrawLine(new Vector3(lastX, lastY, 0), new Vector3(x, y, 0));
            }

        }

    }


}
