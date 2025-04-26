using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using System.Text;
using UnityNN;
using UnityNN.Data;

public class TrainFashionMnist : MonoBehaviour
{
    public string dataTestName = "fashion-mnist_test";
    public string dataTrainName = "fashion-mnist_train";
    public string trainResultName = "train-wb-data";

    public NeuralNet neuralNet;
    public NetArgs netArgs;


    public float targetError = 0.001f;
    public float curError;
    public int batch = 64;
    public float learningRate = 0.01f;

    public int curEpochs = 0;
    public int maxEpochs = 200;

    public bool isTraining;

    public int trainBatchIndex = 0;
    public int trainBatchTotal = 0;

    public int testBatchIndex = 0;
    public int testBatchTotal = 0;


    private Dataset dataset;

    private byte[] rawTrainArray;
    private byte[] rawTestArray;
    public ComputeBuffer rawTrainBuffer;
    public ComputeBuffer rawTestBuffer;
    private int lineDataSize = 785; //一行数据大小 类型1+28*28 像素大小
    public int trainDataCount;
    public int testDataCount;
    public int dataInputCount = 784;
    public int dataOutputCount = 10;

    private string[] class_names = new string[]
     {"T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot"
     };

    private float[] lossArray;
    private float[] predictArray;

    public bool saveWeightAndBias = false;

    public int predictCount;
    public int predictCorrect;


    // Start is called before the first frame update
    void Start()
    {

        LoadRawData(dataTrainName, ref trainDataCount, ref rawTrainArray, ref rawTrainBuffer);
        LoadRawData(dataTestName, ref testDataCount, ref rawTestArray, ref rawTestBuffer);

        trainBatchTotal = trainDataCount / batch;
        testBatchTotal = testDataCount / batch;

        dataset = new Dataset();
        dataset.batchSize = batch;
        dataset.inputCount = dataInputCount;
        dataset.outputCount = dataOutputCount;
        dataset.inputBuffer = new ComputeBuffer(dataInputCount * batch, sizeof(float));
        dataset.outputBuffer = new ComputeBuffer(dataOutputCount * batch, sizeof(float));


        List<Layer> layers = new List<Layer>();
        layers.Add(new Linear(512));
        layers.Add(new ActivationFunc(ActivationFunc.Type.ReLU));
        layers.Add(new Linear(512));
        layers.Add(new ActivationFunc(ActivationFunc.Type.ReLU));
        //layers.Add(new Dropout(0.1f));
        layers.Add(new Linear(10));
        layers.Add(new ActivationFunc(ActivationFunc.Type.Softmax));

        Optimizer.UpdateParam(curEpochs, learningRate);

        neuralNet = new NeuralNet(netArgs, layers, dataset);
        neuralNet.Init();

        lossArray = new float[batch];
        curError = float.MaxValue;
        predictArray = new float[batch * dataOutputCount];

        if (netArgs.netType == NeuralNet.Type.Predict || netArgs.initType == Initialize.Type.None)
        {
            string filePath = Application.dataPath + $"{CommonUtils.FashionMnistDataPath}/{trainResultName}.bin";
            using FileStream fs = new FileStream(filePath, FileMode.Open);
            neuralNet.LoadWeightAndBias(fs);
        }
    }


    private void LoadRawData(string dataName, ref int dataCount, ref byte[] array, ref ComputeBuffer computeBuffer)
    {
        string filePath = Application.dataPath + $"{CommonUtils.FashionMnistDataPath}/{dataName}.bin";
        byte[] dataArray = File.ReadAllBytes(filePath);
        dataCount = dataArray.Length / lineDataSize;
        int size = Mathf.CeilToInt(1.0f * dataArray.Length / sizeof(uint));
        computeBuffer = new ComputeBuffer(size, sizeof(uint), ComputeBufferType.Raw);
        computeBuffer.SetData(dataArray);
        array = dataArray;
    }

    private void RandomRawData(int dataCount, ref byte[] array, ref ComputeBuffer computeBuffer)
    {
        int[] indexMap = new int[dataCount];
        for (int i = 0; i < dataCount; i++)
        {
            indexMap[i] = i;
        }
        for (int i = dataCount - 1; i >= 0; i--)
        {
            int ti = Random.Range(0, i);
            int temp = indexMap[i];
            indexMap[i] = indexMap[ti];
            indexMap[ti] = temp;
        }

        byte[] resultArray = new byte[array.Length];

        for (int i = 0; i < dataCount; i++)
        {
            int ri = indexMap[i];
            int wi = i;
            System.Buffer.BlockCopy(array, ri * lineDataSize, resultArray, wi * lineDataSize, lineDataSize);
        }

        computeBuffer.SetData(resultArray);
        array = resultArray;
    }


    // Update is called once per frame
    void Update()
    {
        if (isTraining && netArgs.netType == NeuralNet.Type.Train && curError > targetError)
        {
            curEpochs = trainBatchIndex / trainBatchTotal + 1;
            int batchIndex = trainBatchIndex % trainBatchTotal;
            Optimizer.UpdateParam(curEpochs, learningRate);

            FashionMnist.TransformRawToInput(rawTrainBuffer, dataset.inputBuffer, batchIndex, dataInputCount, batch, lineDataSize);
            FashionMnist.TransformRawToOutput(rawTrainBuffer, dataset.outputBuffer, batchIndex, dataOutputCount, batch, lineDataSize);

            neuralNet.Forward();
            neuralNet.Backward();

            neuralNet.loss.dataBuffer.GetData(lossArray);

            curError = 0;
            foreach (var item in lossArray)
            {
                curError += item;
            }
            curError /= lossArray.Length;


            Debug.Log($"epoch:{curEpochs}  batch:{batchIndex + 1}/{trainBatchTotal} curError:{curError}");

            trainBatchIndex++;


            if (saveWeightAndBias)
            {
                using FileStream fs = new FileStream(Application.dataPath + $"/Train{System.DateTime.Now.ToString("yyyy-MM-dd-HH-mm-ss")}.bin", FileMode.OpenOrCreate);
                neuralNet.SaveWeightAndBias(fs);
                saveWeightAndBias = false;
            }

            if (trainBatchIndex % trainBatchTotal == trainBatchTotal - 1)
            {
                RandomRawData(trainDataCount, ref rawTrainArray, ref rawTrainBuffer);
            }
        }



        if (netArgs.netType == NeuralNet.Type.Predict && testBatchIndex < testBatchTotal)
        {
            int batchIndex = testBatchIndex % testBatchTotal;

            FashionMnist.TransformRawToInput(rawTestBuffer, dataset.inputBuffer, batchIndex, dataInputCount, batch, lineDataSize);
            FashionMnist.TransformRawToOutput(rawTestBuffer, dataset.outputBuffer, batchIndex, dataOutputCount, batch, lineDataSize);
            neuralNet.Predict();
            neuralNet.layers[neuralNet.layers.Count - 1].dataBuffer.GetData(predictArray);

            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < batch; i++)
            {
                int actualIndex = rawTestArray[(batchIndex * batch + i) * lineDataSize + 0];
                int predictIndex = GetPredictIndex(predictArray, i);

                string actualClass = class_names[actualIndex];
                string predictClass = class_names[predictIndex];


                if (actualIndex == predictIndex)
                    predictCorrect++;
                predictCount++;

                sb.AppendLine($"actual:{actualClass} predict:{predictClass}");
            }


            Debug.Log(sb.ToString());
            Debug.Log($"正确率:{1.0f * predictCorrect / predictCount * 100.0f}%");

            testBatchIndex++;
        }
    }

    private int GetPredictIndex(float[] array, int index)
    {
        int maxIndex = 0;
        float maxValue = -1e10f;
        int classCount = class_names.Length;
        for (int i = 0; i < classCount; i++)
        {
            float val = array[index * classCount + i];
            if (val > maxValue)
            {
                maxValue = val;
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    private void OnDestroy()
    {
        rawTrainBuffer?.Release();
        rawTestBuffer?.Release();
        neuralNet?.Release();
        dataset?.Release();
    }
}
