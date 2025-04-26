using Cysharp.Threading.Tasks;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using UnityEngine.Rendering;
using Unity.Collections;
using System;
namespace UnityNN
{

    [System.Serializable]
    public class NetArgs
    {
        public NeuralNet.Type netType;
        public Initialize.Type initType;
        public Optimizer.Type optimType;
        public LossFunc.Type lossType;

        //梯度裁剪
        public bool clipGrad = false;
        public float maxGradNorm = 0.5f;
    }



    public class NeuralNet
    {
        public enum Type
        {
            Train, Predict,
        }

        public NetArgs args;

        public List<Layer> layers;

        public Layer loss;

        public Dataset dataset;

        public LayerSwapBuffer swapBuffer;

        public ClipGradNorm clipGradNorm;

        public NeuralNet(NetArgs args, List<Layer> layers, Dataset dataset)
        {
            this.args = args;
            this.layers = layers;
            this.loss = args.lossType != LossFunc.Type.None ? new LossFunc(args.lossType) : null;
            this.dataset = dataset;
            this.swapBuffer = CreateSwapBuffer();
        }


        private LayerSwapBuffer CreateSwapBuffer()
        {
            int maxNodeCount = 0;
            for (int i = 0; i < layers.Count; i++)
            {
                maxNodeCount = Mathf.Max(maxNodeCount, layers[i].nodeCount);
            }
            LayerSwapBuffer swapBuffer = new LayerSwapBuffer();
            swapBuffer.dInputBuffer = new ComputeBuffer(dataset.batchSize * maxNodeCount, sizeof(float));
            swapBuffer.dOutputBuffer = new ComputeBuffer(dataset.batchSize * maxNodeCount, sizeof(float));
            if (!args.clipGrad)
            {
                swapBuffer.dWeightBuffer = new ComputeBuffer(maxNodeCount * maxNodeCount, sizeof(float));
                swapBuffer.dBiasBuffer = new ComputeBuffer(maxNodeCount, sizeof(float));
            }
            return swapBuffer;
        }



        public void Init()
        {
            for (int i = 0; i < layers.Count; i++)
            {
                Layer prev = i == 0 ? null : layers[i - 1];
                layers[i].Init(args, i == 0 ? null : layers[i - 1], dataset, swapBuffer);
            }
            loss?.Init(args, layers[^1], dataset, swapBuffer);

            if (args.clipGrad)
            {
                clipGradNorm = new ClipGradNorm(this);
            }
        }


        public int GetWeightAndBiasSize()
        {
            int count = 0;
            for (int i = 0; i < layers.Count; i++)
            {
                Linear layer = layers[i] as Linear;
                if (layer != null)
                {
                    count += layer.weightCount;
                    count += layer.biasCount;
                }
            }
            return count;
        }


        public void SaveWeightAndBias(Stream stream)
        {
            int bufferSize = 1024;
            byte[] buffer = new byte[bufferSize];
            for (int i = 0; i < layers.Count; i++)
            {
                Linear layer = layers[i] as Linear;
                if (layer != null)
                {
                    WriteBufferToStream(stream, layer.weightBuffer, layer.weightCount, buffer, bufferSize);
                    WriteBufferToStream(stream, layer.biasBuffer, layer.biasCount, buffer, bufferSize);
                }
            }
        }

        private void WriteBufferToStream(Stream stream, ComputeBuffer gpuBuffer, int weightCount, byte[] buffer, int bufferSize)
        {
            float[] array = new float[weightCount];
            gpuBuffer.GetData(array);


            int totalSize = sizeof(float) * weightCount;
            int currentSize = 0;
            while (currentSize < totalSize)
            {
                int remainSize = totalSize - currentSize;
                int everySize = remainSize > bufferSize ? bufferSize : remainSize;
                Buffer.BlockCopy(array, currentSize, buffer, 0, everySize);
                stream.Write(buffer, 0, everySize);
                currentSize += everySize;
            }
        }


        public async UniTask SaveWeightAndBiasAsync(Stream stream)
        {
            int bufferSize = 1024;
            byte[] buffer = new byte[bufferSize];
            for (int i = 0; i < layers.Count; i++)
            {
                Linear layer = layers[i] as Linear;
                if (layer != null)
                {
                    await WriteBufferToStreamAsync(stream, layer.weightBuffer, layer.weightCount, buffer, bufferSize);
                    await WriteBufferToStreamAsync(stream, layer.biasBuffer, layer.biasCount, buffer, bufferSize);
                }
            }
        }


        private async UniTask WriteBufferToStreamAsync(Stream stream, ComputeBuffer gpuBuffer, int weightCount, byte[] buffer, int bufferSize)
        {
            var result = await AsyncGPUReadback.Request(gpuBuffer, weightCount, 0);
            var array = result.GetData<byte>();

            int totalSize = sizeof(float) * weightCount;
            int currentSize = 0;
            while (currentSize < totalSize)
            {
                int remainSize = totalSize - currentSize;
                int everySize = remainSize > bufferSize ? bufferSize : remainSize;
                NativeArray<byte>.Copy(array, currentSize, buffer, 0, everySize);
                stream.Write(buffer, 0, everySize);
                currentSize += everySize;
            }
        }


        public void LoadWeightAndBias(Stream stream)
        {
            int bufferSize = 1024;
            byte[] buffer = new byte[bufferSize];
            for (int i = 0; i < layers.Count; i++)
            {
                Linear layer = layers[i] as Linear;
                if (layer != null)
                {
                    ReadBufferFromStream(stream, layer.weightBuffer, layer.weightCount, buffer, bufferSize);
                    ReadBufferFromStream(stream, layer.biasBuffer, layer.biasCount, buffer, bufferSize);
                }
            }
        }

        private void ReadBufferFromStream(Stream stream, ComputeBuffer weightBuffer, int weightCount, byte[] buffer, int bufferSize)
        {
            float[] array = new float[weightCount];

            int totalSize = sizeof(float) * weightCount;
            int currentSize = 0;
            while (currentSize < totalSize)
            {
                int remainSize = totalSize - currentSize;
                int everySize = remainSize > bufferSize ? bufferSize : remainSize;
                stream.Read(buffer, 0, everySize);
                Buffer.BlockCopy(buffer, 0, array, currentSize, everySize);
                currentSize += everySize;
            }

            weightBuffer.SetData(array);
        }





        public void Predict()
        {
            for (int i = 0; i < layers.Count; i++)
            {
                layers[i].Forward();
            }
        }

        public void Forward()
        {
            //if (args.netType == Type.Predict) return;
            Predict();
            loss?.Forward();
        }


        public void Backward()
        {
            if (args.netType == Type.Predict) return;
            loss?.Backward();
            for (int i = layers.Count - 1; i >= 0; i--)
            {
                layers[i].Backward();
            }

            if (args.clipGrad)
            {
                clipGradNorm.Clip(args.maxGradNorm);
                for (int i = layers.Count - 1; i >= 0; i--)
                {
                    if (layers[i] is Linear)
                        (layers[i] as Linear).UpdateWeightBias();
                }
            }



        }

        public void Release()
        {
            loss?.Release();
            if (layers != null)
            {
                for (int i = 0; i < layers.Count; i++)
                {
                    layers[i].Release();
                }
            }
            swapBuffer?.Release();
            clipGradNorm?.Release();
        }

    }

}
