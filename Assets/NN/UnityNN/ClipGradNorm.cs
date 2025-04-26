using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace UnityNN
{
    public class ClipGradNorm
    {
        private const int THREADS_PER_GROUP = 256;

        public static ComputeShader _compute;
        public static ComputeShader compute
        {
            get
            {
                if (_compute == null)
                {
                    _compute = Resources.Load<ComputeShader>("Compute/ClipGradNorm");
                }
                return _compute;
            }
        }

        public static int _kernelSquareSum = -1;
        public static int _kernelComputeClipCoef = -1;
        public static int _kernelApplyClipCoef = -1;
        public static int kernelSquareSum
        {
            get
            {
                if (_kernelSquareSum == -1) _kernelSquareSum = compute.FindKernel("SquareSum");
                return _kernelSquareSum;
            }
        }
        public static int kernelComputeClipCoef
        {
            get
            {
                if (_kernelComputeClipCoef == -1) _kernelComputeClipCoef = compute.FindKernel("ComputeClipCoef");
                return _kernelComputeClipCoef;
            }
        }
        public static int kernelApplyClipCoef
        {
            get
            {
                if (_kernelApplyClipCoef == -1) _kernelApplyClipCoef = compute.FindKernel("ApplyClipCoef");
                return _kernelApplyClipCoef;
            }
        }


        public ComputeBuffer inputBuffer;
        public ComputeBuffer outputBuffer;
        public ComputeBuffer resultBuffer;


        public List<Linear> layers;
        public List<ComputeBuffer> layerBuffer;




        public ClipGradNorm(NeuralNet net)
        {
            int maxParamCount = 0;
            layers = new List<Linear>();
            layerBuffer = new List<ComputeBuffer>();
            foreach (var item in net.layers)
            {
                if (item is Linear linear)
                {
                    maxParamCount = Mathf.Max(maxParamCount, linear.weightCount, linear.biasCount);
                    layers.Add(linear);
                    layerBuffer.Add(linear.dWeightBuffer);
                    layerBuffer.Add(linear.dBiasBuffer);
                }
            }

            if (layers.Count == 0) return;

            inputBuffer = new ComputeBuffer(maxParamCount, sizeof(float));
            outputBuffer = new ComputeBuffer(maxParamCount, sizeof(float));
            resultBuffer = new ComputeBuffer(layers.Count, sizeof(float));
        }


        public void Clip(float maxNorm)
        {
            for (int i = 0; i < layerBuffer.Count; i++)
            {
                SquareSum(layerBuffer[i], layerBuffer[i].count, resultBuffer, i);
            }
            ComputeClipCoef(resultBuffer, layerBuffer.Count, maxNorm);
            ApplyClipCoef();
        }


        private void ApplyClipCoef()
        {
            int kernel = kernelApplyClipCoef;

            for (int i = 0; i < layerBuffer.Count; i++)
            {
                int maxCount = layerBuffer[i].count;
                compute.SetBuffer(kernel, ShaderID.inputBuffer, resultBuffer);
                compute.SetBuffer(kernel, ShaderID.outputBuffer, layerBuffer[i]);
                compute.SetInt(ShaderID.maxCount, maxCount);

                int groupCount = Mathf.CeilToInt(maxCount / (float)THREADS_PER_GROUP);
                compute.Dispatch(kernel, groupCount, 1, 1);
            }
        }


        private void ComputeClipCoef(ComputeBuffer buffer, int count, float maxNorm)
        {
            int kernel = kernelComputeClipCoef;
            compute.SetInt(ShaderID.maxCount, count);
            compute.SetFloat(ShaderID.maxNorm, maxNorm);
            compute.SetBuffer(kernel, ShaderID.outputBuffer, buffer);
            compute.Dispatch(kernel, 1, 1, 1);
        }


        private void SquareSum(ComputeBuffer buffer, int inputSize, ComputeBuffer resultBuffer, int index)
        {
            int threadCount = THREADS_PER_GROUP * 2;
            int kernel = kernelSquareSum;

            int count = inputSize == 1 ? 1 : Mathf.CeilToInt(Mathf.Log(inputSize) / Mathf.Log(threadCount));

            int outputSize = Mathf.CeilToInt(inputSize / (float)threadCount);

            for (int i = 0; i < count; i++)
            {
                if (i == 0)
                {
                    compute.SetBuffer(kernel, ShaderID.inputBuffer, buffer);
                    compute.EnableKeyword("DISPATCH_BEGIN");
                }
                else
                {
                    compute.SetBuffer(kernel, ShaderID.inputBuffer, inputBuffer);
                    compute.DisableKeyword("DISPATCH_BEGIN");
                }

                if (i == count - 1)
                {
                    compute.SetInt(ShaderID.outputOffset, index);
                    compute.SetBuffer(kernel, ShaderID.outputBuffer, resultBuffer);
                    compute.EnableKeyword("DISPATCH_END");
                }
                else
                {
                    compute.SetInt(ShaderID.outputOffset, 0);
                    compute.SetBuffer(kernel, ShaderID.outputBuffer, outputBuffer);
                    compute.DisableKeyword("DISPATCH_END");
                }

                compute.SetInt(ShaderID.maxCount, inputSize);
                compute.Dispatch(kernel, outputSize, 1, 1);

                inputSize = outputSize;
                outputSize = Mathf.CeilToInt(outputSize / (float)threadCount);

                SwapBuffer();
            }
        }


        private void SwapBuffer()
        {
            ComputeBuffer temp = inputBuffer;
            inputBuffer = outputBuffer;
            outputBuffer = temp;
        }


        public void Release()
        {
            inputBuffer?.Release();
            outputBuffer?.Release();
            resultBuffer?.Release();
        }

        private class ShaderID
        {
            public static readonly int inputBuffer = Shader.PropertyToID("inputBuffer");
            public static readonly int outputBuffer = Shader.PropertyToID("outputBuffer");
            public static readonly int outputOffset = Shader.PropertyToID("outputOffset");

            public static readonly int maxCount = Shader.PropertyToID("maxCount");
            public static readonly int maxNorm = Shader.PropertyToID("maxNorm");
        }
    }


}
