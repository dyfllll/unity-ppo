using System.Collections;
using System.Collections.Generic;
using UnityEngine;
namespace UnityNN
{
    public class ActivationFunc : Layer
    {
        public enum Type
        {
            ReLU, Softmax, Sigmoid, Tanh, LeakyReLU
        }

        public Type type;

        public static ComputeShader _compute;
        public static ComputeShader compute
        {
            get
            {
                if (_compute == null)
                {
                    _compute = Resources.Load<ComputeShader>("Compute/ActivationFunc");
                }
                return _compute;
            }
        }

        public Dictionary<string, int> kernelMap = new Dictionary<string, int>();



        public ActivationFunc(Type type) : base(Layer.LayerType.Activation, -1)
        {
            this.type = type;
        }

        public override void Init(NetArgs init, Layer prev, Dataset dataset, LayerSwapBuffer swapBuffer)
        {
            base.Init(init, prev, dataset, swapBuffer);

            this.nodeCount = prev.nodeCount;

            dataBuffer = new ComputeBuffer(dataset.batchSize * nodeCount, sizeof(float));
        }


        private int GetKernel(Type type, Direction direction)
        {
            string key = $"{direction}{type}";
            if (!kernelMap.TryGetValue(key, out int kernel))
            {
                kernel = compute.FindKernel(key);
                kernelMap.Add(key, kernel);
            }
            return kernel;
        }


        public override void Forward()
        {
            compute.SetInt(ShaderID.batch, dataset.batchSize);
            compute.SetInt(ShaderID.inputCount, nodeCount);

            int kernel = GetKernel(type, Direction.Forward);

            compute.SetBuffer(kernel, ShaderID.inputBuffer, prevNodeBuffer);
            compute.SetBuffer(kernel, ShaderID.outputBuffer, dataBuffer);

            int groupX = Mathf.CeilToInt(nodeCount / (THREAD_GROUP_SIZE_X * 1.0f));
            int groupY = Mathf.CeilToInt(dataset.batchSize / (THREAD_GROUP_SIZE_Y * 1.0f));

            compute.Dispatch(kernel, groupX, groupY, 1);
        }

        public override void Backward()
        {
            compute.SetInt(ShaderID.batch, dataset.batchSize);
            compute.SetInt(ShaderID.inputCount, nodeCount);

            int kernel = GetKernel(type, Direction.Backward);

            compute.SetBuffer(kernel, ShaderID.inputBuffer, prevNodeBuffer);
            compute.SetBuffer(kernel, ShaderID.outputBuffer, dataBuffer);
            compute.SetBuffer(kernel, ShaderID.dInputBuffer, swapBuffer.dInputBuffer);
            compute.SetBuffer(kernel, ShaderID.dOutputBuffer, swapBuffer.dOutputBuffer);


            int groupX = Mathf.CeilToInt(nodeCount / (THREAD_GROUP_SIZE_X * 1.0f));
            int groupY = Mathf.CeilToInt(dataset.batchSize / (THREAD_GROUP_SIZE_Y * 1.0f));

            compute.Dispatch(kernel, groupX, groupY, 1);

            swapBuffer.Swap();
        }

        public override void Release()
        {
            base.Release();
            dataBuffer?.Release();
        }

        private class ShaderID
        {
            public static readonly int inputBuffer = Shader.PropertyToID("inputBuffer");
            public static readonly int outputBuffer = Shader.PropertyToID("outputBuffer");
            public static readonly int dInputBuffer = Shader.PropertyToID("dInputBuffer");
            public static readonly int dOutputBuffer = Shader.PropertyToID("dOutputBuffer");

            public static readonly int inputCount = Shader.PropertyToID("inputCount");
            public static readonly int batch = Shader.PropertyToID("batch");
        }
    }
}
