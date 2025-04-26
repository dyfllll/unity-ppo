using System.Collections;
using System.Collections.Generic;
using UnityEngine;
namespace UnityNN
{
    public class LossFunc : Layer
    {
        public enum Type
        {
            None = -1, MSE, CrossEntropy
        }

        public Type type;

        public static ComputeShader _compute;
        public static ComputeShader compute
        {
            get
            {
                if (_compute == null)
                {
                    _compute = Resources.Load<ComputeShader>("Compute/LossFunc");
                }
                return _compute;
            }
        }

        public Dictionary<string, int> kernelMap = new Dictionary<string, int>();



        public LossFunc(Type type) : base(LayerType.Loss, -1)
        {
            this.type = type;
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

        public override void Init(NetArgs init, Layer prev, Dataset dataset, LayerSwapBuffer swapBuffer)
        {
            // base.Init(init, prev, dataset, swapBuffer);
            this.prev = prev;
            this.dataset = dataset;
            this.swapBuffer = swapBuffer;
            this.nodeCount = prev.nextNodeCount;

            dataBuffer = new ComputeBuffer(dataset.batchSize, sizeof(float));
        }

        public override void Forward()
        {
            compute.SetInt(ShaderID.batch, dataset.batchSize);
            compute.SetInt(ShaderID.inputCount, nodeCount);

            int kernel = GetKernel(type, Direction.Forward);

            compute.SetBuffer(kernel, ShaderID.targetBuffer, dataset.outputBuffer);
            compute.SetBuffer(kernel, ShaderID.outputBuffer, prev.dataBuffer);
            compute.SetBuffer(kernel, ShaderID.lossBuffer, dataBuffer);

            int groupCount = Mathf.CeilToInt(dataset.batchSize / (THREAD_GROUP_SIZE * 1.0f));

            compute.Dispatch(kernel, 1, groupCount, 1);

        }
        public override void Backward()
        {
            compute.SetInt(ShaderID.batch, dataset.batchSize);
            compute.SetInt(ShaderID.inputCount, nodeCount);

            int kernel = GetKernel(type, Direction.Backward);

            compute.SetBuffer(kernel, ShaderID.targetBuffer, dataset.outputBuffer);
            compute.SetBuffer(kernel, ShaderID.outputBuffer, prev.dataBuffer);
            compute.SetBuffer(kernel, ShaderID.dInputBuffer, swapBuffer.dInputBuffer);

            int backwardGroupX = Mathf.CeilToInt(nodeCount / (THREAD_GROUP_SIZE_X * 1.0f));
            int backwardGroupY = Mathf.CeilToInt(dataset.batchSize / (THREAD_GROUP_SIZE_Y * 1.0f));

            compute.Dispatch(kernel, backwardGroupX, backwardGroupY, 1);

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
            public static readonly int targetBuffer = Shader.PropertyToID("targetBuffer");
            public static readonly int lossBuffer = Shader.PropertyToID("lossBuffer");
            public static readonly int outputBuffer = Shader.PropertyToID("outputBuffer");
            public static readonly int dInputBuffer = Shader.PropertyToID("dInputBuffer");

            public static readonly int inputCount = Shader.PropertyToID("inputCount");
            public static readonly int batch = Shader.PropertyToID("batch");
        }

    }
}
