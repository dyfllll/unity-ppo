using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace UnityNN
{

    public class Dropout : Layer
    {

        public static ComputeShader _compute;
        public static ComputeShader compute
        {
            get
            {
                if (_compute == null)
                {
                    _compute = Resources.Load<ComputeShader>("Compute/Dropout");
                }
                return _compute;
            }
        }

        public ComputeBuffer maskBuffer;

        public float dropoutRatio;


        public static int _kernelForwardPredict = -1;
        public static int kernelForwardPredict
        {
            get
            {
                if (_kernelForwardPredict == -1) _kernelForwardPredict = compute.FindKernel("Forward");
                return _kernelForwardPredict;
            }
        }

        public static int _kernelBackward = -1;
        public static int kernelBackward
        {
            get
            {
                if (_kernelBackward == -1) _kernelBackward = compute.FindKernel("Backward");
                return _kernelBackward;
            }
        }

        public static int _kernelForwardTrain = -1;
        public static int kernelForwardTrain
        {
            get
            {
                if (_kernelForwardTrain == -1) _kernelForwardTrain = compute.FindKernel("ForwardTrain");
                return _kernelForwardTrain;
            }
        }


        public int kernelForward = -1;

        public Dropout(float dropoutRatio = 0.1f) : base(LayerType.Dropout, -1)
        {
            this.dropoutRatio = dropoutRatio;
        }


        public override void Init(NetArgs init, Layer prev, Dataset dataset, LayerSwapBuffer swapBuffer)
        {
            base.Init(init, prev, dataset, swapBuffer);

            this.nodeCount = prev.nodeCount;

            dataBuffer = new ComputeBuffer(dataset.batchSize * nodeCount, sizeof(float));

            if (init.netType == NeuralNet.Type.Train)
            {
                maskBuffer = new ComputeBuffer(dataset.batchSize * nodeCount, sizeof(float));
            }

            kernelForward = init.netType == NeuralNet.Type.Train ? kernelForwardTrain : kernelForwardPredict;
        }
        public override void Forward()
        {
            compute.SetInt(ShaderID.batch, dataset.batchSize);
            compute.SetInt(ShaderID.inputCount, nodeCount);
            compute.SetFloat(ShaderID.time, Time.time);
            compute.SetFloat(ShaderID.dropoutRatio, dropoutRatio);

            int kernel = kernelForward;

            compute.SetBuffer(kernel, ShaderID.inputBuffer, prevNodeBuffer);
            compute.SetBuffer(kernel, ShaderID.outputBuffer, dataBuffer);

            if (maskBuffer != null) compute.SetBuffer(kernel, ShaderID.maskBuffer, maskBuffer);

            int groupX = Mathf.CeilToInt(nodeCount / (THREAD_GROUP_SIZE_X * 1.0f));
            int groupY = Mathf.CeilToInt(dataset.batchSize / (THREAD_GROUP_SIZE_Y * 1.0f));


            compute.Dispatch(kernel, groupX, groupY, 1);
        }

        public override void Backward()
        {
            compute.SetInt(ShaderID.batch, dataset.batchSize);
            compute.SetInt(ShaderID.inputCount, nodeCount);

            int kernel = kernelBackward;

            compute.SetBuffer(kernel, ShaderID.maskBuffer, maskBuffer);
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
            maskBuffer?.Release();
        }

        private class ShaderID
        {
            public static readonly int inputBuffer = Shader.PropertyToID("inputBuffer");
            public static readonly int outputBuffer = Shader.PropertyToID("outputBuffer");
            public static readonly int dInputBuffer = Shader.PropertyToID("dInputBuffer");
            public static readonly int dOutputBuffer = Shader.PropertyToID("dOutputBuffer");

            public static readonly int dMaskBuffer = Shader.PropertyToID("dMaskBuffer");
            public static readonly int maskBuffer = Shader.PropertyToID("maskBuffer");

            public static readonly int inputCount = Shader.PropertyToID("inputCount");
            public static readonly int batch = Shader.PropertyToID("batch");
            public static readonly int time = Shader.PropertyToID("time");
            public static readonly int dropoutRatio = Shader.PropertyToID("dropoutRatio");
        }
    }

}
