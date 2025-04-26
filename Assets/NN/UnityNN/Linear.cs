using System.Collections;
using System.Collections.Generic;
using UnityEngine;
namespace UnityNN
{


    public class Linear : Layer
    {
        public static ComputeShader _compute;
        public static ComputeShader compute
        {
            get
            {
                if (_compute == null)
                {
                    _compute = Resources.Load<ComputeShader>("Compute/Linear");
                }
                return _compute;
            }
        }

        public static int _kernelForward = -1;
        public static int _kernelBackward = -1;
        public static int _kernelGrads = -1;
        public static int kernelForward
        {
            get
            {
                if (_kernelForward == -1) _kernelForward = compute.FindKernel("Forward");
                return _kernelForward;
            }
        }
        public static int kernelBackward
        {
            get
            {
                if (_kernelBackward == -1) _kernelBackward = compute.FindKernel("Backward");
                return _kernelBackward;
            }
        }
        public static int kernelGrads
        {
            get
            {
                if (_kernelGrads == -1) _kernelGrads = compute.FindKernel("ComputeGrads");
                return _kernelGrads;
            }
        }


        //weight和bias  
        public int weightCount;
        public int biasCount;
        public ComputeBuffer weightBuffer;
        public ComputeBuffer biasBuffer;


        public Optimizer optimizer;

        //梯度裁剪
        public bool clipGrad;
        public float maxGradNorm;
        public ComputeBuffer dWeightBuffer;
        public ComputeBuffer dBiasBuffer;


        public Linear(int nodeCount) : base(LayerType.Linear, nodeCount)
        {
        }


        public override void Init(NetArgs init, Layer prev, Dataset dataset, LayerSwapBuffer swapBuffer)
        {
            base.Init(init, prev, dataset, swapBuffer);

            if (init.netType == NeuralNet.Type.Train)
            {
                optimizer = new Optimizer(init.optimType);
                optimizer.Init(prevNodeCount, nodeCount);
            }

            dataBuffer = new ComputeBuffer(dataset.batchSize * nodeCount, sizeof(float));

            weightCount = prevNodeCount * nodeCount;
            biasCount = nodeCount;
            weightBuffer = new ComputeBuffer(weightCount, sizeof(float));
            biasBuffer = new ComputeBuffer(biasCount, sizeof(float));

            if (init.netType == NeuralNet.Type.Train && init.initType != Initialize.Type.None)
            {
                Initialize.InitWeight(init.initType, weightBuffer, prevNodeCount * nodeCount);
                Initialize.InitZero(biasBuffer, nodeCount);
            }
            clipGrad = init.clipGrad;
            maxGradNorm = init.maxGradNorm;
            if (clipGrad)
            {
                dWeightBuffer = new ComputeBuffer(weightCount, sizeof(float));
                dBiasBuffer = new ComputeBuffer(biasCount, sizeof(float));
            }
        }

        public override void Forward()
        {
            compute.SetInt(ShaderID.batch, dataset.batchSize);
            compute.SetFloat(ShaderID.batchInv, 1.0f / dataset.batchSize);
            compute.SetInt(ShaderID.inputCount, prevNodeCount);
            compute.SetInt(ShaderID.outputCount, nodeCount);

            compute.SetBuffer(kernelForward, ShaderID.inputBuffer, prevNodeBuffer);
            compute.SetBuffer(kernelForward, ShaderID.weightBuffer, weightBuffer);
            compute.SetBuffer(kernelForward, ShaderID.biasBuffer, biasBuffer);
            compute.SetBuffer(kernelForward, ShaderID.outputBuffer, dataBuffer);

            int groupX = Mathf.CeilToInt(nodeCount / (THREAD_GROUP_SIZE_X * 1.0f));
            int groupY = Mathf.CeilToInt(dataset.batchSize / (THREAD_GROUP_SIZE_Y * 1.0f));

            compute.Dispatch(kernelForward, groupX, groupY, 1);
        }


        public override void Backward()
        {
            compute.SetInt(ShaderID.batch, dataset.batchSize);
            compute.SetFloat(ShaderID.batchInv, 1.0f / dataset.batchSize);
            compute.SetInt(ShaderID.inputCount, prevNodeCount);
            compute.SetInt(ShaderID.outputCount, nodeCount);


            int backwardGroupX = Mathf.CeilToInt(prevNodeCount / (THREAD_GROUP_SIZE_X * 1.0f));
            int backwardGroupY = Mathf.CeilToInt(dataset.batchSize / (THREAD_GROUP_SIZE_Y * 1.0f));

            //反向传播到上一层
            compute.SetBuffer(kernelBackward, ShaderID.dOutputBuffer, swapBuffer.dOutputBuffer);
            compute.SetBuffer(kernelBackward, ShaderID.dInputBuffer, swapBuffer.dInputBuffer);
            compute.SetBuffer(kernelBackward, ShaderID.weightBuffer, weightBuffer);
            compute.Dispatch(kernelBackward, backwardGroupX, backwardGroupY, 1);



            int gradsGroupX = Mathf.CeilToInt(prevNodeCount / (THREAD_GROUP_SIZE_X * 1.0f));
            int gradsGroupY = Mathf.CeilToInt(nodeCount / (THREAD_GROUP_SIZE_Y * 1.0f));

            if (clipGrad)
            {
                //更新weight和bias的梯度
                compute.SetBuffer(kernelGrads, ShaderID.dOutputBuffer, swapBuffer.dOutputBuffer);
                compute.SetBuffer(kernelGrads, ShaderID.inputBuffer, prevNodeBuffer);
                compute.SetBuffer(kernelGrads, ShaderID.dWeightBuffer, dWeightBuffer);
                compute.SetBuffer(kernelGrads, ShaderID.dBiasBuffer, dBiasBuffer);
                compute.Dispatch(kernelGrads, gradsGroupX, gradsGroupY, 1);
            }
            else
            {
                //更新weight和bias的梯度
                compute.SetBuffer(kernelGrads, ShaderID.dOutputBuffer, swapBuffer.dOutputBuffer);
                compute.SetBuffer(kernelGrads, ShaderID.inputBuffer, prevNodeBuffer);
                compute.SetBuffer(kernelGrads, ShaderID.dWeightBuffer, swapBuffer.dWeightBuffer);
                compute.SetBuffer(kernelGrads, ShaderID.dBiasBuffer, swapBuffer.dBiasBuffer);
                compute.Dispatch(kernelGrads, gradsGroupX, gradsGroupY, 1);

                //更新weight
                optimizer.Update(weightBuffer, biasBuffer, swapBuffer.dWeightBuffer, swapBuffer.dBiasBuffer);
            }

            //交换输入输出
            swapBuffer.Swap();
        }


        public void UpdateWeightBias()
        {
            optimizer.Update(weightBuffer, biasBuffer, dWeightBuffer, dBiasBuffer);
        }


        public override void Release()
        {
            base.Release();

            dataBuffer?.Release();
            weightBuffer?.Release();
            biasBuffer?.Release();
            optimizer?.Release();
            dWeightBuffer?.Release();
            dBiasBuffer?.Release();
        }

        private class ShaderID
        {
            public static readonly int inputBuffer = Shader.PropertyToID("inputBuffer");
            public static readonly int weightBuffer = Shader.PropertyToID("weightBuffer");
            public static readonly int biasBuffer = Shader.PropertyToID("biasBuffer");
            public static readonly int outputBuffer = Shader.PropertyToID("outputBuffer");

            public static readonly int dOutputBuffer = Shader.PropertyToID("dOutputBuffer");
            public static readonly int dInputBuffer = Shader.PropertyToID("dInputBuffer");
            public static readonly int dWeightBuffer = Shader.PropertyToID("dWeightBuffer");
            public static readonly int dBiasBuffer = Shader.PropertyToID("dBiasBuffer");

            public static readonly int batch = Shader.PropertyToID("batch");
            public static readonly int batchInv = Shader.PropertyToID("batchInv");
            public static readonly int outputCount = Shader.PropertyToID("outputCount");
            public static readonly int inputCount = Shader.PropertyToID("inputCount");
        }
    }


}
