using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace UnityNN
{

    public class Optimizer
    {
        public enum Type
        {
            SGD,
            Momentum,
            Nesterov,
            AdaGrad,
            RMSprop,
            Adam,
        }

        public static ComputeShader _compute;
        public static ComputeShader compute
        {
            get
            {
                if (_compute == null)
                {
                    _compute = Resources.Load<ComputeShader>("Compute/UpdateWFunc");
                }
                return _compute;
            }
        }

        public Type type;

        public bool needMBuffer => type == Type.Adam || type == Type.Momentum || type == Type.Nesterov || type == Type.RMSprop;
        public bool needVBuffer => type == Type.Adam || type == Type.AdaGrad;

        //非SGD时使用
        public ComputeBuffer weightMBuffer;
        public ComputeBuffer weightVBuffer;

        public ComputeBuffer biasMBuffer;
        public ComputeBuffer biasVBuffer;

        public int inputCount;
        public int outputCount;
        public int kernelUpdate;


        public static float learningRate = -1; //学习率
        public static float beta1 = -1;
        public static float beta2 = -1;
        public static float minDelta = -1;
        public static float curEpochs = -1;


        public Optimizer(Type type)
        {
            this.type = type;
            this.kernelUpdate = compute.FindKernel($"Update{this.type.ToString()}");
        }

        public static void UpdateParam(float curEpochs, float learningRate = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float minDelta = 1e-7f)
        {

            bool change = false;
            if (Optimizer.learningRate != learningRate) { compute.SetFloat(ShaderID.learningRate, learningRate); change = true; }
            if (Optimizer.beta1 != beta1) { compute.SetFloat(ShaderID.beta1, beta1); change = true; }
            if (Optimizer.beta2 != beta2) { compute.SetFloat(ShaderID.beta2, beta2); change = true; }
            if (Optimizer.minDelta != minDelta) { compute.SetFloat(ShaderID.minDelta, minDelta); change = true; }
            if (Optimizer.curEpochs != curEpochs) { compute.SetFloat(ShaderID.curEpochs, curEpochs); change = true; }

            if (change)
            {
                float learningRate_t = learningRate * Mathf.Sqrt(1.0f - Mathf.Pow(beta2, curEpochs)) / (1.0f - Mathf.Pow(beta1, curEpochs));
                compute.SetFloat(ShaderID.learningRate_t, learningRate_t);
            }

            Optimizer.learningRate = learningRate;
            Optimizer.beta1 = beta1;
            Optimizer.beta2 = beta2;
            Optimizer.minDelta = minDelta;
            Optimizer.curEpochs = curEpochs;
        }

        public void Init(int inputCount, int outputCount)
        {
            this.inputCount = inputCount;
            this.outputCount = outputCount;

            if (needMBuffer)
            {
                weightMBuffer = new ComputeBuffer(inputCount * outputCount, sizeof(float));
                biasMBuffer = new ComputeBuffer(outputCount, sizeof(float));
                Initialize.InitZero(weightMBuffer, inputCount * outputCount);
                Initialize.InitZero(biasMBuffer, outputCount);
            }

            if (needVBuffer)
            {
                weightVBuffer = new ComputeBuffer(inputCount * outputCount, sizeof(float));
                biasVBuffer = new ComputeBuffer(outputCount, sizeof(float));
                Initialize.InitZero(weightVBuffer, inputCount * outputCount);
                Initialize.InitZero(biasVBuffer, outputCount);
            }
        }



        public void Update(ComputeBuffer weightBuffer, ComputeBuffer biasBuffer, ComputeBuffer dWeightBuffer, ComputeBuffer dBiasBuffer)
        {

            compute.SetInt(ShaderID.inputCount, inputCount);
            compute.SetInt(ShaderID.outputCount, outputCount);

            compute.SetBuffer(kernelUpdate, ShaderID.gradsBuffer, dWeightBuffer);
            if (needMBuffer) compute.SetBuffer(kernelUpdate, ShaderID.gradsMBuffer, weightMBuffer);
            if (needVBuffer) compute.SetBuffer(kernelUpdate, ShaderID.gradsVBuffer, weightVBuffer);
            compute.SetBuffer(kernelUpdate, ShaderID.updatedBuffer, weightBuffer);

            int groupX = Mathf.CeilToInt(inputCount / (Layer.THREAD_GROUP_SIZE_X * 1.0f));
            int groupY = Mathf.CeilToInt(outputCount / (Layer.THREAD_GROUP_SIZE_Y * 1.0f));
            compute.Dispatch(kernelUpdate, groupX, groupY, 1);


            compute.SetBuffer(kernelUpdate, ShaderID.gradsBuffer, dBiasBuffer);
            if (needMBuffer) compute.SetBuffer(kernelUpdate, ShaderID.gradsMBuffer, biasMBuffer);
            if (needVBuffer) compute.SetBuffer(kernelUpdate, ShaderID.gradsVBuffer, biasVBuffer);
            compute.SetBuffer(kernelUpdate, ShaderID.updatedBuffer, biasBuffer);

            groupX = 1;
            groupY = Mathf.CeilToInt(outputCount / (Layer.THREAD_GROUP_SIZE_Y * 1.0f));
            compute.Dispatch(kernelUpdate, groupX, groupY, 1);
        }


        public void Release()
        {
            weightMBuffer?.Release();
            weightVBuffer?.Release();
            biasMBuffer?.Release();
            biasVBuffer?.Release();
        }

        private class ShaderID
        {
            public static readonly int gradsBuffer = Shader.PropertyToID("gradsBuffer");
            public static readonly int gradsMBuffer = Shader.PropertyToID("gradsMBuffer");
            public static readonly int gradsVBuffer = Shader.PropertyToID("gradsVBuffer");
            public static readonly int updatedBuffer = Shader.PropertyToID("updatedBuffer");

            public static readonly int learningRate = Shader.PropertyToID("learningRate");
            public static readonly int learningRate_t = Shader.PropertyToID("learningRate_t");
            public static readonly int beta1 = Shader.PropertyToID("beta1");
            public static readonly int beta2 = Shader.PropertyToID("beta2");
            public static readonly int minDelta = Shader.PropertyToID("minDelta");
            public static readonly int curEpochs = Shader.PropertyToID("curEpochs");

            public static readonly int inputCount = Shader.PropertyToID("inputCount");
            public static readonly int outputCount = Shader.PropertyToID("outputCount");
        }
    }

}
