using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace UnityNN.Data
{
    public class FashionMnist
    {
        public static ComputeShader _compute;
        public static ComputeShader compute
        {
            get
            {
                if (_compute == null)
                {
                    _compute = Resources.Load<ComputeShader>("Compute/FashionMnist");
                }
                return _compute;
            }
        }

        public static int _kernelTransformRawToInput = -1;

        public static int kernelTransformRawToInput
        {
            get
            {
                if (_kernelTransformRawToInput == -1) _kernelTransformRawToInput = compute.FindKernel("TransformRawToInput");
                return _kernelTransformRawToInput;
            }
        }

        public static int _kernelTransformRawToOutput = -1;

        public static int kernelTransformRawToOutput
        {
            get
            {
                if (_kernelTransformRawToOutput == -1) _kernelTransformRawToOutput = compute.FindKernel("TransformRawToOutput");
                return _kernelTransformRawToOutput;
            }
        }



        public static void TransformRawToInput(ComputeBuffer rawBuffer, ComputeBuffer targetBuffer, int batchIndex, int inputCount, int batchSize, int rawLineSize)
        {
            int kernel = kernelTransformRawToInput;

            compute.SetInt(ShaderID.inputCount, inputCount);
            compute.SetInt(ShaderID.batch, batchSize);


            int rawDataStart = (batchIndex * batchSize) * rawLineSize; //raw起始位置
            int rawEveryOffset = 1; //raw每行数据开始第一个是类型


            compute.SetInt(ShaderID.rawLineSize, rawLineSize);
            compute.SetInt(ShaderID.rawOffset, rawDataStart + rawEveryOffset);

            compute.SetBuffer(kernel, ShaderID.rawBuffer, rawBuffer);
            compute.SetBuffer(kernel, ShaderID.targetBuffer, targetBuffer);

            int groupsX = Mathf.CeilToInt(inputCount / (Layer.THREAD_GROUP_SIZE_X * 1.0f));
            int groupsY = Mathf.CeilToInt(batchSize / (Layer.THREAD_GROUP_SIZE_Y * 1.0f));

            compute.Dispatch(kernel, groupsX, groupsY, 1);
        }

        public static void TransformRawToOutput(ComputeBuffer rawBuffer, ComputeBuffer targetBuffer, int batchIndex, int inputCount, int batchSize, int rawLineSize)
        {
            int kernel = kernelTransformRawToOutput;

            compute.SetInt(ShaderID.inputCount, inputCount);
            compute.SetInt(ShaderID.batch, batchSize);


            int rawDataStart = (batchIndex * batchSize) * rawLineSize; //raw起始位置
            int rawEveryOffset = 0; //raw每行数据开始第一个是类型


            compute.SetInt(ShaderID.rawLineSize, rawLineSize);
            compute.SetInt(ShaderID.rawOffset, rawDataStart + rawEveryOffset);

            compute.SetBuffer(kernel, ShaderID.rawBuffer, rawBuffer);
            compute.SetBuffer(kernel, ShaderID.targetBuffer, targetBuffer);

            int groupsX = 1;
            int groupsY = Mathf.CeilToInt(batchSize / (Layer.THREAD_GROUP_SIZE_Y * 1.0f));

            compute.Dispatch(kernel, groupsX, groupsY, 1);
        }


        private class ShaderID
        {

            public static readonly int inputCount = Shader.PropertyToID("inputCount");
            public static readonly int batch = Shader.PropertyToID("batch");
            public static readonly int targetBuffer = Shader.PropertyToID("targetBuffer");

            public static readonly int rawBuffer = Shader.PropertyToID("rawBuffer");
            public static readonly int rawLineSize = Shader.PropertyToID("rawLineSize");
            public static readonly int rawOffset = Shader.PropertyToID("rawOffset");
        }


    }

}

