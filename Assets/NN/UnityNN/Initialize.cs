using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace UnityNN
{

    public class Initialize
    {
        public enum Type
        {
            Random, Xavier, He, None
        }

        public static ComputeShader _compute;
        public static ComputeShader compute
        {
            get
            {
                if (_compute == null)
                {
                    _compute = Resources.Load<ComputeShader>("Compute/InitFunc");
                }
                return _compute;
            }
        }

        public static int _kernelWeightsRandom = -1;

        public static int kernelWeightsRandom
        {
            get
            {
                if (_kernelWeightsRandom == -1) _kernelWeightsRandom = compute.FindKernel("InitWeightsRandom");
                return _kernelWeightsRandom;
            }
        }

        public static int _kernelWeightsXavier = -1;

        public static int kernelWeightsXavier
        {
            get
            {
                if (_kernelWeightsXavier == -1) _kernelWeightsXavier = compute.FindKernel("InitWeightsXavier");
                return _kernelWeightsXavier;
            }
        }

        public static int _kernelWeightsHe = -1;

        public static int kernelWeightsHe
        {
            get
            {
                if (_kernelWeightsHe == -1) _kernelWeightsHe = compute.FindKernel("InitWeightsHe");
                return _kernelWeightsHe;
            }
        }


        public static int _kernelZero = -1;

        public static int kernelZero
        {
            get
            {
                if (_kernelZero == -1) _kernelZero = compute.FindKernel("InitZero");
                return _kernelZero;
            }
        }

   

        private static int GetWeightkernel(Type type)
        {
            switch (type)
            {
                case Type.Random:
                    return kernelWeightsRandom;
                case Type.Xavier:
                    return kernelWeightsXavier;
                case Type.He:
                    return kernelWeightsHe;
                default:
                    return -1;
            }
        }

        public static void InitWeight(Type type, ComputeBuffer buffer, int count)
        {
            int kernel = GetWeightkernel(type);
            if (kernel == -1) return;

            float scale = 1.0f / Mathf.Sqrt(1.0f * count);
            float stdDev = Mathf.Sqrt(2.0f / count);

            compute.SetInt(ShaderID.weightCount, count);
            compute.SetFloat(ShaderID.weightScale, scale);
            compute.SetFloat(ShaderID.weightStdDev, stdDev);
            compute.SetBuffer(kernel, ShaderID.weightBuffer, buffer);

            int groupsX = Mathf.CeilToInt(count / (Layer.THREAD_GROUP_SIZE * 1.0f));

            compute.Dispatch(kernel, groupsX, 1, 1);
        }


        public static void InitZero(ComputeBuffer buffer, int count)
        {
            int kernel = kernelZero;

            compute.SetInt(ShaderID.bufferCount, count);
            compute.SetBuffer(kernel, ShaderID.targetBuffer, buffer);

            int groupsX = Mathf.CeilToInt(count / (Layer.THREAD_GROUP_SIZE * 1.0f));

            compute.Dispatch(kernel, groupsX, 1, 1);
        }

    


        private class ShaderID
        {
            public static readonly int weightCount = Shader.PropertyToID("weightCount");
            public static readonly int weightScale = Shader.PropertyToID("weightScale");
            public static readonly int weightStdDev = Shader.PropertyToID("weightStdDev");
            public static readonly int bufferCount = Shader.PropertyToID("bufferCount");

            public static readonly int weightBuffer = Shader.PropertyToID("weightBuffer");
            public static readonly int targetBuffer = Shader.PropertyToID("targetBuffer");

 
        }

    }

}
