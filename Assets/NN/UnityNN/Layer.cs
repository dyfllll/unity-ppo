using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace UnityNN
{

    public abstract class Layer
    {

        public const int THREAD_GROUP_SIZE_X = 8;
        public const int THREAD_GROUP_SIZE_Y = 8;
        public const int THREAD_GROUP_SIZE = 64;

        public enum LayerType
        {
            Linear,
            Activation,
            Loss,
            Dropout,
        }

        public enum Direction
        {
            Forward,
            Backward
        }

        public Layer prev;
        public Layer next;


        public LayerType type;
        public int nodeCount;
 

        //当前层的输出结果
        public ComputeBuffer dataBuffer;

        //用于交换的buffer
        public LayerSwapBuffer swapBuffer;

        public Dataset dataset;


        public Layer(LayerType type, int nodeCount)
        {
            this.type = type;
            this.nodeCount = nodeCount;
        }


        public virtual void Init(NetArgs init, Layer prev, Dataset dataset, LayerSwapBuffer swapBuffer)
        {
            if (prev != null)
            {
                this.prev = prev;
                this.prev.next = this;
            }
            this.dataset = dataset;
            this.swapBuffer = swapBuffer;
        }


        public int prevNodeCount
        {
            get
            {
                if (prev != null) return prev.nodeCount;
                else return dataset.inputCount;
            }
        }

        public int nextNodeCount
        {
            get
            {
                if (next != null) return next.nodeCount;
                else return dataset.outputCount;
            }
        }

        public ComputeBuffer prevNodeBuffer
        {
            get
            {
                if (prev != null) return prev.dataBuffer;
                else return dataset.inputBuffer;
            }
        }

        public ComputeBuffer nextNodeBuffer
        {
            get
            {
                if (next != null) return next.dataBuffer;
                else return dataset.outputBuffer;
            }
        }


        public abstract void Forward();
        public abstract void Backward();
        public virtual void Release()
        {

        }
    }


    public class LayerSwapBuffer
    {
        public ComputeBuffer dInputBuffer;
        public ComputeBuffer dOutputBuffer;
        public ComputeBuffer dWeightBuffer;
        public ComputeBuffer dBiasBuffer;

        public void Swap()
        {
            ComputeBuffer temp = dInputBuffer;
            dInputBuffer = dOutputBuffer;
            dOutputBuffer = temp;
        }

        public void Release()
        {
            dInputBuffer?.Release();
            dOutputBuffer?.Release();
            dWeightBuffer?.Release();
            dBiasBuffer?.Release();
        }
    }

    public class Dataset
    {
        public int batchSize;
        public int inputCount;
        public int outputCount;
        public ComputeBuffer inputBuffer;
        public ComputeBuffer outputBuffer;

        public void Release()
        {
            inputBuffer?.Release();
            outputBuffer?.Release();
        }
    }

}
