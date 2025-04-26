using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityNN;
using Cysharp.Threading.Tasks;
using System.IO;
using Unity.Mathematics;

//代码主要来自
//https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py
namespace UnityRL.CartPole
{

    [Serializable]
    public class Args
    {
        public int seed = 1;

        public int num_steps = 128;
        public int minibatch_size = 32;
        public int update_epochs = 4;

        public float learning_rate = 2.5e-4f;
        public float learning_rate_now = 2.5e-4f;

        //Toggle learning rate annealing for policy and value networks
        public float gamma = 0.99f;
        //the discount factor gamma
        public float gae_lambda = 0.95f;


        public bool norm_adv = true;
        public bool clip_vloss = true;
        public float clip_coef = 0.2f;
        public float ent_coef = 0.01f;
        public float vf_coef = 0.5f;

        public bool normalizeState = false;
        public bool scaleReward = false;
        public bool learningRateDecay = false;
    }




    [Serializable]
    public class CartPoleAgent
    {
        public NeuralNet actor;
        public NeuralNet critic;
        public NetArgs netArgs;

        public string weightBaisName = "";

        public Dataset actorDataset;
        public Dataset criticDataset;

        public CartPoleEnv env;

        //状态空间与动作空间
        public int observationSpaceCount = 4;
        public int actionSpaceCount = 2;

        public int cacheCount;

        private float[] dones;
        private float[] rewards;
        private float[] obs;
        private float[] logprobs;
        private int[] actions;
        private float[] values;

        private float[] advantages;//优势函数
        private float[] returns;


        public Args args;

        private float[] next_obs;
        private float next_done;

        public int trainCount = 0;
        public int steps = 0;
        public int iterationCount = 0;
        public int maxIterationCount = 4000;

        public int minKeepCount = 10000;
        public int maxKeepCount = -1;
        public RunningMeanStd avgKeepCount = new RunningMeanStd(1);
        public int keepCount = 0;

        //保存weight和bias到文件
        public bool saveWeightAndBias = false;

        //快速训练,就是把update多调用几遍,fastTrainRatio是调用的次数
        public bool fastTrain = false;
        public int fastTrainRatio = 1;


        public Normalization obsNormalize;
        public RewardScaling rewardScaling;


        public bool printLoss = true;
        public bool printKeepCount = true;
        public bool printActorProb = true;


        public int ringBufferCapacity => args.num_steps;

        public void Init(CartPoleEnv cartPoleEnv)
        {
            CommonUtils.InitRandom(args.seed);

            env = cartPoleEnv;

            InitActor();
            InitCritic();

            obsNormalize = new Normalization(observationSpaceCount);
            rewardScaling = new RewardScaling();

            //if (args.normalizeState) next_obs = obsNormalize.Compute(env.Reset());
            //if (args.normalizeState) next_obs = obsNormalize.Compute(env.Reset());

            next_obs = env.Reset();
            next_done = 0;
            if (args.normalizeState) next_obs = obsNormalize.Compute(next_obs);


            dones = new float[args.num_steps];
            rewards = new float[args.num_steps];
            obs = new float[observationSpaceCount * args.num_steps];
            logprobs = new float[args.num_steps];
            actions = new int[args.num_steps];
            values = new float[args.num_steps];

            advantages = new float[args.num_steps];
            returns = new float[args.num_steps];

            if (netArgs.netType == NeuralNet.Type.Predict || netArgs.initType == Initialize.Type.None)
            {
                {
                    string filePath = Application.dataPath + $"{CommonUtils.CartPoleAgentDataPath}/Actor{weightBaisName}.bin";
                    using FileStream fs = new FileStream(filePath, FileMode.Open);
                    actor.LoadWeightAndBias(fs);
                }
                {
                    string filePath = Application.dataPath + $"{CommonUtils.CartPoleAgentDataPath}/Critic{weightBaisName}.bin";
                    using FileStream fs = new FileStream(filePath, FileMode.Open);
                    critic.LoadWeightAndBias(fs);
                }
            }

        }

        private void InitActor()
        {
            int maxBatch = args.minibatch_size;
            actorDataset = new Dataset();
            actorDataset.batchSize = maxBatch;
            actorDataset.inputCount = observationSpaceCount;
            actorDataset.outputCount = actionSpaceCount;
            actorDataset.inputBuffer = new ComputeBuffer(observationSpaceCount * maxBatch, sizeof(float));

            List<Layer> layers = new List<Layer>()
            {
                new Linear(64),
                new ActivationFunc(ActivationFunc.Type.Tanh),
                new Linear(64),
                new ActivationFunc(ActivationFunc.Type.Tanh),
                new Linear(actionSpaceCount)
            };

            actor = new NeuralNet(netArgs, layers, actorDataset);
            actor.Init();
        }

        private void InitCritic()
        {
            int maxBatch = args.minibatch_size;
            criticDataset = new Dataset();
            criticDataset.batchSize = maxBatch;
            criticDataset.inputCount = observationSpaceCount;
            criticDataset.outputCount = 1;
            criticDataset.inputBuffer = new ComputeBuffer(observationSpaceCount * maxBatch, sizeof(float));

            List<Layer> layers = new List<Layer>()
            {
                new Linear(64),
                new ActivationFunc(ActivationFunc.Type.Tanh),
                new Linear(64),
                new ActivationFunc(ActivationFunc.Type.Tanh),
                new Linear(1)
            };

            critic = new NeuralNet(netArgs, layers, criticDataset);
            critic.Init();
        }


        public void UpdateFrame()
        {
            UpdateRender();
            UpdateTrain();
        }

        public void UpdateFast()
        {
            for (int i = 0; i < args.num_steps; i++)
            {
                UpdateRender();
            }
            UpdateTrain();
        }

        public void Update()
        {
            if (fastTrain)
            {
                for (int i = 0; i < fastTrainRatio; i++)
                {
                    UpdateFast();
                }
            }
            else
            {
                UpdateFrame();
            }



            if (saveWeightAndBias)
            {
                string timeStr = System.DateTime.Now.ToString("yyyy-MM-dd-HH-mm-ss");
                {
                    string filePath = Application.dataPath + $"{CommonUtils.CartPoleAgentDataPath}/Actor_{timeStr}.bin";
                    using FileStream fs = new FileStream(filePath, FileMode.OpenOrCreate);
                    actor.SaveWeightAndBias(fs);
                }

                {
                    string filePath = Application.dataPath + $"{CommonUtils.CartPoleAgentDataPath}/Critic_{timeStr}.bin";
                    using FileStream fs = new FileStream(filePath, FileMode.OpenOrCreate);
                    critic.SaveWeightAndBias(fs);
                }

                saveWeightAndBias = false;
            }
        }


        private void SetObsArray(float[] state, int index)
        {
            for (int i = 0; i < observationSpaceCount; i++)
            {
                this.obs[observationSpaceCount * index + i] = state[i];
            }
        }

        private void CopyObsArray(float[] src, int srcIndex, float[] dst, int dstIndex)
        {
            for (int i = 0; i < observationSpaceCount; i++)
            {
                dst[dstIndex * observationSpaceCount + i] = this.obs[observationSpaceCount * srcIndex + i];
            }
        }


        /// <summary>
        /// 每帧产生一条数据
        /// </summary>
        public void UpdateRender()
        {
            if (cacheCount >= args.num_steps)
                return;

            SetObsArray(next_obs, cacheCount);
            this.dones[cacheCount] = next_done;

            float[] logits = ForwardActor(next_obs, 1);
            float[] values = ForwardCritic(next_obs, 1);

            Categorical categorical = new Categorical(logits, null, 1, false);
            int[] actionArray = categorical.Sample();
            float[] logProbArray = categorical.LogProb(actionArray, null);

            if (printActorProb)
            {
                Debug.Log("obs: " + next_obs[0] + " " + next_obs[1] + " " + next_obs[2] + " " + next_obs[3]);
                Debug.Log("x: " + logits[0] + " " + logits[1]);
                Debug.Log("logits: " + categorical.logits[0] + " " + categorical.logits[1]);
                Debug.Log("probs: " + categorical.probs[0] + " " + categorical.probs[1]);
            }

            int action = actionArray[0];
            float logProb = logProbArray[0];
            float value = values[0];

            var next = env.steps_beyond_terminated == -1 ? env.Step(action) : env.ResetEnv();
            steps++;
            if (args.normalizeState) next.state = obsNormalize.Compute(next.state);
            if (args.scaleReward) next.reward = rewardScaling.Compute(next.reward);

            bool isDone = next.terminated || next.truncated;


            this.actions[cacheCount] = action;
            this.logprobs[cacheCount] = logProb;
            this.values[cacheCount] = value;
            this.rewards[cacheCount] = next.reward;

            next_obs = next.state;
            next_done = isDone ? 1 : 0;

            keepCount++;
            if (isDone)
            {
                if (netArgs.netType == NeuralNet.Type.Predict && printKeepCount)
                    Debug.Log(keepCount);
                minKeepCount = Mathf.Min(minKeepCount, keepCount);
                maxKeepCount = Mathf.Max(maxKeepCount, keepCount);
                avgKeepCount.Update(keepCount);
                keepCount = 0;

            }
            cacheCount++;
        }


        private void UpdateLearningRate()
        {
            if (args.learningRateDecay)
            {
                float frac = 1.0f - (Mathf.Min(iterationCount, maxIterationCount) - 1.0f) / maxIterationCount;
                float lrnow = frac * args.learning_rate;
                args.learning_rate_now = lrnow;
            }
            else
            {
                args.learning_rate_now = args.learning_rate;
            }
        }


        public void UpdateTrain()
        {
            if (netArgs.netType == NeuralNet.Type.Predict)
            {
                cacheCount = 0;
                return;
            }

            if (cacheCount < args.num_steps)
                return;

            minKeepCount = Mathf.Min(minKeepCount, keepCount);
            maxKeepCount = Mathf.Max(maxKeepCount, keepCount);
            avgKeepCount.Update(keepCount);
            UpdateLearningRate();


            //GAE
            float lastgaelam = 0;
            float[] next_value = ForwardCritic(next_obs, 1);
            for (int t = args.num_steps - 1; t >= 0; t--)
            {
                float nextnonterminal, nextvalues;
                if (t == args.num_steps - 1)
                {
                    nextnonterminal = 1.0f - next_done;
                    nextvalues = next_value[0];
                }
                else
                {
                    nextnonterminal = 1.0f - dones[t + 1];
                    nextvalues = values[t + 1];
                }

                float delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t];
                advantages[t] = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam;
                returns[t] = advantages[t] + values[t];
                lastgaelam = advantages[t];
            }

            for (int epoch = 0; epoch < args.update_epochs; epoch++)
            {
                int[] batchDataMap = GetRandomIndex(args.num_steps);

                Optimizer.UpdateParam(iterationCount + 1, args.learning_rate_now, minDelta: 1e-5f);

                int batchCount = args.num_steps / args.minibatch_size;
                for (int batchIndex = 0; batchIndex < batchCount; batchIndex++)
                {
                    int dataStart = batchIndex * args.minibatch_size;

                    float[] obs = new float[args.minibatch_size * observationSpaceCount];
                    for (int b = 0; b < args.minibatch_size; b++)
                    {
                        CopyObsArray(this.obs, batchDataMap[dataStart + b], obs, b);
                    }

                    float[] actor = ForwardActor(obs, args.minibatch_size);
                    float[] critic = ForwardCritic(obs, args.minibatch_size);


                    PPOLoss lossCompute = new PPOLoss();
                    lossCompute.b_logprobs = logprobs;
                    lossCompute.b_actions = actions;
                    lossCompute.b_values = values;
                    lossCompute.b_advantages = advantages;
                    lossCompute.b_returns = returns;
                    lossCompute.logits = actor;
                    lossCompute.newvalue = critic;
                    lossCompute.batch = args.minibatch_size;
                    lossCompute.args = args;
                    lossCompute.b_inds = batchDataMap;
                    lossCompute.b_offset = dataStart;
                    (float loss, float[] dActor, float[] dCritic) = lossCompute.Compute();


                    if (printLoss)
                        Debug.Log($"iterationCount:{iterationCount} trainCount:{trainCount} epoch:{epoch + 1} dataStart:{dataStart} loss:{loss}");

                    BackwardActor(dActor, args.minibatch_size);
                    BackwardCritic(dCritic, args.minibatch_size);

                    trainCount++;
                }

            }
            cacheCount -= args.num_steps;

            iterationCount++;

            if (printKeepCount)
                Debug.Log($"minKeepCount:{minKeepCount} maxKeepCount:{maxKeepCount} avgKeepCount:{avgKeepCount.mean[0]}");
            minKeepCount = 10000;
            maxKeepCount = -1;
            avgKeepCount.Reset();

        }

        private int[] GetRandomIndex(int count)
        {
            int[] result = new int[count];
            for (int i = 0; i < count; i++)
            {
                result[i] = i;
            }
            for (int i = count - 1; i >= 0; i--)
            {
                int ti = CommonUtils.random.NextInt(0, i);
                int temp = result[i];
                result[i] = result[ti];
                result[ti] = temp;
            }
            return result;
        }


        private float[] ForwardCritic(float[] obs, int batch)
        {
            float[] output = new float[batch];
            critic.dataset.batchSize = batch;
            critic.dataset.inputBuffer.SetData(obs);
            critic.Forward();
            critic.layers[^1].dataBuffer.GetData(output);
            return output;
        }

        private float[] ForwardActor(float[] obs, int batch)
        {
            float[] output = new float[batch * actionSpaceCount];
            actor.dataset.batchSize = batch;
            actor.dataset.inputBuffer.SetData(obs);
            actor.Forward();
            actor.layers[^1].dataBuffer.GetData(output);
            return output;
        }



        private void BackwardActor(float[] loss_actor, int batch)
        {
            actor.swapBuffer.dOutputBuffer.SetData(loss_actor);
            actor.dataset.batchSize = batch;
            actor.Backward();
        }


        private void BackwardCritic(float[] loss_d_critic, int batch)
        {
            critic.swapBuffer.dOutputBuffer.SetData(loss_d_critic);
            critic.dataset.batchSize = batch;
            critic.Backward();
        }


        public class RewardScaling
        {
            public float gamma;
            public float[] R;
            public RunningMeanStd running_ms;

            public RewardScaling(float gamma = 0.99f)
            {
                this.running_ms = new RunningMeanStd(1);
                this.gamma = gamma;
                this.R = new float[] { 0 };
            }

            public float Compute(float x)
            {
                this.R[0] = this.gamma * this.R[0] + x;
                this.running_ms.Update(R);
                x = x / (this.running_ms.std[0] + 1e-8f);
                return x;
            }

        }




        public class RunningMeanStd
        {
            public int n;
            public float[] mean;
            private float[] s;
            public float[] std;
            public int dim;
            public float[] temp;

            public RunningMeanStd(int dim)
            {
                this.dim = dim;
                this.mean = new float[dim];
                this.s = new float[dim];
                this.std = new float[dim];
                this.n = 0;
            }


            public void Update(float[] v)
            {
                n++;
                if (n == 1)
                {
                    for (int i = 0; i < dim; i++)
                    {
                        mean[i] = v[i];
                        std[i] = v[i];//避免除0情况
                        s[i] = 0;
                    }
                }
                else
                {
                    for (int i = 0; i < dim; i++)
                    {
                        float oldMean = mean[i];
                        mean[i] = oldMean + (v[i] - oldMean) / n;
                        s[i] = s[i] + (v[i] - oldMean) * (v[i] - mean[i]);
                        std[i] = Mathf.Sqrt(s[i] / n);
                    }

                }
            }

            public void Update(float v)
            {
                if (dim != 1) throw new Exception("dim != 1");
                if (temp == null) temp = new float[1];
                temp[0] = v;
                Update(temp);
            }



            public void Reset()
            {
                for (int i = 0; i < dim; i++)
                {
                    mean[i] = 0;
                    std[i] = 0;
                    s[i] = 0;
                }
                n = 0;
            }

        }
        public class Normalization
        {
            public RunningMeanStd running_ms;

            public Normalization(int dim)
            {
                running_ms = new RunningMeanStd(dim);
            }

            public float[] Compute(float[] v)
            {
                running_ms.Update(v);

                for (int i = 0; i < running_ms.dim; i++)
                {
                    v[i] = (v[i] - running_ms.mean[i]) / (running_ms.std[i] + 1e-8f);
                }

                return v;
            }

        }



        public void Release()
        {
            actorDataset?.Release();
            criticDataset?.Release();
            actor?.Release();
            critic?.Release();
        }


    }
}