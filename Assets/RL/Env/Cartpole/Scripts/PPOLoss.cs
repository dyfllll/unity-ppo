#define Backward
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
namespace UnityRL.CartPole
{
    public class PPOLoss
    {
        public float[] b_logprobs;
        public int[] b_actions;
        public float[] b_values;

        public float[] b_advantages;
        public float[] b_returns;

        public float[] logits;
        public float[] newvalue;

        public int batch;
        public Args args;

        public int[] b_inds;
        public int b_offset;

        public PPOLoss()
        {
        }


        public T Get<T>(T[] array, int i)
        {
            return array[b_inds[b_offset + i]];
        }


#if Backward
        public (float loss, float[] dActor, float[] dCritic) Compute()
#else
        public float Compute()
#endif
        {
#if Backward
            Categorical categorical = new Categorical(logits, null, batch, true);
#else
            Categorical categorical = new Categorical(logits, null, batch, false);
#endif

            int[] action = new int[batch];
            for (int i = 0; i < batch; i++)
            {
                action[i] = Get(b_actions, i);
            }

#if Backward
            float[] newlogprob_d_actor = new float[logits.Length];
            float[] entropy_d_actor = new float[logits.Length];
            float[] newlogprob = categorical.LogProb(action, newlogprob_d_actor);
            float[] entropy = categorical.Entropy(entropy_d_actor);
#else
            float[] newlogprob = categorical.LogProb(action, null);
            float[] entropy = categorical.Entropy(null);
#endif


            float[] ratio = new float[batch];
            float[] mb_advantages = new float[batch];

#if Backward
            float[] ratio_d_newlogprob = new float[batch];
#endif

            for (int i = 0; i < batch; i++)
            {
                float logratio = newlogprob[i] - Get(b_logprobs, i);
                ratio[i] = Mathf.Exp(logratio);
                mb_advantages[i] = Get(b_advantages, i);

#if Backward
                float d_logratio = 1;
                ratio_d_newlogprob[i] = Mathf.Exp(logratio) * d_logratio;
#endif
            }


            if (args.norm_adv)
            {
                mb_advantages = NormalizeAdvantages(mb_advantages);
            }

#if Backward
            float[] loss_d_ratio = new float[batch];
            float[] loss_d_newvalue = new float[batch];
            float[] loss_d_entropy = new float[batch];
            float loss = ComputeLoss(mb_advantages, ratio, newvalue, entropy, loss_d_ratio, loss_d_newvalue, loss_d_entropy);
#else
            float loss = ComputeLoss(mb_advantages, ratio, newvalue, entropy);
#endif

#if Backward
            float[] loss_d_actor = new float[logits.Length];
            float[] loss_d_critic = loss_d_newvalue;

            for (int b = 0; b < batch; b++)
            {
                float it_loss_d_ratio = loss_d_ratio[b];
                float it_loss_d_entropy = loss_d_entropy[b];

                float loss_d_newlogprob = it_loss_d_ratio * ratio_d_newlogprob[b];

                for (int i = 0; i < categorical.count; i++)
                {
                    int index = b * categorical.count + i;
                    float it_newlogprob_d_actor = newlogprob_d_actor[index];
                    float it_entropy_d_actor = entropy_d_actor[index];

                    float it_loss_d_actor = loss_d_newlogprob * it_newlogprob_d_actor + it_loss_d_entropy * it_entropy_d_actor;

                    loss_d_actor[index] = it_loss_d_actor;
                }
            }

            return (loss, loss_d_actor, loss_d_critic);
#else
            return loss;
#endif


        }


        private float ComputeLoss(float[] mb_advantages, float[] ratio, float[] newvalue, float[] entropy
#if Backward
    , float[] d_ratio, float[] d_newvalue, float[] d_entropy
#endif
    )
        {
            //Policy loss
            float pg_loss_sum = 0;
            for (int i = 0; i < batch; i++)
            {
                float pg_loss1 = -mb_advantages[i] * ratio[i];
                float pg_loss2 = -mb_advantages[i] * Mathf.Clamp(ratio[i], 1 - args.clip_coef, 1 + args.clip_coef);
                pg_loss_sum += Mathf.Max(pg_loss1, pg_loss2);

#if Backward

                float pg_loss1_dra = -mb_advantages[i];
                float pg_loss2_dra = -mb_advantages[i] * (ratio[i] > (1 + args.clip_coef) || ratio[i] < (1 - args.clip_coef) ? 0 : 1);
                float max_dra = pg_loss1 > pg_loss2 ? pg_loss1_dra : pg_loss2_dra;
                d_ratio[i] = max_dra / batch;
#endif
            }
            float pg_loss = pg_loss_sum / batch;


            //Value loss
            float v_loss;
            if (args.clip_vloss)
            {
                float sum = 0;
                for (int i = 0; i < batch; i++)
                {
                    float i_returns = Get(b_returns, i);
                    float i_values = Get(b_values, i);


                    float v_loss_unclipped = Mathf.Pow(newvalue[i] - i_returns, 2);
                    float v_clipped = i_values + Mathf.Clamp(newvalue[i] - i_values, -args.clip_coef, args.clip_coef);
                    float v_loss_clipped = Mathf.Pow(v_clipped - i_returns, 2);
                    sum += Mathf.Max(v_loss_unclipped, v_loss_clipped);

#if Backward
                    float v_loss_unclipped_dnv = 2.0f * (newvalue[i] - i_returns);
                    float v_clipped_dnv = (newvalue[i] - i_values) > args.clip_coef || (newvalue[i] - i_values) < -args.clip_coef ? 0 : 1;
                    float v_loss_clipped_dnv = v_clipped_dnv * 2.0f * (v_clipped - i_returns);
                    float max_dnv = v_loss_unclipped > v_loss_clipped ? v_loss_unclipped_dnv : v_loss_clipped_dnv;
                    d_newvalue[i] = 0.5f * max_dnv / batch;
#endif
                }
                v_loss = 0.5f * sum / batch;
            }
            else
            {
                float sum = 0;
                for (int i = 0; i < batch; i++)
                {
                    float i_returns = Get(b_returns, i);
                    sum += Mathf.Pow(newvalue[i] - i_returns, 2);
#if Backward
                    d_newvalue[i] = 0.5f * 2.0f * (newvalue[i] - i_returns) / batch;
#endif
                }
                v_loss = 0.5f * sum / batch;
            }

            //entropy_loss
            float entropy_loss_sum = 0;
            for (int i = 0; i < batch; i++)
            {
                entropy_loss_sum += entropy[i];
#if Backward
                d_entropy[i] = 1.0f / batch;
#endif
            }
            float entropy_loss = entropy_loss_sum / batch;


            float loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef;

#if Backward
            for (int i = 0; i < batch; i++)
            {
                d_entropy[i] *= -args.ent_coef;
                d_newvalue[i] *= args.vf_coef;
            }
#endif

            return loss;
        }

        private float[] NormalizeAdvantages(float[] mb_advantages)
        {
            float mean = 0;
            for (int i = 0; i < batch; i++)
            {
                mean += mb_advantages[i];
            }
            mean = mean / batch;

            float std = 0;
            for (int i = 0; i < batch; i++)
            {
                float diff = mb_advantages[i] - mean;
                std += diff * diff;
            }
            std = Mathf.Sqrt(std / (batch - 1));
            //std = Mathf.Sqrt(std / batch);

            for (int i = 0; i < batch; i++)
            {
                mb_advantages[i] = (mb_advantages[i] - mean) / (std + 1e-8f);
            }

            return mb_advantages;
        }
    }

    public class Categorical
    {
        public float[] logits;
        public float[] probs;
        public int batch;
        public int count;


        private bool hasGrad;
        //存logits对input 的雅可比矩阵 每个batch是count*count大小
        private float[] d_logits;
        private float[] d_probs;


        public Categorical(float[] inputLogits, float[] inputProbs, int batch, bool hasGrad)
        {
            if (inputLogits == null && inputProbs == null)
                throw new ArgumentNullException("Either `probs` or `logits` must be specified, but not both.");

            this.batch = batch;
            this.count = (inputLogits != null ? inputLogits.Length : inputProbs.Length) / batch;
            this.hasGrad = hasGrad;

            if (hasGrad)
            {
                d_logits = new float[batch * count * count];
                d_probs = new float[batch * count * count];
            }

            if (inputProbs == null)
            {

                this.logits = LogSumExp(inputLogits, d_logits);
                this.probs = Softmax(inputLogits, d_probs); //Softmax(LogSumExp(x)) 能抵消,相当于直接对输入Softmax,即Softmax(x)
            }
            else
            {
                this.probs = InitProb(inputProbs, d_probs);
                this.logits = ProbToLogits(this.probs, d_probs, d_probs);
            }

        }

        public float eps = 2.220446049250313e-16f;

        private float[] ProbToLogits(float[] array, float[] d_array, float[] d_result)
        {
            float[] result = new float[array.Length];
            for (int b = 0; b < batch; b++)
            {
                int offset = b * count;
                for (int i = 0; i < count; ++i)
                {
                    float prob = array[offset + i];
                    prob = Mathf.Clamp(prob, eps, 1 - eps);
                    result[offset + i] = Mathf.Log(prob);

                    if (hasGrad)
                    {
                        float d_prob = 1;
                        d_prob *= (array[offset + i] < eps || array[offset + i] > 1 - eps) ? 0 : 1;
                        d_prob *= 1 / prob;

                        for (int j = 0; j < count; j++)
                        {
                            int index = b * count * count + i * count + j;
                            d_result[index] = d_prob * d_array[index];
                        }
                    }
                }
            }
            return result;
        }


        private float[] InitProb(float[] array, float[] d_result)
        {
            float[] result = new float[array.Length];
            for (int b = 0; b < batch; b++)
            {
                int offset = b * count;
                float sum = 0.0f;
                for (int i = 0; i < count; ++i)
                {
                    sum += array[offset + i];
                }

                for (int i = 0; i < count; ++i)
                {
                    result[offset + i] = array[offset + i] / sum;

                    if (hasGrad)
                    {
                        //param i d param j
                        for (int j = 0; j < count; j++)
                        {
                            int index = b * count * count + i * count + j;
                            if (i == j)
                            {
                                d_result[index] = 1 / sum + array[offset + i] * -1 / (sum * sum);
                            }
                            else
                            {
                                d_result[index] = array[offset + i] * -1 / (sum * sum);
                            }
                        }
                    }

                }
            }
            return result;
        }


        private float[] LogSumExp(float[] array, float[] d_result)
        {
            float[] result = new float[array.Length];
            float[] d_expSum = new float[count];

            for (int b = 0; b < batch; b++)
            {
                int offset = b * count;

                var expSum = 0.0f;
                for (int i = 0; i < count; ++i)
                {
                    expSum += Mathf.Exp(array[offset + i]);

                    if (hasGrad)
                    {
                        float exp_dar = Mathf.Exp(array[offset + i]);
                        d_expSum[i] = exp_dar;
                    }
                }
                if (hasGrad)
                {
                    for (int i = 0; i < count; i++)
                    {
                        d_expSum[i] *= 1.0f / expSum;
                    }
                }
                expSum = Mathf.Log(expSum);
                for (int i = 0; i < count; ++i)
                {
                    result[offset + i] = array[offset + i] - expSum;

                    if (hasGrad)
                    {
                        //param i d param j
                        for (int j = 0; j < count; j++)
                        {
                            int index = b * count * count + i * count + j;
                            if (i == j)
                            {
                                d_result[index] = 1 - d_expSum[j];
                            }
                            else
                            {
                                d_result[index] = -d_expSum[j];
                            }
                        }
                    }

                }
            }
            return result;
        }


        private float[] Softmax(float[] array, float[] d_result)
        {

            float[] result = new float[array.Length];
            float[] d_expSum = new float[count];
            for (int b = 0; b < batch; b++)
            {
                int offset = b * count;
                var expSum = 0.0f;
                for (int i = 0; i < count; ++i)
                {
                    float exp = Mathf.Exp(array[offset + i]);
                    expSum += exp;

                    if (hasGrad)
                    {
                        float exp_dar = Mathf.Exp(array[offset + i]);
                        d_expSum[i] = exp_dar;
                    }
                }
                for (int i = 0; i < count; ++i)
                {
                    result[offset + i] = Mathf.Exp(array[offset + i]) / expSum;
                    if (hasGrad)
                    {
                        for (int j = 0; j < count; j++)
                        {
                            int index = b * count * count + i * count + j;
                            if (i == j)
                            {
                                float expSum_dar = Mathf.Exp(array[offset + i]) * -1 / (expSum * expSum) * d_expSum[i];
                                float exp_dar = 1 / expSum * Mathf.Exp(array[offset + i]);
                                d_result[index] = (expSum_dar + exp_dar);
                            }
                            else
                            {
                                float expSum_dar = Mathf.Exp(array[offset + i]) * -1 / (expSum * expSum) * d_expSum[j];
                                float exp_dar = 0;
                                d_result[index] = (expSum_dar + exp_dar);
                            }
                        }
                    }

                }
            }

            return result;
        }


        public float[] Entropy(float[] d_entropy)
        {
            float[] result = new float[batch];
            if (hasGrad)
            {
                for (int i = 0; i < d_entropy.Length; i++)
                {
                    d_entropy[i] = 0;
                }
            }
            for (int b = 0; b < batch; b++)
            {
                float entropy = 0.0f;
                for (int i = 0; i < count; i++)
                {
                    entropy += probs[b * count + i] * logits[b * count + i];

                    if (hasGrad)
                    {
                        float en_dprob = -logits[b * count + i];
                        float en_dlogit = -probs[b * count + i];

                        for (int j = 0; j < count; j++)
                        {
                            float logits_d_aj = d_logits[b * count * count + i * count + j];
                            float prob_d_aj = d_probs[b * count * count + i * count + j];
                            d_entropy[b * count + j] += en_dprob * prob_d_aj + en_dlogit * logits_d_aj;
                        }
                    }

                }
                result[b] = -entropy;
            }

            return result;
        }


        public int[] Sample()
        {
            float[] randoms = new float[batch];
            int[] result = new int[batch];

            for (int b = 0; b < batch; b++)
            {
                float cumulative = 0.0f;
                float randomValue = CommonUtils.random.NextFloat(); // 生成 [0, 1) 的随机数
                int sample = count - 1;
                for (int i = 0; i < count; i++)
                {
                    cumulative += probs[b * count + i];
                    if (cumulative > randomValue)
                    {
                        sample = i;
                        break;
                    }
                }
                result[b] = sample;
            }
            return result;
        }


        public float[] LogProb(int[] array, float[] d_log_prob)
        {
            float[] result = new float[batch];
            if (hasGrad)
            {
                for (int i = 0; i < d_log_prob.Length; i++)
                {
                    d_log_prob[i] = 0;
                }
            }
            for (int b = 0; b < batch; b++)
            {
                result[b] = logits[b * count + array[b]];

                if (hasGrad)
                {
                    //result对logits导数(count) logits对input的导数(count*count)
                    //然后 得出result对input的导数
                    for (int i = 0; i < count; i++)
                    {
                        float r_d_logits = i == array[b] ? 1 : 0;

                        for (int j = 0; j < count; j++)
                        {
                            d_log_prob[b * count + j] += r_d_logits * d_logits[b * count * count + i * count + j];
                        }
                    }
                }
            }
            return result;
        }


    }


}
