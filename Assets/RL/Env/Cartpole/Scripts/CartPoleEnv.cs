
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace UnityRL.CartPole
{

    //https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/classic_control/cartpole.py
    public class CartPoleEnv
    {
        private bool _sutton_barto_reward;
        public float gravity;
        public float masscart;
        public float masspole;
        public float total_mass;
        public float length;
        public float polemass_length;
        public float force_mag;
        public float tau;
        public string kinematics_integrator;
        public float theta_threshold_radians;
        public float x_threshold;

        public float[] high;

        public float[] action_space;
        public float[] observation_space_min;
        public float[] observation_space_max;

        public int steps_beyond_terminated;

        public float[] state;

        public System.Action<float[]> render;

        public int max_episode_steps = 500;

        public CartPoleEnv(bool sutton_barto_reward = false)
        {

            _sutton_barto_reward = sutton_barto_reward;

            gravity = 9.8f;
            masscart = 1.0f;
            masspole = 0.1f;
            total_mass = masspole + masscart;
            length = 0.5f;  // actually half the pole's length
            polemass_length = masspole * length;
            force_mag = 10.0f;
            tau = 0.02f; // seconds between state updates
            kinematics_integrator = "euler";

            // Angle at which to fail the episode
            theta_threshold_radians = 12.0f * 2 * Mathf.PI / 360.0f;
            x_threshold = 2.4f;

            //Angle limit set to 2 * theta_threshold_radians so failing observation
            // is still within bounds.
            high = new float[] {
                x_threshold * 2,
                float.MaxValue,
                theta_threshold_radians * 2,
                float.MaxValue};


            action_space = new float[] { 0, 1 };

            observation_space_min = new float[] { -high[0], -high[1], -high[2], -high[3] };
            observation_space_max = new float[] { high[0], high[1], high[2], high[3] };

            steps_beyond_terminated = -1;

            state = new float[4];

        }


        private float[] GetResult()
        {
            float[] result = new float[4];
            for (int i = 0; i < 4; i++)
            {
                result[i] = state[i];
            }
            return result;
        }


        public float[] Reset(float default_low = -0.05f, float default_high = 0.05f)
        {
            for (int i = 0; i < 4; i++)
            {
                state[i] = CommonUtils.random.NextFloat(default_low, default_high);
            }
            steps_beyond_terminated = -1;

            render?.Invoke(state);

            return GetResult();
        }

        public Result ResetEnv()
        {
            return new Result() { reward = 0, state = Reset(), terminated = false, truncated = false };
        }


        public Result Step(float action, bool isRender = true)
        {
            float x = state[0];
            float x_dot = state[1];
            float theta = state[2];
            float theta_dot = state[3];

            float force = action == 1 ? force_mag : -force_mag;
            float costheta = Mathf.Cos(theta);
            float sintheta = Mathf.Sin(theta);

            //For the interested reader:
            //https://coneural.org/florian/papers/05_cart_pole.pdf
            float temp = (force + this.polemass_length * theta_dot * theta_dot * sintheta) / this.total_mass;
            float thetaacc = (this.gravity * sintheta - costheta * temp) / (this.length * (4.0f / 3.0f - this.masspole * costheta * costheta / this.total_mass));
            float xacc = temp - this.polemass_length * thetaacc * costheta / this.total_mass;

            if (this.kinematics_integrator == "euler")
            {
                x = x + this.tau * x_dot;
                x_dot = x_dot + this.tau * xacc;
                theta = theta + this.tau * theta_dot;
                theta_dot = theta_dot + this.tau * thetaacc;
            }
            else  // semi-implicit euler
            {
                x_dot = x_dot + this.tau * xacc;
                x = x + this.tau * x_dot;
                theta_dot = theta_dot + this.tau * thetaacc;
                theta = theta + this.tau * theta_dot;
            }

 
            state[0] = x;
            state[1] = x_dot;
            state[2] = theta;
            state[3] = theta_dot;

            bool terminated =
                x < -this.x_threshold ||
                x > this.x_threshold ||
                theta < -this.theta_threshold_radians ||
                theta > this.theta_threshold_radians;


            float reward;
            if (!terminated)
                reward = _sutton_barto_reward ? 0.0f : 1.0f;
            else if (steps_beyond_terminated == -1)
            {
                steps_beyond_terminated = 0;
                reward = _sutton_barto_reward ? -1.0f : 1.0f;
            }
            else
            {
                if (steps_beyond_terminated == 0)
                    Debug.Log(
                        "You are calling 'step()' even though this environment has already returned terminated = True. \n"
                        +
                        "You should always call 'reset()' once you receive 'terminated = True' -- any further steps are undefined behavior."
                    );
                steps_beyond_terminated += 1;
                reward = _sutton_barto_reward ? -1.0f : 0.0f;
            }

            //bool truncated = steps_beyond_terminated >= max_episode_steps;

            if (isRender) render?.Invoke(state);

            return new Result() { reward = reward, state = GetResult(), terminated = terminated, truncated = false };
        }

        public struct Result
        {
            public float reward;
            public float[] state;
            public bool terminated;
            public bool truncated;
        }

        public void Close()
        {

        }

    }
}

