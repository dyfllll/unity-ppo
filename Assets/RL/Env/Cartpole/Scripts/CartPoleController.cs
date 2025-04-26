using Cysharp.Threading.Tasks;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
namespace UnityRL.CartPole
{
    public class CartPoleController : MonoBehaviour
    {
        public CartPoleEnv env;
        public int seed = 1000;

        public CartPoleAgent agent;

        public Transform pole;
        public Transform cart;

        public float poleLen;

        public Transform leftBounds;
        public Transform rightBounds;

        public bool isRender = true;
        // Start is called before the first frame update
        void Start()
        {
            poleLen = pole.transform.position.y;


            env = new CartPoleEnv();
            env.render = Render;

            leftBounds.transform.position = new Vector3(-env.x_threshold, 0, 0);
            rightBounds.transform.position = new Vector3(env.x_threshold, 0, 0);

            agent.Init(env);
        }



        public void Render(float[] state)
        {
            if (!isRender) return;

            float pos = state[0];
            float angle = state[2];

            float offset = pole.transform.position.y;

            Quaternion poleRot = Quaternion.Euler(0, 0, -angle * Mathf.Rad2Deg);
            Vector3 polePos = Vector3.Normalize(poleRot * Vector3.up) * poleLen;

            cart.transform.position = new Vector3(pos, 0, 0);

            pole.transform.position = cart.transform.position + polePos;
            pole.transform.localRotation = poleRot;
        }


        // Update is called once per frame
        void Update()
        {
            agent.Update();
            //int action = 2;
            //if (Input.GetKeyDown(KeyCode.A))
            //{
            //    action = 0;
            //}

            //if (Input.GetKeyDown(KeyCode.D))
            //{
            //    action = 1;
            //}

            //if (action != 2)
            //{
            //    var result = env.Step(action);

            //    Debug.Log(result.reward);
            //    if (result.terminated || result.truncated)
            //        env.Reset();
            //}

            //Render(new float[] { testPos, 0, testRot * Mathf.Deg2Rad, 0 });

        }

        private void OnDestroy()
        {
            agent?.Release();
        }
    }

}
