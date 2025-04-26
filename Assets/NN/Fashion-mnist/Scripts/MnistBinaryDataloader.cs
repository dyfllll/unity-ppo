using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class MnistBinaryDataloader : MonoBehaviour
{
    public string dataName = "fashion-mnist_test";

    public TMP_Text uiText;

    public Button btnNext;
    public Button btnPrev;
    public Button btnRandom;

    public Texture2D texture;

    public RawImage uiImage;


    public int index = 0;
    public int count = 0;

    private int lineDataSize = 785;

    private byte[] dataArray;


    public string[] class_names = new string[]
    {"T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
    };
    // Start is called before the first frame update
    void Start()
    {
        string filePath = Application.dataPath + $"{CommonUtils.FashionMnistDataPath}/{dataName}.bin";



        if (File.Exists(filePath))
        {
            dataArray = File.ReadAllBytes(filePath);

            count = dataArray.Length / lineDataSize;

            Debug.Log($"read {dataArray.Length} total:{count}");

            DisplayIndex(0);
        }
        else
        {
            Debug.LogError("File not found at: " + filePath);
        }

        btnNext.onClick.AddListener(() =>
        {
            DisplayIndex(++index);
        });

        btnPrev.onClick.AddListener(() =>
        {
            DisplayIndex(--index);
        });
        btnRandom.onClick.AddListener(() =>
        {
            index = Random.Range(0, count);
            DisplayIndex(index);
        });
    }


    public void DisplayIndex(int index)
    {
        index = Mathf.Clamp(index, 0, count - 1);


        int start = index * lineDataSize;



        int labelIndex = dataArray[start];


        byte[] colors = new byte[lineDataSize - 1];

        for (int i = 1; i < lineDataSize; i++)
        {
            colors[i - 1] = dataArray[start + i];
        }

        if (texture != null)
            GameObject.DestroyImmediate(texture);

        texture = new Texture2D(28, 28, TextureFormat.R8, false);
        texture.LoadRawTextureData(colors);
        texture.Apply();

        uiImage.texture = texture;

        this.index = index;

        uiText.text = $"{index + 1}/{count} class:{class_names[labelIndex]} labelIndex:{labelIndex}";
    }


    // Update is called once per frame
    void Update()
    {

    }

    public void SaveCSVToBin(string dataName)
    {
        string filePath = Application.dataPath + $"{CommonUtils.FashionMnistDataPath}/{dataName}.csv";
        string savePath = Application.dataPath + $"{CommonUtils.FashionMnistDataPath}/{dataName}.bin";


        using FileStream fs = new FileStream(savePath, FileMode.OpenOrCreate);
        byte[] buffer = new byte[785];

        if (File.Exists(filePath))
        {
            string[] lines = File.ReadAllLines(filePath);


            Debug.Log($"total: {lines.Length - 1}");

            for (int i = 1; i < lines.Length; i++)
            {
                string line = lines[i];

                string[] values = line.Split(',');

                for (int j = 0; j < values.Length; j++)
                {
                    buffer[j] = byte.Parse(values[j]);
                }

                fs.Write(buffer, 0, buffer.Length);
            }




        }
        fs.Flush();

        Debug.Log($"save to: {savePath}");
    }

    private void OnDestroy()
    {
        if (texture)
            GameObject.DestroyImmediate(texture);
    }
}
