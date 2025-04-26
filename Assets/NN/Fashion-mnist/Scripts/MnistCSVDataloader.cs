using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class MnistCSVDataloader : MonoBehaviour
{

    public string dataName = "fashion-mnist_test";

    public TMP_Text uiText;

    public Button btnNext;
    public Button btnPrev;
    public Button btnRandom;

    public Texture2D texture;

    public RawImage uiImage;


    private List<string> dataLines = new List<string>();

    public int index = 0;



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
        //SaveCSVToBin(dataName);
        string filePath = Application.dataPath + $"{CommonUtils.FashionMnistDataPath}/{dataName}.csv";
        if (File.Exists(filePath))
        {
            string[] lines = File.ReadAllLines(filePath);

            for (int i = 1; i < lines.Length; i++)
            {
                dataLines.Add(lines[i]);
            }

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
            index = Random.Range(0, dataLines.Count);
            DisplayIndex(index);
        });
    }


    public void DisplayIndex(int index)
    {
        index = Mathf.Clamp(index, 0, dataLines.Count - 1);

        string line = dataLines[index];

        string[] values = line.Split(',');

        int labelIndex = int.Parse(values[0]);


        byte[] colors = new byte[values.Length - 1];

        for (int i = 1; i < values.Length; i++)
        {
            colors[i - 1] = byte.Parse(values[i]);
        }

        if (texture != null)
            GameObject.DestroyImmediate(texture);

        texture = new Texture2D(28, 28, TextureFormat.R8, false);
        texture.LoadRawTextureData(colors);
        texture.Apply();

        uiImage.texture = texture;

        this.index = index;

        uiText.text = $"{index + 1}/{dataLines.Count} class:{class_names[labelIndex]}";
    }


    // Update is called once per frame
    void Update()
    {

    }

    public void SaveCSVToBin(string dataName)
    {
        string filePath = Application.dataPath + $"{CommonUtils.FashionMnistDataPath}{dataName}.csv";
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
