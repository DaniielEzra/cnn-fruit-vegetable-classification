# Put the trained model here

This folder is empty until you train the model in Google Colab.

After you run `notebook.ipynb` and download `vision_outputs.zip`, copy these files into this folder:

- `vision_model.keras` (the trained CNN)
- `model_info.json` (TensorFlow version, image size, test accuracy)

The final structure must be:

```
project03_vision/
    model/
        vision_model.keras
        model_info.json
```

The website will not classify images until **both** files exist. `model_info.json` is required: the server reads the confidence threshold from it and checks the class order and preprocessing against it.
