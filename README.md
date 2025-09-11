# Massachusetts Building Segmentation

This repository provides a complete deep learning pipeline for **building footprint segmentation** using the [Massachusetts Buildings Dataset](https://www.cs.toronto.edu/~vmnih/data/). The model generates **binary segmentation masks**, accurately identifying building structures in large-scale aerial imagery (1000x1000 px).

## Key Features & Design Choices

* **Efficient U-Net Architecture**
  A **lightweight U-Net** with an *EfficientNet-B3* encoder backbone, for fast inference.

* **Multi-Scale Feature Extraction with Inception Modules**
  Tested **inception-style convolutional blocks** in the decoder to improve **scale invariance**, allowing the model to capture both fine building details and larger structures effectively.

* **Transfer Learning for Data Efficiency**
  Leveraging pretrained ImageNet weights to overcome **limited dataset size**.

* **Hybrid Loss Function**
  A combination of **Weighted Binary Cross-Entropy** and **Dice Loss** balances pixel-level precision with overall shape quality.
  *(Focal and Tversky losses were also tested and yielded similar performance.)*

* **Dynamic Learning Rate Scheduling**
  **Cosine Annealing** is used to adjust the learning rate during training, helping the model converge.

## Results

Training with the EfficientNet-B3 backbone and inception-based multi-scale features achieved:

* **Validation IoU:** \~0.64
* **Validation Dice Score:** \~0.78

These results demonstrate **robust segmentation performance** in complex urban environments, even with relatively limited labeled data.

## Run Inference

You can generate building segmentation masks on new aerial images using:

```bash
python src/inference.py --input path/to/image.png --output predictions/
```

## Example Predictions

Some sample outputs showing predicted building footprints highlighted in white:

<p>
  <img src="./assets/inference_4.png" width="900">
  <img src="./assets/inference_2.png" width="900">
  <img src="./assets/inference_0.png" width="900">
</p>