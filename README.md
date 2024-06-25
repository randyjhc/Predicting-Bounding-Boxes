# Predicting-Bounding-Boxes

This is the week 1 programming assignment for the Advanced Computer Vision course.

In this assignment, we'll be building a model to predict bounding boxes around images.

We will use transfer learning on one of the pre-trained models available in Keras. We'll be using the [Caltech Birds - 2010 dataset](https://www.vision.caltech.edu/datasets/).

# Model Graph

We build a feature extractor using MobileNetV2.

Next, we define the dense layers following the extractor.

Lastly, we define a dense layer that outputs the bounding box predictions.

![Model Graph](./results/01_model_graph.png)

