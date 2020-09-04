# Handwritten Letter Classification
In this project, the Keras and TensorFlow packages were used to implement a neural network model. A sequential model was used to identify grayscale images of handwritten letters. These images were converted to vectors of pixel values beforehand. The pixel values were pre-processed via binarization and pixels with low variability were omitted.

After training, the model achieved an accuracy **over 98%**. All letters had a recall rate and precision rate of over 90%. Most of the letters obtained over 95% recall rate and some even achieved upwards 99%. Similarly, the precision rates were also very high as they were mostly over 97%.

A brief analysis of the prevalence of the data showed that some letters appeared more often than others. For example, the letters O and S appeared the most, while I and F appeared the least. Despite the disparity, each letter had over 1,000 images.

The data can be accessed here:
<https://www.kaggle.com/ashishguptajiit/handwritten-az>