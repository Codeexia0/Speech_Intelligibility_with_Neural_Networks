# Speech Intelligibility Prediction with Neural Networks

This repository contains code for predicting speech intelligibility using deep learning models based on the Clarity Prediction Challenge ([CPC1](https://claritychallenge.org/docs/cpc1/cpc1_intro)) dataset.

This work was part of a B\.Sc. thesis that aimed to investigate and develop neural networks for speech intelligibility prediction, evaluating their effectiveness against traditional intrusive objective metrics such as STOI[^1].

---

## Overview

The project focuses on utilizing Convolutional Neural Networks (CNN) and Recurrent Convolutional Neural Networks (RCNN) to predict how intelligible speech is to human listeners, especially in noisy environments.

### Why Spectrograms and Cochleograms?

- **Spectrograms**: Provide a time-frequency representation of the audio signal, capturing the spectral features important for speech intelligibility prediction.
  
- **Cochleograms**: Mimic the human ear's processing of sound, offering a more robust feature representation, particularly in noisy environments.

Using both spectrograms and cochleograms gives the model different ways of interpreting the speech signal, improving its performance in various conditions.

---

## Models

The following deep learning models were implemented and evaluated for speech intelligibility prediction:

### CNN v1
The CNN v1 model consists of several convolutional layers designed to capture spatial features from input spectrograms and cochleograms.

<div align="center">
  <img src="images/CNN_V1_.png" width="1000">
</div>

### CNN v2
The CNN v2 model is an improved version of CNN v1, with a deeper architecture and optimized layers to achieve better performance in speech intelligibility prediction.

<div align="center">
  <img src="images/CNN_V11_.png" width="1000">
</div>

### ResCNN
The ResCNN model incorporates residual connections to allow for more efficient training and to mitigate the vanishing gradient problem, resulting in improved model performance.

<div align="center">
  <img src="images/RESCNN.png" width="1000">
</div>

---

## Experimentation

### Predicted Intelligibility Score Distributions  

The histograms below compare the predicted intelligibility score distributions across all CNN models for both **spectrogram** and **cochleogram** features.

#### Spectrogram Features  
<div align="center">
  <img src="images/his_spec.png" width="750">
</div>  

#### Cochleogram Features  
<div align="center">
  <img src="images/his_coch.png" width="750">
</div>  

### Scatter Plot of Predicted vs. True Scores  

The scatter plots below illustrate the relationship between the true and predicted intelligibility scores across all CNN models using **spectrogram** and **cochleogram** features.

#### Spectrogram Features  
<div align="center">
  <img src="images/scat_spec.png" width="750">
</div>  

#### Cochleogram Features  
<div align="center">
  <img src="images/scat_coch.png" width="750">
</div>  

### Key Findings  

- **CNN v2 with spectrogram input** achieved the best balance between accuracy and computational efficiency.  
- **ResCNN** required the longest training time but showed promising performance for deeper feature extraction.  
- **CNN v1** trained the fastest but had lower accuracy compared to CNN v2.  

---


## Conclusion of the Thesis

Through experimentation, we compared the performance of CNN and RCNN models using RMSE and CC, demonstrating that deep learning-based approaches can outperform traditional methods like STOI in speech intelligibility prediction. CNNs trained on spectrograms and cochleograms effectively captured speech intelligibility patterns, leveraging both spectral and temporal information. Among the models tested, CNN v2 with spectrogram input achieved the best trade-off between accuracy and computational efficiency.

---

## Further Work  

- **Hybrid Models**: Future research should explore hybrid architectures that combine CNNs with recurrent components to enhance temporal modeling in speech intelligibility prediction.  
- **Alternative Feature Representations**: Investigating other auditory-inspired feature representations beyond spectrograms and cochleograms may improve model robustness in diverse listening conditions.  
- **Transformer Models**: Exploring transformer-based models, such as Whisper, could provide better feature extraction and advance intelligibility prediction performance.  

## Acknowledgements
This thesis is the research and work of [Emin Shirinov](https://github.com/Codeexia0) and [George Punnoose](https://github.com/George-P-1). The dataset used in this work was provided by [The Clarity Project](https://claritychallenge.org).

[^1]: Cees H. Taal, Richard C. Hendriks, Richard Heusdens, and Jesper Jensen. “An Algorithm for Intelligibility Prediction of Time-Frequency Weighted Noisy Speech”. In: IEEE Transactions on Audio, Speech, and Language Processing 19.7 (Sept. 2011), pp. 2125–2136. doi: 10.1109/TASL.2011.2114881.
