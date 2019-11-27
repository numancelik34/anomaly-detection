# Anomaly detection
Anomaly detection is a common problem that is applied to machine learning/deep learning research. Here we will apply an LSTM autoencoder (AE) to identify ECG anomaly detections. In our experiments, anomaly detection problem is a rare-event classification problem. Therefore we will train our LSTM AE with major class, then we would have a higher mean squared error when model sees a minor class in the dataset.

The proposed LSTM autoencoder model was trained on ECG signal sequences those obtained from MIT database normal patients. The data files are under the training folder in this repository. 
Then the model was evauated on random data files that includes ECG signal sequences and the mean squared errors are calculated as loss functions after reconstructing the ECG signals.

In the *LSTM_AE.py* file, we are attempting to identify a rare event classification problem in the given *outfinaltest62.csv* file where class '0' is a minor class in the dataset. 

## Dependencies
  * Python 3.5 or over
  * Keras tensorflow backend
  * numpy, pandas, sckitlearn and matplotlib libraries
  
This model can also be used for anomaly detection of other types of modalities such as detecting abnormalities in an image, loss in banking management, or destruction in sentiment analysis.. 

