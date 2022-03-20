# Predicting-Distraction

This is a part of my graduate research at the University of Tennessee-Knoxville that processes driving data from multiple .xdf files, combines them, and writes it to a new .npy file. This data is then classified and analyzed with the relevant .py files. 

# Data

The feature matrix is composed of 13 different time series taking the form (n_samples,n_features,n_timesteps). Each set of time series data is classified as being part of 1 of 3 total classes: undistracted, distracted, and very distracted. This gives your label vector dimensions (n_samples,). In the Statistics.py file, this changes to give each point in the time series a label, so the label vector becomes an additional "feature" when working in that file.

# Experiment

The data was obtained in a VR simulation designed to collect drivers' data on the same stretch of road under varying levels of distraction. The distraction for this experiment was a grid of arrows on the center console of the vehicle, where one arrow faced a different direction to the others, and that arrow's grid position had to be identified verbally before showing a different grid of arrows. The size of the grid was increased for the 'Very Distracted' state.
