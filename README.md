# 1. Dependencies
	1.1.python == 3.6.3
	1.2.h5py == 2.10.0
	1.3.numpy == 1.19.5
	1.4.torch == 1.10.2
Model has been written and tested with the above dependencies. Performance with other versions has not been tested.
# 2. Steps to use the model to process data
## Step1: Preprocess the original FID (free induction decay). 
Firstly, zero-fill the original FID with zeros to match the network input dimensions (Zero-fill the FID containing fewer than 4096 complex points to 4096 complex points. And if the FID contains more than 4096 complex points, zero-fill it to 8192 or more (integer multiples of 4096). Then, divide the spectrum into two or more spectra, use spectra with 4096 points as the network input, and finally splice the processed spectra to obtain a complete spectrum.). Secondly, apply the Fourier transform on the padded FID, take the real part, and finally normalize it based on the maximum value. Then, use MATLAB or Python to store the normalized data (to be processed, the input of the network) in '.mat' format (mat v7.3) with the file name 'data.mat'.
## Step 2: Process the 'data.mat' file using the network. 
Run the 'detector.py' file with loading the trained network weight file named 'net.pt' in the 'params' folder to process the data. The output of the network will be saved in the '.mat' format in the 'predict' folder, and can be further visualized through plotting using the 'plot_result.m' file.
# 3. Experimental data
Four experimental data including 'exp_asarone.mat', 'exp_azithromycin.mat', 'exp_estradiol.mat', and 'exp_mixture.mat' used in our manuscript are provided in the 'exp' folder, which can be used as example data.