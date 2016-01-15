# SONIG
Matlab source code for the SONIG algorithm: Sparse Online Noisy-Input Gaussian process regression.

In this GitHub repository, you find the functions applying the SONIG algorithm. With it, you can apply sparse online Gaussian process regression using noisy input points.

To use the SONIG toolbox, download the entire repository to your system. Then make sure that the paths to the appropriate folders are added to Matlab. It may be wise to walk through the files in the main directory. These are files applying the toolbox to sample problems.

To start using the SONIG toolbox in your Matlab code, first you will have to define a hyperparameter object. This is an object with four parameters:
- lx: The length scales for the input. This should be a vector of nx high and 1 wide, with nx the number of input dimensions. (Optionally, it can also be nx high and ny wide, if you want to use different input length scales for each output direction.)
- sx: The noise length scales (standard deviations) for the input. This should be a vector of nx high and 1 wide.
- ly: The length scales for the output. This should be a vector of ny high and 1 wide.
- sy: The noise length scales (standard deviations) for the output. This should be a vector of ny high and 1 wide.
Once a hyperparameter object is set up in this way, you can create a SONIG object using "sonig = createSONIG(hyperparameters);". Now your SONIG object is ready for usage. With it, you can add inducing input points (or have it done automatically), implement measurements, make predictions and more. For details on how to do this, see the respective functions from the SONIG folder.
