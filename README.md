# SONIG
Here you find the Matlab source code for the SONIG algorithm: Sparse Online Noisy-Input Gaussian process regression.

## Setting up the SONIG toolbox

In this GitHub repository, you find the functions applying the SONIG algorithm. With it, you can apply sparse online Gaussian process regression using noisy input points. This source code accompanies the paper on the SONIG algorithm. It would be wise to read that paper before applying the SONIG toolbox so you at least have a clue about what's going on.

To use the SONIG toolbox, download the entire repository to your system. Then make sure that the paths to the appropriate folders are added to Matlab. It may be wise to walk through the files in the main directory. These are files applying the toolbox to sample problems.

## Setting up the SONIG object

To start using the SONIG toolbox in your Matlab code, first you will have to define a hyperparameter object. This is an object with four parameters:
- `lx`: The length scales for the input. This should be a vector of `nx` high and 1 wide, with `nx` the number of input dimensions. (Optionally, it can also be `nx` high and `ny` wide, if you want to use different input length scales for each output direction.)
- `sx`: The noise length scales (standard deviations) for the input. This should be a vector of `nx` high and 1 wide.
- `ly`: The length scales for the output. This should be a vector of `ny` high and 1 wide.
- `sy`: The noise length scales (standard deviations) for the output. This should be a vector of `ny` high and 1 wide.

Once a hyperparameter object is set up in this way, you can create a SONIG object using
```
sonig = createSONIG(hyperparameters);
```
Now your SONIG object is ready for usage. With it, you can add inducing input points (or have it done automatically), implement measurements, make predictions and more.

Keep in mind that, once set, the hyperparameters cannot change anymore. Changing them anyway would result in invalid results.

## Adding inducing input points

The SONIG algorithm makes use of Inducing Input Points (IIPs). Very simply put, these are points at which the SONIG algorithm will try to approximate and remember the function value. It then uses these remembered function values when making predictions about other points.

There are two ways to add inducing input points. The first way is to do so manually. This is done through
```
sonig = addInducingInputPoint(sonig, xu);
```
It is possible to add one inducing input point `xu` (which should then be a vector of size `nx` by `1`) or add multiple points at the same time (in which case `xu` is a matrix of size `nx` by `nu`, with `nu` the number of inducing input points to be added).

The second option is to set the `addIIPDistance` parameter of the SONIG object, like in
```
sonig.addIIPDistance = 1;
```
Now, whenever the SONIG algorithm is given a new measurement, it will check if the input point of this measurement is close to any already existing IIPs. (With 'close' here meaning 'within the normalized IIP distance that is given'.) If it's not close to any IIP, then the SONIG algorithm will add the measurement input point as an inducing input point.

Should your algorithm become too slow, because too many inducing input points are added, then you can increase this distance. This will generally make sure less inducing input points are added.

## Implementing measurements

TODO

