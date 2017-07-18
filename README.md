# SONIG
Here you find the Matlab source code for the SONIG algorithm: Sparse Online Noisy-Input Gaussian process regression. You can find the paper I wrote on it through [arXiv](https://arxiv.org/abs/1601.08068). For a full introduction into Gaussian process regression, including the SONIG algorithm, you can read my [Ph.D. thesis](http://hildobijl.com/Downloads/GPRT.pdf), available through my [personal website](http://hildobijl.com/Research.php).

## Setting up the SONIG toolbox

In this GitHub repository, you find the functions applying the SONIG algorithm. With it, you can apply sparse online Gaussian process regression using noisy input points. This source code accompanies the paper on the SONIG algorithm. It would be wise to read that paper before applying the SONIG toolbox so you at least have a clue about what's going on.

To use the SONIG toolbox, download the entire repository to your system. Then make sure that the paths to the appropriate folders are added to Matlab. It may be wise to walk through the files in the main directory. These are files applying the toolbox to sample problems.

## Setting up a SONIG object

To start using the SONIG toolbox in your Matlab code, first you will have to define a hyperparameter object. This is an object with four parameters:
- `lx`: The length scales for the input. This should be a vector of `dx` high and 1 wide, with `dx` the number of input dimensions. (Optionally, it can also be `dx` high and `dy` wide, if you want to use different input length scales for each output direction.)
- `sx`: The noise length scales (standard deviations) for the input. This should be a vector of `dx` high and 1 wide.
- `ly`: The length scales for the output. This should be a vector of `dy` high and 1 wide.
- `sy`: The noise length scales (standard deviations) for the output. This should be a vector of `dy` high and 1 wide.

So an example of code that sets this up, for `dx = 3` and `dy = 2`, is
```
hyperparameters.lx = [1;1;0.1];
hyperparameters.sx = 0.1*hyperparameters.lx;
hyperparameters.ly = [10;20];
hyperparameters.sy = [0.1;0.1];
```
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
It is possible to add one inducing input point `xu` (which should then be a vector of size `dx` by `1`) or add multiple points at the same time (in which case `xu` is a matrix of size `dx` by `nu`, with `nu` the number of inducing input points to be added).

The second option is to set the `addIIPDistance` parameter of the SONIG object, like in
```
sonig.addIIPDistance = 1;
```
Now, whenever the SONIG algorithm is given a new measurement, it will check if the input point of this measurement is close to any already existing IIPs. (With 'close' here meaning 'within the normalized IIP distance that is given'.) If it's not close to any IIP, then the SONIG algorithm will add the measurement input point as an inducing input point.

Should your algorithm become too slow, because too many inducing input points are added, then you can increase this distance. This will generally make sure less inducing input points are added.

## Working with distributions

Before we continue, we should note that the SONIG algorithm works with Gaussian distributions. These distributions have a mean and a covariance. Any input point and any output value of the SONIG algorithm should be given as a distribution object.

To make a distribution object, you can use
```
mean = [0;0]; % This is just an example mean vector.
cov = [1,0;0,2]; % This is just an example covariance vector.
dist = createDistribution(mean, cov);
```
The resulting `dist` parameter now is a distribution object.

It is also possible to merge distributions, if their covariance matrix is known. This is done through
```
x = createDistribution(0, 1);
y = createDistribution([0;0], eye(2));
xyCovariance = [0.2, 0.4];
xy = joinDistributions(x, y, xyCovariance);
```
There are more functions working with distributions, but for now you know enough of the basics. Let's use them.

## Implementing measurements

Next, we can start implementing measurements into the SONIG framework. This can only be done one measurement at a time. This is done using
```
[sonig, inputPost, outputPost, jointPost] = implementMeasurement(sonig, inputPrior, outputPrior, jointPrior);
```
Here, the `inputPrior` and the `outputPrior` are distribution objects. The `jointPrior` object is optional. In case the `inputPrior` and `outputPrior` distributions are correlated, this correlation can be taken into account by setting up the `jointPrior` distribution of the `inputPrior` and the `outputPrior`. If omitted, it is assumed that the `inputPrior` and the `outputPrior` are independent.

The resulting output of the function will be the posterior distributions of the input and the output, as well as their posterior joint distribution `jointPost`. For some applications, this may be useful data. More importantly, the information extracted from the measurement is now stored in the SONIG object, and it will be taken into account whenever predictions will be made.

## Making predictions

There are two ways to make predictions. First of all, we can make predictions for deterministic input points. This is done through
```
[postMean, postCov, postStd] = makeSonigPrediction(sonig, xt);
```
For a single input point, `xt` should be a vector of size `dx` by `1`. For multiple input points, `xt` should be a matrix of size `dx` by `nt`, with `nt` the number of trial input points. In this case, the outcome will be given by three parameters.
- `postMean` is an `nt` by `dy` matrix containing the mean vectors of all the predictions.
- `postCov` is an `nt` by `nt` by `dy` three-dimensional matrix containing the covariance matrix between all input points for each output dimension.
- `postStd` is an `nt` by `dy` matrix containing the standard deviations for each of the predictions. This is basically the square root of the diagonal of the `postCov` matrix.
 
It is also possible to make predictions for stochastic trial points. In this case, we can use
```
xDist = createDistribution([0;0], eye(2)); % This is just an example distribution.
[fDist] = makeSONIGStochasticPrediction(xDist);
```
Here `xDist` should naturally be a distribution of size `dx`. The resulting `fDist` parameter will be a distribution of size `dy`. When making this prediction, the uncertainty present in `xDist` is naturally taken into account.

## Further remarks

The SONIG toolbox provided here is far from a finished product. You may encounter bugs while using it. Should you find any, make sure to notify us of them, so we try to fix them. And other feedback is of course also always welcome.
