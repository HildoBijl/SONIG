% This file compares the NIGP regression and the SONIG regression algorithm using a plot of a small number of measurement points.

% We clear all data we have previously obtained.
clear all;
clc;

% We add the folders we need for both regression algorithms.
addpath('NIGP/');
addpath('NIGP/util/');
addpath('NIGP/tprod/');
addpath('SONIG/');
addpath('GPSupport/');

% We define the range of the plot we will make.
xMin = -5; % What is the minimum x value?
xMax = -xMin; % What is the maximum x value?

% We define numbers of points and set up the corresponding point spaces.
nm = 30; % This is the number of available measurement points.
np = 101; % This is the number of plot points.
nu = 11; % The number of inducing input points.
xp = linspace(xMin,xMax,np); % These are the plot points.
xu = linspace(xMin,xMax,nu); % These are the inducing input points.

% We define some settings for the noise and the GP.
sn = 0.05; % This is the noise standard deviation on the function output.
sx = 0.2; % This is the noise standard deviation on the function input.
alpha = 1; % This is the length scale of the output.
len = 1; % This is the length scale for the input. So it's the square root of Lambda.
Lambda = len^2;

% We set up the input points.
xmr = xMin + rand(1,nm)*(xMax - xMin); % These are the real measurement input points without noise.
xm = xmr + sx*randn(1,nm); % These are the measured input points.

% We calculate covariance matrices.
input = [xu,xm,xp,xmr];
diff = repmat(input,[size(input,2),1]) - repmat(input',[1,size(input,2)]);
K = alpha^2*exp(-1/2*diff.^2/Lambda);
Kuu = K(1:nu,1:nu);
Kmu = K(nu+1:nu+nm,1:nu);
Kpu = K(nu+nm+1:nu+nm+np,1:nu);
Kru = K(nu+nm+np+1:nu+nm+np+nm,1:nu);
Kum = K(1:nu,nu+1:nu+nm);
Kmm = K(nu+1:nu+nm,nu+1:nu+nm);
Kpm = K(nu+nm+1:nu+nm+np,nu+1:nu+nm);
Krm = K(nu+nm+np+1:nu+nm+np+nm,nu+1:nu+nm);
Kup = K(1:nu,nu+nm+1:nu+nm+np);
Kmp = K(nu+1:nu+nm,nu+nm+1:nu+nm+np);
Kpp = K(nu+nm+1:nu+nm+np,nu+nm+1:nu+nm+np);
Krp = K(nu+nm+np+1:nu+nm+np+nm,nu+nm+1:nu+nm+np);
Kur = K(1:nu,nu+nm+np+1:nu+nm+np+nm);
Kmr = K(nu+1:nu+nm,nu+nm+np+1:nu+nm+np+nm);
Kpr = K(nu+nm+1:nu+nm+np,nu+nm+np+1:nu+nm+np+nm);
Krr = K(nu+nm+np+1:nu+nm+np+nm,nu+nm+np+1:nu+nm+np+nm);

% To generate a random sample with covariance matrix K, we first have to find the Cholesky decomposition of K. That's what we do here.
epsilon = 0.0000001; % We add some very small noise to prevent K from being singular.
L = chol([Krr,Krp;Kpr,Kpp] + epsilon*eye(nm+np))'; % We take the Cholesky decomposition to be able to generate a sample with a distribution according to the right covariance matrix. (Yes, we could also use the mvnrnd function, but that one gives errors more often than the Cholesky function.)
sample = L*randn(nm+np,1);

% We create the measurements.
ymr = sample(1:nm)'; % These are the real function measurements, done at the real measurement input points, without any noise.
ym = ymr + sn*randn(1,nm); % We add noise to the function measurements, to get the noisy measurements.
yp = sample(nm+1:nm+np)'; % This is the function value of the function we want to approximate at the plot points.

% We make a plot of the function which we want to approximate, including the real measurements and the noisy measurements.
figure(1);
clf(1);
hold on;
grid on;
plot(xp, yp, 'b-');
plot(xmr(1:nm), ymr(1:nm), 'g+');
plot(xm(1:nm), ym(1:nm), 'ro');
xlabel('Input');
ylabel('Output');
legend('Real function','Noiseless measurements','Noisy measurements');

% The next step is to train the NIGP algorithm. We start doing that now.
seard = log([len;alpha;sn]); % We give the NIGP algorithm the true hyperparameters as starting point for its tuning. It's slightly cheating, but NIGP is likely to find the same hyperparameters with similar initializations, so this just speeds things up a little bit.
lsipn = log(sx);
evalc('[model, nigp] = trainNIGP(permute(xm,[2,1]),permute(ym,[2,1]),-500,1,seard,lsipn);'); % We apply the NIGP training algorithm. We put this in an evalc function to suppress the output made by the NIGP algorithm.

% We extract the derived hyperparameters from the NIGP results.
len = exp(model.seard(1,1));
alpha = exp(model.seard(2,1));
sn = exp(model.seard(3,1));
sx = exp(model.lsipn);
Lambda = len^2;
Sx = sx^2;
disp(['Hyperparameters found. lx: ',num2str(len),', sx: ',num2str(sx),', ly: ',num2str(alpha),', sy: ',num2str(sn),'.']);

% We recalculate covariance matrices for the new hyperparameters.
input = [xu,xm,xp,xmr];
diff = repmat(input,[size(input,2),1]) - repmat(input',[1,size(input,2)]);
K = alpha^2*exp(-1/2*diff.^2/Lambda);
Kuu = K(1:nu,1:nu);
Kmu = K(nu+1:nu+nm,1:nu);
Kpu = K(nu+nm+1:nu+nm+np,1:nu);
Kru = K(nu+nm+np+1:nu+nm+np+nm,1:nu);
Kum = K(1:nu,nu+1:nu+nm);
Kmm = K(nu+1:nu+nm,nu+1:nu+nm);
Kpm = K(nu+nm+1:nu+nm+np,nu+1:nu+nm);
Krm = K(nu+nm+np+1:nu+nm+np+nm,nu+1:nu+nm);
Kup = K(1:nu,nu+nm+1:nu+nm+np);
Kmp = K(nu+1:nu+nm,nu+nm+1:nu+nm+np);
Kpp = K(nu+nm+1:nu+nm+np,nu+nm+1:nu+nm+np);
Krp = K(nu+nm+np+1:nu+nm+np+nm,nu+nm+1:nu+nm+np);
Kur = K(1:nu,nu+nm+np+1:nu+nm+np+nm);
Kmr = K(nu+1:nu+nm,nu+nm+np+1:nu+nm+np+nm);
Kpr = K(nu+nm+1:nu+nm+np,nu+nm+np+1:nu+nm+np+nm);
Krr = K(nu+nm+np+1:nu+nm+np+nm,nu+nm+np+1:nu+nm+np+nm);

% We make the NIGP prediction for the test points.
mupNIGP = Kpm(:,1:nm)/(Kmm(1:nm,1:nm) + sn^2*eye(nm) + diag(model.dipK(1:nm)))*ym(1:nm)'; % This is the mean at the plot points.
SpNIGP = Kpp - Kpm(:,1:nm)/(Kmm(1:nm,1:nm) + sn^2*eye(nm) + diag(model.dipK(1:nm)))*Kmp(1:nm,:); % This is the covariance matrix at the plot points.
stdpNIGP = sqrt(diag(SpNIGP)); % This is the standard deviation at the plot points.
[hMean,hStd] = makeGPPlot(2, xp, mupNIGP, stdpNIGP); % We make a GP plot of the results.
hMeasurements = plot(xm(1:nm),ym(1:nm),'bx');
hFunction = plot(xp,yp,'b-');
title('Prediction of the NIGP algorithm');
xlabel('Input');
ylabel('Output');
legend([hFunction,hMeasurements,hMean,hStd],'Original function','Measurements','GP prediction mean','GP 95% certainty region','Location','SouthEast');

% We examine the results.
MSE = mean((mupNIGP' - yp).^2);
meanVariance = mean(stdpNIGP.^2);
disp(['For NIGP the MSE is ',num2str(MSE),', the mean variance is ',num2str(meanVariance),' and the ratio between these is ',num2str(MSE/meanVariance),'.']);

% Next, we set up the SONIG algorithm to make a similar kind of prediction.
% We start by setting up a SONIG object which we can apply GP regression on.
hyp = NIGPModelToHyperparameters(model);
sonig = createSONIG(hyp);
sonig = addInducingInputPoint(sonig, xu);

% We implement the measurements one by one.
for i = 1:nm
	inputDist = createDistribution(xm(:,i), hyp.sx^2); % This is the prior distribution of the input point.
	outputDist = createDistribution(ym(:,i), hyp.sy^2); % This is the prior distribution of the output point.
	[sonig, inputPost, outputPost] = implementMeasurement(sonig, inputDist, outputDist); % We implement the measurement into the SONIG object.
end

% We predict the plot points and make a plot out of it.
[mupSONIG, SpSONIG, stdpSONIG] = makeSonigPrediction(sonig, xp); % Here we make the prediction.
[hMean,hStd] = makeGPPlot(3, xp, mupSONIG, stdpSONIG); % We make a GP plot.
hMeasurements = plot(xm(1:nm),ym(1:nm),'bx');
hFunction = plot(xp,yp,'b-');
hIIP = plot(sonig.Xu,sonig.fu{1}.mean,'ko');
title('Prediction of the SONIG algorithm');
xlabel('Input');
ylabel('Output');

% We examine the results.
MSE = mean((mupSONIG' - yp).^2);
meanVariance = mean(stdpSONIG.^2);
disp(['For SONIG the MSE is ',num2str(MSE),', the mean variance is ',num2str(meanVariance),' and the ratio between these is ',num2str(MSE/meanVariance),'.']);