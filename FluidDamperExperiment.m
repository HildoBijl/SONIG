% This file tests the SONIG algorithm on the fluid-damper system provided via http://mathworks.com/help/ident/examples/nonlinear-modeling-of-a-magneto-rheological-fluid-damper.html. We use part
% of the data for training and the remainder for testing. In this file, we use the current output and the previous three inputs to predict the next output.

% We clear all data we have previously obtained.
clear all;
clc;

% We add the folders we need for both regression algorithms.
addpath('NIGP/');
addpath('NIGP/util/');
addpath('NIGP/tprod/');
addpath('SONIG/');
addpath('GPSupport/');

% We now load the data.
load(fullfile(matlabroot, 'toolbox', 'ident', 'iddemos', 'data', 'mrdamper.mat'));
z = iddata(F, V, Ts, 'Name', 'MR damper', ...
    'InputName', 'v', 'OutputName', 'f',...
    'InputUnit', 'cm/s', 'OutputUnit', 'N');

% We split the data up into an estimation and a validation set.
ze = z(1:2000); % This is the training (estimation) data.
zv = z(2001:end); % This is the evaluation (validation) data.

% We plot the input and output data.
figure(1);
clf(1);
plot(ze, zv);
legend('Estimation data','Validation data');

% We look at which ARX structure appears to be best to apply here.try some of the system identification functions built into Matlab.
V = arxstruc(ze,zv,struc(1:5, 1:5,1:5)); % We try values in the range 1:5 for na, nb and nk.
Order = selstruc(V,'aic'); % We select the orders by Akaike's Information Criterion.

% We set up some linear models and see how well they perform.
LinMod1 = arx(ze, [2 4 1]); % We set up an ARX model Ay = Bu + e.
LinMod2 = oe(ze, [4 2 1]); % We set up an OE model y = B/F u + e.
LinMod3 = ssest(ze); % We create a state space model of order 3.
% compare(ze, LinMod1, LinMod2, LinMod3); % We compare the models with the training set.
compare(zv, LinMod1, LinMod2, LinMod3); % We compare the models with the evaluation set.

% We ask for advice on whether the system has any nonlinear behavior. Based on this advice, we set up a nonlinear ARX model.
% advice(ze, 'nonlinearity'); 
Options = nlarxOptions('SearchMethod', 'lm'); % We use the Levenberg-Marquardt least-squares search.
Options.SearchOption.MaxIter = 50;
Narx1 = nlarx(ze, [2 4 1], 'sigmoidnet',Options); % We set up the nonlinear ARX model using a sigmoidal function approximation.
% compare(ze, Narx1); % We compare the model with the training set.
compare(zv, Narx1); % We compare the model with the evaluation set.

% Next, we try some other nonlinear ARX model structures to see how well they do.
Narx2{1} = nlarx(ze, [3 4 1], 'sigmoidnet',Options); Narx2{1}.Name = 'Narx2_1';
Narx2{2} = nlarx(ze, [2 5 1], 'sigmoidnet',Options); Narx2{2}.Name = 'Narx2_2';
Narx2{3} = nlarx(ze, [3 5 1], 'sigmoidnet',Options); Narx2{3}.Name = 'Narx2_3';
Narx2{4} = nlarx(ze, [1 4 1], 'sigmoidnet',Options); Narx2{4}.Name = 'Narx2_4';
Narx2{5} = nlarx(ze, [2 3 1], 'sigmoidnet',Options); Narx2{5}.Name = 'Narx2_5';
Narx2{6} = nlarx(ze, [1 3 1], 'sigmoidnet',Options); Narx2{6}.Name = 'Narx2_6';
% compare(ze, Narx1, Narx2{:}); % We compare the models with the training set.
compare(zv, Narx1, Narx2{:}); % We compare the models with the evaluation set.

% We calculate and plot the simulation results of the best NARX model.
simResults = sim(Narx2{6}, zv);
num = size(simResults.y,1);
t = 0:simResults.Ts:(num-1)*simResults.Ts;
figure(5);
clf(5);
hold on;
grid on;
plot(t,simResults.y);
plot(t,zv.y);
xlabel('Input');
ylabel('Output');
title('Prediction by the best nonlinear ARX model');
legend('Simulation results','Real results');
% We also evaluate the simultion results.
diff = zv.y - simResults.y;
dataFit = 100*(1-norm(diff)/norm(zv.y-mean(zv.y)));
RMSE = sqrt(mean(diff.^2));
disp(['The best NARX model gave a fit of ',num2str(dataFit),'%. The RMSE was ',num2str(RMSE),'.']);

% Now it's time to try the SONIG algorithm. We set up the input data for the algorithm first.
u = ze.u';
y = ze.y';
nm = size(u,2);

% We define length scales and measurement noise.
lu = 14;
ly = 70;
su = 0.1;
sy = 2;

% We define hyperparameters.
hyp.sx = [su;su;su;sy];
hyp.sy = sy;
hyp.lx = [10;10;20;ly];
hyp.ly = ly;

% Next, we set up a SONIG object which we can apply GP regression on.
sonig = createSONIG(hyp);
sonig.addIIPDistance = 30/ly; % This is the distance (normalized with respect to the length scales) above which new inducing input points are added.

% We now start to implement measurements.
tic;
lastDisplay = toc;
disp('Starting to implement measurements into the SONIG algorithm.');
ypo = y; % This will contain the posterior mean of y.
upo = u; % This will contain the posterior mean of u.
ystd = sy*ones(size(ypo)); % This will contain the poster standard deviation of y.
ustd = su*ones(size(upo)); % This will contain the poster standard deviation of u.
jointMean = [0;u(1);u(2);0;y(3)]; % This will be the joint vector of inputs and outputs which we're currently applying. It will basically "shift through time" upon each iteration. The first three entries are for inputs (being u(k-2), u(k-1), u(k)) and the other two entries are for outputs (being y(k) and y(k+1)).
jointCov = [su^2*eye(3),zeros(3,2);zeros(2,3),sy^2*eye(2)]; % This will contain the covariance matrix of the SONIG input which we're currently applying.
informationTimeStep = 5; % This is the time after which we give information on how far the algorithm is along.
for i = 3:nm-1 % We walk over all measurements.
	% We display regularly timed updates.
	while toc > lastDisplay + informationTimeStep
		lastDisplay = lastDisplay + informationTimeStep;
		disp(['Time passed is ',num2str(lastDisplay),' seconds. We are currently at measurement ',num2str(i-2),' of ',num2str(nm-3),', with ',num2str(sonig.nu),' IIPs.']);
	end
	% We set up the input and the output distributions, taking into account all the covariances of the parameters.
	jointMean = [jointMean(2:3);u(i);jointMean(5);y(i+1)]; % We shift the mean matrix one further.
	jointCov = [jointCov(2:3,2:3),zeros(2,1),jointCov(2:3,5),zeros(2,1);...
				zeros(1,2),su^2,zeros(1,2);...
				jointCov(5,2:3),0,jointCov(5,5),0;...
				zeros(1,4),sy^2]; % We shift the covariance matrix one further.
	jointDist = createDistribution(jointMean, jointCov); % We create a distribution from the mean and the covariance. This is the joint distribution for the SONIG input and SONIG output.
	inputDist = getSubDistribution(jointDist, [1,2,3,4]); % We extract the SONIG input, which is evidently required by the SONIG algorithm.
	outputDist = getSubDistribution(jointDist, 5); % We extract the SONIG output, which is also required.
	% We implement the measurement into the SONIG algorithm.
	[sonig, inputPost, outputPost, jointPost] = implementMeasurement(sonig, inputDist, outputDist, jointDist);
	% We update the distributions of all our points.
	jointMean = jointPost.mean; % We will need this one for the next iteration.
	jointCov = jointPost.cov; % We will need this one for the next iteration.
	ypo(i:i+1) = jointPost.mean([4,5]);
	upo(i-2:i) = jointPost.mean([1,2,3]);
	stds = sqrt(diag(jointCov)');
	ystd(i:i+1) = stds([4,5]);
	ustd(i-2:i) = stds([1,2,3]);
end
disp(['Finished implementing ',num2str(sonig.nm),' measurements in ',num2str(toc),' seconds, using ',num2str(sonig.nu),' IIPs.']);

% We run a simulation based on the output of the system, for the trial data.
ut = [ze.u(end-2:end)',zv.u']; % We set up the input for the simulation. For this input, we also take the last two inputs of the training phase.
ys = zeros(size(zv.y)); % We will store our predicted outputs in this array.
ystd = ys; % We will store our predicted standard deviations in this array.
yDist = createDistribution(zv.y(1),1e-9); % This will be the distribution of the current output. We will shift it forward in time every time. We initialize it as the first output of the validation set, assuming we at least know our starting point.
for i = 4:size(ut,2)
	xDist = createDistribution([ut(i-3);ut(i-2);ut(i-1);yDist.mean],[eye(3)*su^2,zeros(3,1);zeros(1,3),yDist.cov]); % This is the (prior) distribution of the input for the SONIG algorithm.
	yDist = makeSonigStochasticPrediction(sonig, xDist); % We make a prediction for the output, taking into account the stochastic nature of the input.
	ys(i-3) = yDist.mean; % We extract the mean and store it.
	ystd(i-3) = sqrt(yDist.cov); % We extract the standard deviation and store it.
end

% We evaluate the results.
diff = zv.y - ys;
dataFit = 100*(1-norm(diff)/norm(zv.y-mean(zv.y)));
RMSE = sqrt(mean(diff.^2));
disp(['The SONIG algorithm gave a fit of ',num2str(dataFit),'%. The RMSE was ',num2str(RMSE),'.']);

% We plot the simulation data which we made.
figure(6);
clf(6);
hold on;
grid on;
num = size(ys,1);
t = 0:simResults.Ts:(num-1)*simResults.Ts;
patch([t';flipud(t')],[ys-2*ystd;flipud(ys+2*ystd)],1,'FaceColor',[1,1,1]*0.75,'EdgeColor','none');
set(gca,'layer','top');
fNLARX = plot(t,simResults.y,'-','LineWidth',0.5,'color',[0.8,0,0]);
fSONIG = plot(t,ys,'-','LineWidth',0.5,'color',[0,0,0]);
ftrue = plot(t,zv.y,'-','color',[0,0,0.8]);
legend([ftrue, fSONIG, fNLARX], 'Real measurements', 'SONIG predictions', 'NLARX predictions','Location','NorthEast');
axis([0,7.5,-80,120]);
xlabel('Time [s]');
ylabel('Force [N]');
%% Out of curiosity, we also apply the exact same method - training the NARX function - for regular GP and the NIGP method. First up is GP.

% We set up the measurement matrix and vector.
X = [ze.u(1:end-3)';ze.u(2:end-2)';ze.u(3:end-1)';ze.y(3:end-1)'];
yr = [ze.y(4:end)];

% We apply training of the GP.
diff = repmat(permute(X,[3,2,1]),size(X,2),1) - repmat(permute(X,[2,3,1]),1,size(X,2)); % This is matrix containing differences between input points. We have rearranged things so that indices 1 and 2 represent the numbers of vectors, while index 3 represents the element within the vector.
K = hyp.ly^2*exp(-1/2*sum(diff.^2./repmat(permute(hyp.lx.^2,[3,2,1]),size(X,2),size(X,2)), 3)); % This is the covariance matrix. It contains the covariances of each combination of points.
Sfm = hyp.sy^2*eye(length(yr)); % This is the noise covariance matrix.
beta = (K+Sfm)\yr;

% We now run a simulation.
xs = [ze.u(end-2);ze.u(end-1);ze.u(end);ze.y(end)];
res = zeros(size(zv.y));
for i = 1:length(zv.u)
	Ks = hyp.ly^2*exp(-1/2*sum((repmat(xs,1,size(X,2)) - X).^2./repmat(hyp.lx.^2,1,size(X,2)),1));
	ys = Ks*beta;
	res(i) = ys;
	xs = [xs(2);xs(3);zv.u(i);ys];
end

% And we analyze the data.
err = zv.y - res;
dataFit = 100*(1-norm(err)/norm(zv.y-mean(zv.y)));
RMSE = sqrt(mean(err.^2));
disp(['The GP algorithm gave a fit of ',num2str(dataFit),'%. The RMSE was ',num2str(RMSE),'.']);

%% And we do the same for the NIGP method.

% We set up the training. This could take a while.
seard = log([hyp.lx;hyp.ly;hyp.sy]);
lsipn = log(hyp.sx);
tic;
disp('Starting the NIGP training. (This may take five minutes or so.)');
[model, nigp] = trainNIGP(permute(X,[2,1]),yr,-500,1,seard,lsipn); % You can put this in the evalc function to suppress output.

% We extract the derived settings and perform training.
lx = exp(model.seard(1:4,1));
lf = exp(model.seard(5,1));
sfm = exp(model.seard(6,1));
sxm = exp(model.lsipn);
K = lf^2*exp(-1/2*sum(diff.^2./repmat(permute(lx.^2,[3,2,1]),size(X,2),size(X,2)), 3)); % This is the covariance matrix. It contains the covariances of each combination of points.
beta = (K + sfm^2*eye(size(X,2)) + diag(model.dipK))\yr;

% We run the simulation.
xs = [ze.u(end-2);ze.u(end-1);ze.u(end);ze.y(end)];
res = zeros(size(zv.y));
for i = 1:length(zv.u)
	Ks = lf^2*exp(-1/2*sum((repmat(xs,1,size(X,2)) - X).^2./repmat(lx.^2,1,size(X,2)),1));
	ys = Ks*beta;
	res(i) = ys;
	xs = [xs(2);xs(3);zv.u(i);ys];
end

% And we analyze the data.
err = zv.y - res;
dataFit = 100*(1-norm(err)/norm(zv.y-mean(zv.y)));
RMSE = sqrt(mean(err.^2));
disp(['The NIGP algorithm gave a fit of ',num2str(dataFit),'%. The RMSE was ',num2str(RMSE),'.']);