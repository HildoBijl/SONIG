% This file compares many ways of GP regression for noisy input measurements. It compares their performance based on various scenarios.
% 1. Perfect GP regression, with exact input measurements (but noisy output measurements) and exact hyperparameters. (Only 200 measurements.)
% 2. Regular GP regression, with noisy input measurements and tuned hyperparameters. (Only 200 measurements.)
% 3. Regular FITC, using the hyperparameters of (2). (The full 800 measurements.)
% 4. The NIGP algorithm, which tunes the hyperparameters itself. (Only 200 measurements.)
% 5. The SONIG algorithm, using the hyperparameters of (4). (Only 200 measurements.)
% 6. The SONIG algorithm, using the hyperparameters of (4). (The full 800 measurements, which gives a roughly equal runtime as (4).)
% 7. The SONIG algorithm, getting an initial estimate using a subset (100) measurement points of the NIGP algorithm. (A total of 800 measurements are used.)
% Note that this order is different than the one in the paper. It just makes it a bit easier to execute in this order.

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
nm = 800; % This is the total number of available measurement points. They are used by algorithms 3, 6 and 7.
nmu = nm/4; % This is the number of measurements used by algorithms 1, 2, 4 and 5.
nmn = nm/8; % This is the number of measurements used for NIGP training in algorithm 7.
np = 101; % This is the number of plot points.
nu = 21; % The number of inducing input points.
xp = linspace(xMin,xMax,np); % These are the plot points.
xu = linspace(xMin,xMax,nu); % These are the inducing input points.

% We do various experiments. We start looping through them here. We also define some storage parameters.
numIterations = 10; % How many functions do we approximate? For the paper we used 200. Keep in mind that (at least on my system) one iteration lasts roughly 20 seconds.
numMethods = 7; % How many algorithms do we have? This is 7, unless you add algorithms yourself.
res = zeros(numMethods,3,numIterations);
muuS = zeros(nu,numMethods,numIterations);
mupS = zeros(np,numMethods,numIterations);
SuS = zeros(nu,nu,numMethods,numIterations);
SpS = zeros(np,np,numMethods,numIterations);
xmS = zeros(nm,numIterations);
xmrS = zeros(nm,numIterations);
ymS = zeros(nm,numIterations);
ymrS = zeros(nm,numIterations);
ypS = zeros(np,numIterations);
counter = 0; % We initialize a counter.
while counter < numIterations
	% We set up some preliminary stuff for the loop.
	counter = counter + 1;
	flag = 0; % This is a flag which checks for problems.
	disp(['Starting loop ',num2str(counter),'.']);

	% We define some settings for the function which we will generate. We generate it by sampling from a GP.
	sn = 0.1;
	sx = 0.4;
	len = 1;
	alpha = 1;
	Lambda = len^2;
	realParams = [len,sx,alpha,sn];

	% We set up the input points and the corresponding covariance matrices.
	xmr = xMin + rand(1,nm)*(xMax - xMin); % These are the real measurement input points.
	xm = xmr + sx*randn(1,nm); % These are the input points corrupted by noise.
	xmrS(:,counter) = xmr'; % We store the measurement points, in case we want to inspect them later.
	xmS(:,counter) = xm'; % We store the measurement points, in case we want to inspect them later.
	
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

	% We create and store the measurements.
	ymr = sample(1:nm)'; % These are the real function measurements, done at the real measurement input points, without any noise.
	ym = ymr + sn*randn(1,nm); % We add noise to the function measurements, to get the noisy measurements.
	yp = sample(nm+1:nm+np)'; % This is the function value of the function we want to approximate at the plot points.
	ymrS(:,counter) = ymr';
	ymS(:,counter) = ym';
	ypS(:,counter) = yp';

	% Method 1.
	% We first set up a GP of the true measurements (with output noise but without input noise) with the actual hyperparameters.
	mupGPn = Kpr(:,1:nmu)/(Krr(1:nmu,1:nmu) + sn^2*eye(nmu))*ym(1:nmu)';
	SpGPn = Kpp - Kpr(:,1:nmu)/(Krr(1:nmu,1:nmu) + sn^2*eye(nmu))*Krp(1:nmu,:);
	stdGPn = sqrt(diag(SpGPn));
	% We examine and store the results.
	mupS(:,1,counter) = mupGPn;
	SpS(:,:,1,counter) = SpGPn;
	res(1,:,counter) = [mean((mupGPn - yp').^2),mean(stdGPn.^2),mean(((mupGPn - yp')./stdGPn).^2)];

	% Method 2.
	% We now set up a GP of the noisy measurements, with tuned hyperparameters. First we tune the hyperparameters.
	tic;
	[sn,alpha,Lambda] = tuneHyperparameters(xm(1:nmu),ym(1:nmu)');
	disp(['GP hyperparameter tuning time is ',num2str(toc),' s. Parameters found were lx: ',num2str(sqrt(Lambda)),', ly: ',num2str(alpha),', sy: ',num2str(sn),'.']);
	param2 = [sn,alpha,Lambda];
	% We recalculate covariance matrices.
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
	% We make a GP prediction using the noisy measurement points.
	mupGPm = Kpm(:,1:nmu)/(Kmm(1:nmu,1:nmu) + sn^2*eye(nmu))*ym(1:nmu)';
	SpGPm = Kpp - Kpm(:,1:nmu)/(Kmm(1:nmu,1:nmu) + sn^2*eye(nmu))*Kmp(1:nmu,:);
	stdGPm = sqrt(diag(SpGPm));
	% We examine and store the results.
	mupS(:,2,counter) = mupGPm;
	SpS(:,:,2,counter) = SpGPm;
	res(2,:,counter) = [mean((mupGPm - yp').^2),mean(stdGPm.^2),mean(((mupGPm - yp')./stdGPm).^2)];

	% Method 3.
	% Next, we set up the FITC algorithm for the noisy measurements, using the hyperparameters which we just found. We use all nm measurements here.
	Lmm = diag(diag(Kmm + sn^2*eye(nm) - Kmu/Kuu*Kum));
	SuFITC = Kuu/(Kuu + Kum/Lmm*Kmu)*Kuu;
	muuFITC = SuFITC/Kuu*Kum/Lmm*ym';
	mupFITC = Kpu/Kuu*muuFITC;
	SpFITC = Kpp - Kpu/Kuu*(Kuu - SuFITC)/Kuu*Kup;
	stdUFITC = sqrt(diag(SpFITC));
	% We examine and store the results.
	muuS(:,3,counter) = muuFITC;
	SuS(:,:,3,counter) = SuFITC;
	mupS(:,3,counter) = mupFITC;
	SpS(:,:,3,counter) = SpFITC;
	res(3,:,counter) = [mean((mupFITC - yp').^2),mean(stdUFITC.^2),mean(((mupFITC - yp')./stdUFITC).^2)];
	
	% Method 4.
	% The next step is to train the NIGP algorithm. We start doing that now. To search more efficiently, we initialize the hyperparameters as the true parameters, but the algorithm won't converge
	% on this anyway. It'll find its own parameters. So no cheating here. Well, not much anyway.
	seard = log(realParams([1,3,4])');
	lsipn = log(realParams(2));
	tic;
	evalc('[model, nigp] = trainNIGP(permute(xm(:,1:nmu),[2,1]),permute(ym(:,1:nmu),[2,1]),-500,1,seard,lsipn);'); % We apply the NIGP training algorithm. We put this in an evalc function to suppress the output made by the NIGP algorithm.
	% We extract the derived settings.
	len = exp(model.seard(1,1));
	alpha = exp(model.seard(2,1));
	sn = exp(model.seard(3,1));
	sx = exp(model.lsipn);
	Lambda = len^2;
	Sx = sx^2;
	disp(['NIGP hyperparameter tuning time is ',num2str(toc),' s. Parameters found were lx: ',num2str(len),', sx: ',num2str(sx),', ly: ',num2str(alpha),', sy: ',num2str(sn),'.']);
	% We recalculate covariance matrices.
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
	mupNIGP = Kpm(:,1:nmu)/(Kmm(1:nmu,1:nmu) + sn^2*eye(nmu) + diag(model.dipK(1:nmu)))*ym(1:nmu)';
	SpNIGP = Kpp - Kpm(:,1:nmu)/(Kmm(1:nmu,1:nmu) + sn^2*eye(nmu) + diag(model.dipK(1:nmu)))*Kmp(1:nmu,:);
	stdpNIGP = sqrt(diag(SpNIGP));
	% We examine and store the results.
	mupS(:,4,counter) = mupNIGP;
	SpS(:,:,4,counter) = SpNIGP;
	res(4,:,counter) = [mean((mupNIGP - yp').^2),mean(stdpNIGP.^2),mean(((mupNIGP - yp')./stdpNIGP).^2)];

	% Method 5/6/7.
	% And now it's time for the SONIG algorithm, done in various ways.
	for method = 5:7
		% We first look at which measurements we use, as well as set up a SONIG object and give it the right inducing input points.
		if method == 5 || method == 6
			% We set which measurement points we will use.
			fromPoint = 1;
			if method == 5
				toPoint = nmu;
			else
				toPoint = nm;
			end
			% We set up a SONIG object with the right hyperparameters.
			hyp = NIGPModelToHyperparameters(model);
			sonig = createSONIG(hyp);
			sonig = addInducingInputPoint(sonig, xu);
		else
			% We set which measurement points we will use.
			fromPoint = nmn+1;
			toPoint = nm;
			% We apply NIGP training on the first set of measurements.
			seard = log(realParams([1,3,4])');
			lsipn = log(realParams(2));
			tic;
			evalc('[model, nigp] = trainNIGP(permute(xm(:,1:nmn),[2,1]),permute(ym(:,1:nmn),[2,1]),-500,1,seard,lsipn);'); % We apply the NIGP training algorithm. We put this in an evalc function to suppress the output made by the NIGP algorithm.
			% We extract the derived settings.
			len = exp(model.seard(1,1));
			alpha = exp(model.seard(2,1));
			sn = exp(model.seard(3,1));
			sx = exp(model.lsipn);
			Lambda = len^2;
			Sx = sx^2;
			% We recalculate covariance matrices.
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
			% We set up a SONIG object and give it the starting distribution given by the NIGP algorithm.
			hyp = NIGPModelToHyperparameters(model); % We use the hyperparameters just provided by the NIGP algorithm.
			sonig = createSONIG(hyp);
			sonig = addInducingInputPoint(sonig, xu);
			muu = Kum(:,1:nmn)/(Kmm(1:nmn,1:nmn) + sn^2*eye(nmn) + diag(model.dipK(1:nmn)))*ym(1:nmn)'; % This is the mean of the inducing input points, predicted by NIGP after nmn measurements.
			Su = Kuu - Kum(:,1:nmn)/(Kmm(1:nmn,1:nmn) + sn^2*eye(nmn) + diag(model.dipK(1:nmn)))*Kmu(1:nmn,:); % And this is the covariance matrix.
			sonig.fu{1} = createDistribution(muu, Su);
		end
		
		% And now we implement all the measurements into the SONIG object.
		for i = fromPoint:toPoint
			inputDist = createDistribution(xm(:,i), hyp.sx^2); % This is the prior distribution of the input point.
			outputDist = createDistribution(ym(:,i), hyp.sy^2); % This is the prior distribution of the output point.
			[sonig, inputPost, outputPost] = implementMeasurement(sonig, inputDist, outputDist); % We implement the measurement into the SONIG object.
		end
		[mupSONIG, SpSONIG, stdpSONIG] = makeSonigPrediction(sonig, xp); % Here we make the prediction.
		
		% We check if the resulting SONIG object is valid. If not, some problem has occurred.
		if sonig.valid == 0
			disp(['Problems occurred. Restarting loop ',num2str(counter),'.']);
			counter = counter - 1;
			continue;
		end
		
		% We examine and store the results.
		muuS(:,method,counter) = sonig.fu{1}.mean;
		SuS(:,:,method,counter) = sonig.fu{1}.cov;
		mupS(:,method,counter) = mupSONIG;
		SpS(:,:,method,counter) = SpSONIG;
		res(method,:,counter) = [mean((mupSONIG - yp').^2),mean(stdpSONIG.^2),mean(((mupSONIG - yp')./stdpSONIG).^2)];
	end
end

% Finally, we evaluate the results. For this, we get rid of the worst parts of the results of each algorithm.
disp('We are done! Results are as follows for the various methods. (Note that the order is different from the order in the paper.)');
disp('	MSE		Mean var.	Ratio	(The MSE and Mean var have been multiplied by 1000 for visibility.)');
result = mean(res(:,1:2,:),3);
disp([result*1e3,result(:,1)./result(:,2)]); % We show the results. We multiply the errors by a thousand to make the numbers more visible in Matlab.

% save('ComparisonScriptExperiments');

%% With this script, we can plot the result of a certain sample from the script above. We can also load in earlier data.

% load('ComparisonScriptExperiments');

% Which sample (or counter number) should we plot?
sample = 1;

% We extract the measurements and plot points for this case.
xm = xmS(:,sample);
xmr = xmrS(:,sample);
ym = ymS(:,sample);
ymr = ymrS(:,sample);
yp = ypS(:,sample);

% We plot the resulting function generated for that case, including the measurements that were done.
figure(11);
clf(11);
hold on;
grid on;
plot(xp, yp, 'b-');
plot(xmr(1:nmu), ymr(1:nmu), 'g+');
plot(xm(1:nmu), ym(1:nmu), 'ro');
title('Sample GP function');
xlabel('Input');
ylabel('Output');

% For each of the algorithms, we plot the results.
pointsUsed = [nmu,nmu,nm,nmu,nmu,nm,nm];
for i = 1:numMethods
	makeGPPlot(i, xp, mupS(:,i,sample), sqrt(diag(SpS(:,:,i,sample))));
	plot(xm(1:pointsUsed(i)), ym(1:pointsUsed(i)), 'ro');
	plot(xp,yp,'b-');
	title(['Case ',num2str(i)]);
	xlabel('Input');
	ylabel('Output');
	if muuS(1,i,sample) ~= 0
		plot(xu,muuS(:,i,sample),'ko');
	end
end