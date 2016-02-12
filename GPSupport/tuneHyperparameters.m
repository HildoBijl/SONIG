function [sn, alpha, Lambda, logp] = tuneHyperparameters(xm, ym)
%tuneHyperparameters This function tunes hyperparameters on a given data set using a simple gradient ascent algorithm.
% The tuneHyperparameters function will tune the hyperparameters of a regular Gaussian process based on an input set xm and an output set ym. Inputs are:
%	xm: The set of input points. This should be a matrix of size d by n, with d the input vector size and n the number of measurement points.
%	ym: The set of output values. This should be a vector of size n (by 1).
% The output will consist of the tuned hyperparameters and the log-likelihood.
%	sn: The standard deviation of the output noise.
%	alpha: The output length scale.
%	Lambda: The squared input length scale, as a vector. This is because we assume Lambda is a diagonal matrix, so we only keep track of its diagonal entries. To get the actual matrix, use diag(Lambda).
%	logp: The log-likelihood of the measurements, with the given hyperparameters.

% We derive some dimensions.
dx = size(xm,1); % This is the input point dimension.
nm = size(xm,2); % This is the number of measurement points.

% We check the size of ym.
if size(ym,2) ~= 1
	error('The tuneHyperparameters function was called with invalid input. The ym parameter should be a column vector. However, the given ym parameter did not have length 1 in the horizontal direction.');
end
if size(ym,1) ~= nm
	error(['The tuneHyperparameters function was called with invalid input. According to the xm parameter, there are ',num2str(nm),' input points, but the ym vector does not have this height.']);
end

% We set up some settings.
numSteps = 40; % This is the number of steps we will do in our gradient ascent.
stepSize = 1; % This is the initial (normalized) step size.
stepSizeFactor = 2; % This is the factor by which we will adjust the step size, in case it turns out to be inadequate.

% We define the preliminary hyperparameters.
sn = 0.1; % This is the initial value of sn.
alpha = 1; % This is the initial value of sf.
Lambda = ones(dx,1); % This is the initial value of Lambda.

% We now set up the parameter array and immediately calculate its derivative.
param = [sn^2;alpha^2;Lambda]; % This is the initial value of the parameter array.
paramDeriv = zeros(size(param)); % We create storage space for the derivative.
newParamDeriv = zeros(size(param)); % We create storage space for the new derivative. (This will be used in the loop.)
diff = repmat(permute(xm,[2,3,1]),[1,nm]) - repmat(permute(xm,[3,2,1]),[nm,1]); % We calculate the difference matrix for the input points.
Kmm = alpha^2*exp(-1/2*sum(diff.^2./repmat(permute(Lambda,[2,3,1]),[nm,nm,1]),3)); % We calculate the covariance matrix for the input points.
P = Kmm + sn^2*eye(nm); % This is a matrix which we will be needing a few times.
logp = -1/2*ym'/P*ym - 1/2*logdet(P); % We calculate the log-likelihood.
beta = P\ym; % This is a supporting parameter.
R = beta*beta' - inv(P); % This is another supporting parameter.
paramDeriv(1) = 1/2*trace(R); % This is the derivative of logp with respect to sn.
paramDeriv(2) = 1/(2*alpha^2)*trace(R*Kmm); % This is the derivative of logp with respect to alpha.
for j = 1:length(Lambda)
	paramDeriv(2+j) = 1/(2*alpha^2)*trace(R*(Kmm.*diff(:,:,j).^2))/(2*Lambda(j)^2); % This is the derivative of logp with respect to element j of the Lambda matrix.
end

% Now it's time to start iterating.
for i = 1:numSteps
	% We try to improve the parameters, all the while checking the step size.
	maxReductions = 100;
	for j = 1:maxReductions
		if j == maxReductions
			disp('Error: something is wrong with the step size in the hyperparameter optimization scheme.');
		end
		% We use the derivative of the hyperparameters to calculate a new possible value for the vector of hyperparameters.
		newParam = param.*(1 + stepSize*param.*paramDeriv);
		if min(newParam > 0) % Are all hyperparameters still positive? If not, there's a problem.
			% We extract the new values of the hyperparameters and calculate the new value of logp.
			sn = sqrt(newParam(1));
			alpha = sqrt(newParam(2));
			Lambda = newParam(3:end);
			Kmm = alpha^2*exp(-1/2*sum(diff.^2./repmat(permute(Lambda,[2,3,1]),[nm,nm,1]),3));
			P = Kmm + sn^2*eye(nm);
			newLogp = -1/2*ym'/P*ym - 1/2*logdet(P);
			% We check if the new value of logp is better than the previous value. If not, there's a problem.
			if newLogp >= logp
				% We have a better point! Let's jump to this point and recalculate the derivative of logp with respect to the hyperparameters from here.
				beta = P\ym;
				R = beta*beta' - inv(P);
				newParamDeriv(1) = 1/2*trace(R);
				newParamDeriv(2) = 1/(2*alpha^2)*trace(R*Kmm);
				for j = 1:length(Lambda)
					newParamDeriv(2+j) = 1/(2*alpha^2)*trace(R*(Kmm.*diff(:,:,j).^2))/(2*Lambda(j)^2);
				end
				% We compare the new derivative with respect to the previous one to update the step size. If the derivatives point in the same direction, we can increase the step size. If they
				% point in opposite directions, we should decrease it. That's what's done here.
				directionConsistency = (paramDeriv'*newParamDeriv)/norm(paramDeriv)/norm(newParamDeriv);
				stepSize = stepSize*stepSizeFactor^directionConsistency;
				break;
			end
		end
		% If there was a problem, then this was caused by a too large step size. So we reduce the step size by the appropriate factor.
		stepSize = stepSize/stepSizeFactor;
	end
	% We update the hyperparameters (and related parameters) to the new values.
	param = newParam;
	paramDeriv = newParamDeriv;
	logp = newLogp;
end

end

