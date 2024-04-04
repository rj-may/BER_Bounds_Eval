function kde = kdeGivenBW(X, h, smoothness, params)
% Implements Kernel Density Estimator with kernels of order floor(smoothness)
% for the given bandwidth. You should cross validate h externally.
% Inputs
%   X: the nxd data matrix
%   h: bandwidth
%   smoothness: If using a Gaussian Kernel this should be 'gaussian'. Otherwise
%    specify the order of the legendre polynomial kernel. 
% Outputs
%   kde: a function handle to estimate the density. kde takes in N points in a
%     Nxd matrix and outputs an Nx1 vector.

  % prelims
  numDims = size(X, 2);
  numPts = size(X, 1);

  if ~exist('params', 'var')
    params = struct;
  end
  if ~isfield(params, 'doBoundaryCorrection')
    params.doBoundaryCorrection = true;
  end
  if ~isfield(params, 'estLowerBound')
    params.estLowerBound = 0;
  end
  if ~isfield(params, 'estUpperBound')
    params.estUpperBound = Inf;
  end

  if ~params.doBoundaryCorrection
    augX = X;
  else
    % First augment the dataset by mirroring the points close to the boundaries.
    augX = zeros(0, numDims);
    % Our augmented space as 3^d regions. The centre region is the actual space
    % but all others are in the boundary. We iterate through them as follows
    for regionIdx = 0:(3^numDims -1)

      dimRegions = dec2base(regionIdx, 3);
      dimRegions = [repmat('0', 1, numDims - numel(dimRegions)) dimRegions];
      % Now dimRegions is a string of dimRegions characters with each character
      % corrsponding to each dimension. If the character is 0, we look on the
      % lower boundary of the dimension and if 2 we look at the higher boundary
      % of the dimension.

      % Now check for points within h of the bounary and add them to the dataset
      toReplicate = ones(numPts, 1);
      replicX = X;
      for d = 1:numDims
        if dimRegions(d) == '0'
          replicX(:,d) = -replicX(:,d);
          toReplicate = toReplicate .* double( X(:,d) < h );
        elseif dimRegions(d) == '2'
          replicX(:,d) = 2 - replicX(:,d);
          toReplicate = toReplicate .* double( 1 - X(:,d) < h ); 
        end
      end
      replicatedPts = replicX( logical(toReplicate), :);
      augX = [augX; replicatedPts];
      % Note that when dimRegions = '11...1', we will add the original X to augX

    end
    numAugPts = size(augX, 1);
%     fprintf('numPts = %d, numAugPts = %d\n', numPts, numAugPts);
  end % ~params.doBoundaryCorrection

  % Now return the function handle
  kde = @(arg) kdeIterative(arg, augX, h, smoothness, params, numPts);
end


% A function which estimates the KDE at pts. We use this to construct the
% function handle which will be returned.
function ests = kdeIterative(pts, augX, h, smoothness, params, numX)

  numPts = size(pts, 1);
  numData = size(augX, 1);
  maxNumPts = max(1e7, numData);
  ptsPerPartition = min( numPts, ceil(maxNumPts/numData) );

  ests = zeros(numPts, 1);
  % Now iterate through each 'partition' and obtain the relevant kernels
  cumNumPts = 0;
  while cumNumPts < numPts
    currNumPts = min(ptsPerPartition, numPts - cumNumPts);
    if isstr(smoothness) & strcmp(lower(smoothness(1:5)), 'gauss')
      K = kdeGaussKernel(pts(cumNumPts+1: cumNumPts+currNumPts, :), augX, h);
      % K
    else
      K = kdeLegendreKernel( pts(cumNumPts+1: cumNumPts+currNumPts, :), ...
            augX, h, smoothness);
    end
    ests(cumNumPts+1 : cumNumPts + currNumPts) = sum(K,2)/numX;
    cumNumPts = cumNumPts + currNumPts;
  end

  % Now truncate those values below and above the bounds
  ests = max(ests, params.estLowerBound);
  ests = min(ests, params.estUpperBound);
end


function K = kdeLegendreKernel(X, C, h, order)
% Returns the value of the kernel evaluated at the points X centred at C and
% with bandwidth h.
% Inputs
% X : nxd data matrix
% C : mxd centre matrix. If empty is initialized to zero(1, d)
% h : the bandwidth of the kernel
% order : order of the kernel
% Ouputs
% K : The nxm kernel matrix where K(i,j) = k(X(i,:), C(j,:))
%%% Warning: make sure mxn < 1e6 to avoid crashing
  
  % Prelims
  numDims = size(X, 2);

  if isempty(C)
    C = zeros(1, numDims);
  end

  numData = size(X, 1);
  numCentres = size(C, 1);

  K = ones(numData, numCentres);
  for d = 1:numDims
    K = K .* kernel1D( X(:, d), C(:, d), h, order);
  end
end

% 1 Dimensional Kernel. The d-dimensional kernel is the product kernel.
function ret = kernel1D(x, c, h, order)
% Same as above but now x and c are 1 dimensional (d=1)

  numCentres = size(c, 1);
  % u is a numData x numCentres matrix, u_ij = (x_i - c_j)/h
  u = bsxfun(@minus, repmat(x, 1, numCentres), c')/h;

  ret = zeros(size(u));
  for m = 0:2:order
    % only need to iterate through even m since legPoly(0,m) = 0 for m odd
    ret = ret + legPoly(0, m) * legPoly(u, m);
  end
  % Finally check if u is within the domain of the kernel and divide by h.
  ret = ret .* double(abs(u) < 1)/h ;
end


function K = kdeGaussKernel(X, Y, h)
% Returns the Kernel Matrix for a Gaussian Kernel of bandwidth h.
% X is an nxd matrix. K is an nxn matrix.
% If Y is nonempty then returns the gaussian kernel for XxY
% h is a column vector of size the dimension of the space - i.e size(X, 1).

  % Prelims
  d = size(X, 2); % dimensions
  if size(h, 1) == 1, h = h'; % if you get a row vector
  end
  if isscalar(h)
    h = h * ones(d, 1);
  end

  if ~exist('Y', 'var') | isempty(Y)
    Y = X;
  end

  D2 = distSquared(X, Y, h);
  K = 1/(sqrt(2*pi)^d * prod(h)) * exp(-D2/2);

end

