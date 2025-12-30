function [xfinal,X] = Simoncini_V2(M1,A1,H,A2,M,H3,A3,B,check_res)
% Generalized Simoncini direct solver for 3-mode tensor linear equations
% Works for non-cubic tensors: X \in R^{n1 x n2 x n3}.
% Arguments correspond to the Kronecker structure:
%   kron( kron(M1, A1), H ) + kron( kron(A2, M), H ) + kron( kron(H3, M), A3 )
% The mode-dimensions must be consistent:
%   - mode-1 matrices: H, A3 (size n1 x n1)
%   - mode-2 matrices: A1, M  (size n2 x n2)
%   - mode-3 matrices: M1, A2, H3 (size n3 x n3)
%
% B is a tensor of size [n1 x n2 x n3] given in a form understood by tens2mat
% (tens2mat(B,1) should be n1 x (n2*n3)).
%
% Note: requires MATLAB's Control System Toolbox (sylvester) and Tensorlab for tens2mat/tens2mat inverse.

if nargin < 9
    check_res = false;
end

% Basic dimension inference & checks
bb = tens2mat(B,1);           % n1 x (n2*n3)
[n1, n23] = size(bb);

n2 = size(A1,1);
n3 = size(M1,1);

% Sanity checks
if n2 * n3 ~= n23
    error('Inconsistent dimensions: size(B,1) = %d but expected n2*n3 = %d', n23, n2*n3);
end
if size(H,1) ~= size(A3,1)
    error('H and A3 must have the same dimension (mode-1).');
end
if size(M,1) ~= n2 || size(A1,1) ~= n2
    error('A1 and M must be n2 x n2.');
end
if size(M1,1) ~= n3 || size(A2,1) ~= n3 || size(H3,1) ~= n3
    error('M1, A2, H3 must be n3 x n3.');
end
if size(H,1) ~= n1 || size(A3,1) ~= n1
    error('H and A3 must be n1 x n1.');
end

% Schur decomposition on (A3'/H')
[Q,R] = schur(full(A3' / H'));   % sizes are n1 x n1

% Build right-hand block matrix g as in paper
g = (bb' / H') * Q;  % (n2*n3) x n1
[~, ng] = size(g);
% Expect ng == n1 (number of columns equal to mode-1 dim)
if ng ~= n1
    warning('Unexpected number of columns in g (%d); expected n1 = %d', ng, n1);
end

% Preallocate storage for intermediate vecs
zz = zeros(n2 * n3, ng);

for k = 1:ng
    % reshape column k into n2 x n3 matrix (matches mode-2 x mode-3)
    Gk = reshape(g(:,k), n2, n3);   % g_k = vec(G_k)
    % F = M^{-1} * Gk * M1^{-T}  (right division by M1' is equivalent)
    F = M \ Gk / M1';
    if k > 1
        % accumulate contributions from previous z's via upper triangular part of R
        % zz(:,1:k-1) contains previous vec(Zhat) with each column length n2*n3,
        % multiply by R(1:k-1,k) (a column vector) to get vector of length n2*n3,
        % then reshape into (n2 x n3).
        W = reshape(zz(:,1:k-1) * R(1:k-1,k), n2, n3);
        % subtract contribution: note original derivation has W*H3' / M1'
        F = F - (W * H3') / M1';
    end
    % Build Sylvester arguments:
    % left operator: M \ A1  (n2 x n2)
    % right operator: (R(k,k)*H3' + A2') / M1'  (n3 x n3)
    Aleft = M \ A1;
    Aright = (R(k,k) * H3' + A2') / M1';
    % Solve Aleft * Z + Z * Aright = F  for Z of size n2 x n3
    Zhat = sylvesterf(Aleft, Aright, F);
    zz(:,k) = reshape(Zhat, n2 * n3, 1);
end

% Reconstruct solution
xx = Q * zz';            % n1 x (n2*n3)
xfinal = real(xx(:));    % vectorized solution
X = reshape(real(xx), n1, n2, n3);


if check_res
    % construct full Kronecker operator L only if requested (can be huge!)
    % L = kron(kron(M1,A1),H) + kron(kron(A2,M),H) + kron(kron(H3,M),A3);
    % vecB = bb(:);
    % normres = norm(L * xfinal - vecB) / norm(vecB);
    % fprintf('final full relative residual: %g\n', normres);
    
end
end
