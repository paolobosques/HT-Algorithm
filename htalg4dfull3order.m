%---------------------
% Date: 12/22/2025
% Author: Paolo Bosques-Paulet
% HT Algorithm for 4D diffusion eqn w/ 3rd order accuracy
%---------------------

close all;clear variables; 


% Params and EQ Def

% ut = d1* ux1x1 + d2 * ux2x2 + d3* ux3x3
d1 = 1/10;
d2 = 1/10;
d3 = 1/10;
d4 = 1/10;

% Number of discrete cells per dimension
N1 = 40;
N2 = 50;
N3 = 40;
N4 = 60;

tf = .5; %15;
L = 2*pi; %14;

% Cell centered grid in space ---

x1vals = linspace(-L/2,L/2,N1+1);
dx1 = x1vals(2)-x1vals(1);
x1vals = x1vals(1:end-1) + dx1/2;

x2vals = linspace(-L/2,L/2,N2+1);
dx2 = x2vals(2)-x2vals(1);
x2vals = x2vals(1:end-1) + dx2/2;

x3vals = linspace(-L/2,L/2,N3+1);
dx3 = x3vals(2)-x3vals(1);
x3vals = x3vals(1:end-1) + dx3/2;

x4vals = linspace(-L/2,L/2,N4+1);
dx4 = x4vals(2)-x4vals(1);
x4vals = x4vals(1:end-1) + dx4/2;

% ---------------------------------

[X1,X2,X3,X4] = ndgrid(x1vals,x2vals,x3vals,x4vals);


%initial
%u0 = 0.8*exp(-15*((X1-6.5).^2+(X2-6.5).^2+(X3-6.5).^2))+0.5*exp(-15*((X1-7.5).^2+(X2-7.5).^2+(X3-7.5).^2));

%u0 = 0.8*exp(-15*(X1-1)).^2+0.5*exp(-15*(X1+1).^2);

%Accuracy Test (Rank1)
% u0 = sin(2*pi/L*X1).*sin(2*pi/L*X2).*sin(2*pi/L*X3).*sin(2*pi/L*X4);
% uexact =  exp(-4*pi^2/L^2 * tf *(d1+d2+d3+d4)).*sin(2*pi/L*X1).*sin(2*pi/L*X2).*sin(2*pi/L*X3).*sin(2*pi/L*X4);

%Accuracy Test (Rank2)
% u0 = sin(2*pi/L*X1).*sin(2*pi/L*X2).*sin(2*pi/L*X3).*sin(2*pi/L*X4)+...
%     sin(2*2*pi/L*X1).*sin(2*2*pi/L*X2).*sin(2*2*pi/L*X3).*sin(2*2*pi/L*X4);
% 
% uexact = exp(-4*pi^2/L^2 * tf *(d1+d2+d3+d4)).*sin(2*pi/L*X1).*sin(2*pi/L*X2).*sin(2*pi/L*X3).*sin(2*pi/L*X4)+...
%     exp(-4*4*pi^2/L^2 * tf *(d1+d2+d3+d4)).*sin(2*2*pi/L*X1).*sin(2*2*pi/L*X2).*sin(2*2*pi/L*X3).*sin(2*2*pi/L*X4);

%Accuracy Test (Rank 3)
u0 = sin(2*pi/L*X1).*sin(2*pi/L*X2).*sin(2*pi/L*X3).*sin(2*pi/L*X4)+...
    sin(2*2*pi/L*X1).*sin(2*2*pi/L*X2).*sin(2*2*pi/L*X3).*sin(2*2*pi/L*X4)+...
    sin(2*3*pi/L*X1).*sin(2*3*pi/L*X2).*sin(2*3*pi/L*X3).*sin(2*3*pi/L*X4);
uexact = exp(-4*pi^2/L^2 * tf *(d1+d2+d3+d4)).*sin(2*pi/L*X1).*sin(2*pi/L*X2).*sin(2*pi/L*X3).*sin(2*pi/L*X4)+...
    exp(-4*4*pi^2/L^2 * tf *(d1+d2+d3+d4)).*sin(2*2*pi/L*X1).*sin(2*2*pi/L*X2).*sin(2*2*pi/L*X3).*sin(2*2*pi/L*X4)+...
    exp(-4*9*pi^2/L^2 * tf *(d1+d2+d3+d4)).*sin(2*3*pi/L*X1).*sin(2*3*pi/L*X2).*sin(2*3*pi/L*X3).*sin(2*3*pi/L*X4);

%SPECTRAL METHODS

% FOR CELL CENTERED GRID: SUBTRACTED EXTRA 1 FROM EACH NX ABOVE
F1 = (2*pi/L)^2*toeplitz([-1/(3*(2*dx1/L)^2)-1/6 ...
    .5*(-1).^(2:N1)./sin((2*pi*dx1/L)*(1:N1-1)/2).^2]);
F1 = d1*F1;

F2 = (2*pi/L)^2*toeplitz([-1/(3*(2*dx2/L)^2)-1/6 ...
    .5*(-1).^(2:N2)./sin((2*pi*dx2/L)*(1:N2-1)/2).^2]);
F2 = d2*F2;

F3 = (2*pi/L)^2*toeplitz([-1/(3*(2*dx3/L)^2)-1/6 ...
    .5*(-1).^(2:N3)./sin((2*pi*dx3/L)*(1:N3-1)/2).^2]);
F3 = d3*F3;

F4 = (2*pi/L)^2*toeplitz([-1/(3*(2*dx4/L)^2)-1/6 ...
    .5*(-1).^(2:N4)./sin((2*pi*dx4/L)*(1:N4-1)/2).^2]);
F4 = d4*F4;


tol = 1e-6; 

%lambdavec = .1;

%Butcher tableau parameters
nu = .435866521508459; %Often Gamma
B1 = -1.5*nu^2 + 4*nu - 0.25;
B2 =  1.5*nu^2 - 5*nu + 1.25;

% lambdavec = 0.1:0.05:3;
lambdavec = 0.5:.05:3;
dtvec = zeros(size(lambdavec));
L1error = zeros(size(lambdavec));

for p = 1:length(lambdavec)
 %HTucker Decomp u0
    opts.max_rank = min([N1,N2,N3,N4]);  % max rank at each node   
    opts.rel_eps = tol; % max relative err. ||X - Xnew|| / ||X|| < rel_eps
    opts.abs_eps = tol; % max abs err, ||X - Xnew|| < abs_eps
    ht_u0 = htensor.truncate_ltr(u0,opts); 
    
    B1234_n = ht_u0.B{1};
    B12_n = ht_u0.B{2};
    B34_n = ht_u0.B{3};
    U1_n = ht_u0.U{4};
    U2_n = ht_u0.U{5};  
    U3_n = ht_u0.U{6};
    U4_n = ht_u0.U{7};

    B12matcT_n = tens2mat(B12_n,3)';
    B34matcT_n = tens2mat(B34_n,3)';
    
    U1star_k1 = U1_n;
    U2star_k1 = U2_n;
    U3star_k1 = U3_n;  
    U4star_k1 = U4_n;
    B12star_k1 = B12_n;
    B34star_k1 = B34_n;
    B12starmatcT_k1 = tens2mat(B12star_k1,3)';
    B34starmatcT_k1 = tens2mat(B34star_k1,3)';


    [~,r1_n] = size(U1_n);
    [~,r2_n] = size(U2_n);
    [~,r3_n] = size(U3_n);
    [~,r4_n] = size(U4_n);

    r1star_k1 = r1_n;
    r2star_k1 = r2_n;
    r3star_k1 = r3_n;
    r4star_k1 = r4_n;
    [r12_n,r34_n] = size(B1234_n);
    r12star_k1 = r12_n;
    r34star_k1 = r34_n;
    
    Btil12mat_n = tens2mat(B12_n,3)'* B1234_n ;
    Btil34mat_n = tens2mat(B34_n,3)'* B1234_n' ;

    Btil12_n = reshape(Btil12mat_n,[r1_n,r2_n,r34_n]) ;
    Btil34_n = reshape(Btil34mat_n,[r3_n,r4_n,r12_n]) ;

    disp(strcat("Lambda = ", num2str(lambdavec(p))))

    dt = lambdavec(p)/(1/dx1+1/dx2+1/dx3+1/dx4);
    dtvec(p) = dt;

    %dt = .5*dx;
    tvals = 0:dt:tf;
    rankvec1 = zeros(numel(tvals),3);
    % three entries for each dimension in each
    % timestep

    if tvals(end) ~= tf
        tvals = [tvals,tf]; %Ensure final t is at tf
    end

    rankvec = zeros(numel(tvals),4);
    rankvec(1,:) = [r1_n,r2_n,r3_n,r4_n];
    Nt = numel(tvals);

   
    mvec = zeros(1,length(2:Nt));
   
    for n = 2:Nt 
       
        disp(tvals(n))
        dtn = tvals(n)-tvals(n-1);
        %% Step 1      
        
        [K1_k1,K2_k1,K3_k1,K4_k1] = Kup4Dfull(dtn*nu, N1,N2,N3,N4,r1star_k1,r2star_k1,r3star_k1,r4star_k1,r12star_k1,r34star_k1,F1,F2,F3,F4,U1_n,U2_n,U3_n,U4_n,U1star_k1,U2star_k1,U3star_k1,U4star_k1,B12_n,B34_n,B12starmatcT_k1,B34starmatcT_k1,Btil12_n,Btil34_n);
        [U1hat_k1,U2hat_k1,U3hat_k1,U4hat_k1,r1hat_k1,r2hat_k1,r3hat_k1,r4hat_k1] = redaugk4(K1_k1,K2_k1,K3_k1,K4_k1,U1_n,U2_n,U3_n,U4_n);

        [r12hat_k1,B12hatmat_k1] = Bupfull(dtn*nu,r1hat_k1,r2hat_k1,r34star_k1,r3star_k1,r4star_k1,F1,F2,F3,F4,U1_n,U2_n,U3_n,U4_n,U1hat_k1,U2hat_k1, U3star_k1, U4star_k1, B34_n,B34starmatcT_k1,Btil12_n);
        [r34hat_k1,B34hatmat_k1] = Bupfull(dtn*nu,r3hat_k1,r4hat_k1,r12star_k1,r1star_k1,r2star_k1,F3,F4,F1,F2,U3_n,U4_n,U1_n,U2_n,U3hat_k1,U4hat_k1, U1star_k1, U2star_k1, B12_n,B12starmatcT_k1,Btil34_n);
                           
 
        B12hat_k1 = reshape(B12hatmat_k1, [r1hat_k1,r2hat_k1,r12hat_k1]);
        B34hat_k1 = reshape(B34hatmat_k1, [r3hat_k1,r4hat_k1,r34hat_k1]);


        % B1234 equation

        A =  speye(r34hat_k1) - (B12hatmat_k1'*kron(dtn*nu*speye(r2hat_k1),U1hat_k1'*(F1*U1hat_k1))*B12hatmat_k1+...
                                 B12hatmat_k1'*kron(dtn*nu*U2hat_k1'*(F2*U2hat_k1),speye(r1hat_k1))*B12hatmat_k1);

        B = -dtn*nu*B34hatmat_k1'*kron(speye(r4hat_k1),((F3*U3hat_k1)'*U3hat_k1))*B34hatmat_k1 ...
            -dtn*nu*B34hatmat_k1'*kron(((F4*U4hat_k1)'*U4hat_k1),speye(r3hat_k1))*B34hatmat_k1;

        B1234hat_k1 = sylvesterf(A, B ,...
            B12hatmat_k1'*kron(U2hat_k1'*U2_n,U1hat_k1'*U1_n)*B12matcT_n*B1234_n*B34matcT_n'*kron(U4_n'*U4hat_k1,U3_n'*U3hat_k1)*B34hatmat_k1 );
        
        %TRUNCATE 
        Uhats_1 = {[],[],[],full(U1hat_k1),full(U2hat_k1),full(U3hat_k1),full(U4hat_k1)};
        Bhats_1 = {full(B1234hat_k1),full(B12hat_k1),full(B34hat_k1),[],[],[],[]};
    
        ht_u0 = htensor(ht_u0.children,ht_u0.dim2ind, Uhats_1, Bhats_1);
        opts.max_rank = min([N1,N2,N3,N4]);
        opts.rel_eps = tol;
        opts.abs_eps = tol;
        ht_u0 = truncate_nonorthog(ht_u0,opts);

        % k1 vals
        B1234_k1 = ht_u0.B{1}; 
        B12_k1 = ht_u0.B{2}; 
        B34_k1 = ht_u0.B{3};
        U1_k1 = ht_u0.U{4};
        U2_k1 = ht_u0.U{5};
        U3_k1 = ht_u0.U{6};
        U4_k1 = ht_u0.U{7};

        B12matcT_k1 = tens2mat(B12_k1,3)';
        B34matcT_k1 = tens2mat(B34_k1,3)';
        
        [~,r1_k1] = size(U1_k1);
        [~,r2_k1] = size(U2_k1);
        [~,r3_k1] = size(U3_k1);
        [~,r4_k1] = size(U4_k1);
        [r12_k1,r34_k1] = size(B1234_k1);

        Btil12mat_k1 = tens2mat(B12_k1,3)'* B1234_k1 ;
        Btil34mat_k1 = tens2mat(B34_k1,3)'* B1234_k1'; %' wuz here
        
        Btil12_k1 = reshape(Btil12mat_k1,[r1_k1,r2_k1,r34_k1]);
        Btil34_k1 = reshape(Btil34mat_k1,[r3_k1,r4_k1,r12_k1]);

        %% Step 2

        % %Stage 2 sol(t(k=2)) = sol(tn)+dt*(1-nu)*D(sol(t(k=1)) + dt*(nu)*sol(t(k=2))
    
        %% HTN
            Uns = {[],[],[],full(U1_n),full(U2_n),full(U3_n),full(U4_n)};
            Bns = {full(B1234_n),full(B12_n),full(B34_n),[],[],[],[]};
        
            htn = htensor(ht_u0.children,ht_u0.dim2ind, Uns, Bns);
        %% HTK1
            Uk1s = {[],[],[],full(U1_k1),full(U2_k1),full(U3_k1),full(U4_k1)};
            Bk1s = {full(B1234_k1),full(B12_k1),full(B34_k1),[],[],[],[]};
        
            htk1 = htensor(ht_u0.children,ht_u0.dim2ind, Uk1s, Bk1s);

        %% HTK2 PREDICTION WITH BEULER
    
        [K1_k2_ddag,K2_k2_ddag,K3_k2_ddag,K4_k2_ddag] = Kup4Dfull(dtn*(1+nu)/2, N1,N2,N3,N4,r1_n,r2_n,r3_n,r4_n,r12_n,r34_n,F1,F2,F3,F4,U1_n,U2_n,U3_n,U4_n,U1_n,U2_n,U3_n,U4_n,B12_n,B34_n,B12matcT_n,B34matcT_n,Btil12_n,Btil34_n);
        [U1hat_k2_ddag,U2hat_k2_ddag,U3hat_k2_ddag,U4hat_k2_ddag,r1hat_k2_ddag,r2hat_k2_ddag,r3hat_k2_ddag,r4hat_k2_ddag] = redaugk4(K1_k2_ddag,K2_k2_ddag,K3_k2_ddag,K4_k2_ddag,U1_n,U2_n,U3_n,U4_n);
        
        [r12hat_k2_ddag,B12hatmat_k2_ddag] = Bupfull(dtn*(1+nu)/2,r1hat_k2_ddag,r2hat_k2_ddag,r34_n,r3_n,r4_n,F1,F2,F3,F4,U1_n,U2_n,U3_n,U4_n,U1hat_k2_ddag,U2hat_k2_ddag, U3_n, U4_n, B34_n,B34matcT_n,Btil12_n);
        [r34hat_k2_ddag,B34hatmat_k2_ddag] = Bupfull(dtn*(1+nu)/2,r3hat_k2_ddag,r4hat_k2_ddag,r12_n,r1_n,r2_n,F3,F4,F1,F2,U3_n,U4_n,U1_n,U2_n,U3hat_k2_ddag,U4hat_k2_ddag, U1_n, U2_n, B12_n,B12matcT_n,Btil34_n);              
 
        B12hat_k2_ddag = reshape(B12hatmat_k2_ddag, [r1hat_k2_ddag,r2hat_k2_ddag,r12hat_k2_ddag]);
        B34hat_k2_ddag = reshape(B34hatmat_k2_ddag, [r3hat_k2_ddag,r4hat_k2_ddag,r34hat_k2_ddag]);


        % B1234 equation

        A =  speye(r34hat_k2_ddag) - (B12hatmat_k2_ddag'*kron(dtn*(1+nu)/2*speye(r2hat_k2_ddag),U1hat_k2_ddag'*(F1*U1hat_k2_ddag))*B12hatmat_k2_ddag+...
                                 B12hatmat_k2_ddag'*kron(dtn*(1+nu)/2*U2hat_k2_ddag'*(F2*U2hat_k2_ddag),speye(r1hat_k2_ddag))*B12hatmat_k2_ddag);

        B = -dtn*(1+nu)/2*B34hatmat_k2_ddag'*kron(speye(r4hat_k2_ddag),((F3*U3hat_k2_ddag)'*U3hat_k2_ddag))*B34hatmat_k2_ddag ...
            -dtn*(1+nu)/2*B34hatmat_k2_ddag'*kron(((F4*U4hat_k2_ddag)'*U4hat_k2_ddag),speye(r3hat_k2_ddag))*B34hatmat_k2_ddag;

        B1234hat_k1 = sylvesterf(A, B ,...
            B12hatmat_k2_ddag'*kron(U2hat_k2_ddag'*U2_n,U1hat_k2_ddag'*U1_n)*B12matcT_n*B1234_n*B34matcT_n'*kron(U4_n'*U4hat_k2_ddag,U3_n'*U3hat_k2_ddag)*B34hatmat_k2_ddag );
        
       
        Uk2s_ddag = {[],[],[],full(U1hat_k1),full(U2hat_k1),full(U3hat_k1),full(U4hat_k1)};
        Bk2s_ddag = {full(B1234hat_k1),full(B12hat_k1),full(B34hat_k1),[],[],[],[]};

        htk2_ddag = htensor(ht_u0.children,ht_u0.dim2ind,Uk2s_ddag,Bk2s_ddag);

        %% Get Stars
        
        htaug1 = plus(plus(htn,htk1),htk2_ddag);
        htaug1 = truncate_nonorthog(htaug1,opts);

        B1234star_k2 = htaug1.B{1}; 
        B12star_k2 = htaug1.B{2}; 
        B34star_k2 = htaug1.B{3};
        U1star_k2 = htaug1.U{4};
        U2star_k2 = htaug1.U{5};
        U3star_k2 = htaug1.U{6};
        U4star_k2 = htaug1.U{7};

        B12starmatcT_k2 = tens2mat(B12star_k2,3)';
        B34starmatcT_k2 = tens2mat(B34star_k2,3)';

        [~,r1star_k2] = size(U1star_k2);
        [~,r2star_k2] = size(U2star_k2);
        [~,r3star_k2] = size(U3star_k2);
        [~,r4star_k2] = size(U4star_k2);
        [r12star_k2,r34star_k2] = size(B1234star_k2);


        disp([r1_k1,r2_k1,r3_k1,r4_k1])
    
        %% Stage 2
        %sol(tn)+dt*nu*D(sol(t(k=1)) + dt*(1-nu)/2*Dsol(t(k=2)) 
        % K steps (2)

        K1_k2 = sylvesterf( (speye(N1)-dtn*nu*F1), ...
                    -dtn*nu*( kron( B34starmatcT_k2' * kron( speye(r4star_k2), U3star_k2'*F3'*U3star_k2 )*B34starmatcT_k2,speye(r2star_k2) ) ...
                            + kron( B34starmatcT_k2' * kron( U4star_k2'*F4'*U4star_k2, speye(r3star_k2) )*B34starmatcT_k2,speye(r2star_k2) )...
                            + kron( B34starmatcT_k2' * kron( speye(r4star_k2), speye(r3star_k2)         )*B34starmatcT_k2,(F2*U2star_k2)' * U2star_k2 )), ...
                   U1_n * tens2mat(Btil12_n,1) * kron( B34matcT_n' * kron(U4_n'*U4star_k2, U3_n'*U3star_k2)*B34starmatcT_k2, U2_n'*U2star_k2) ...
                   + dtn*(1-nu)/2 * (F1* U1_k1 * tens2mat(Btil12_k1,1) * kron( B34matcT_k1'*kron(U4_k1'*U4star_k2,U3_k1'*U3star_k2)     *B34starmatcT_k2,     U2_k1'*U2star_k2 ) ...
                                       + U1_k1 * tens2mat(Btil12_k1,1) * kron( B34matcT_k1'*kron(U4_k1'*U4star_k2,(F3*U3_k1)'*U3star_k2)*B34starmatcT_k2,     U2_k1'*U2star_k2 )...
                                       + U1_k1 * tens2mat(Btil12_k1,1) * kron( B34matcT_k1'*kron((F4*U4_k1)'*U4star_k2,U3_k1'*U3star_k2)*B34starmatcT_k2,     U2_k1'*U2star_k2 )...
                                       + U1_k1 * tens2mat(Btil12_k1,1) * kron( B34matcT_k1'*kron(U4_k1'*U4star_k2,     U3_k1'*U3star_k2)*B34starmatcT_k2,(F2*U2_k1)'*U2star_k2 )) );

       K2_k2 = sylvesterf( (speye(N2)-dtn*nu*F2), ...
                    -dtn*nu*( kron( B34starmatcT_k2' * kron( speye(r4star_k2), U3star_k2'*F3'*U3star_k2 )*B34starmatcT_k2,speye(r1star_k2) ) ...
                            + kron( B34starmatcT_k2' * kron( U4star_k2'*F4'*U4star_k2, speye(r3star_k2) )*B34starmatcT_k2,speye(r1star_k2) )...
                            + kron( B34starmatcT_k2' * kron( speye(r4star_k2), speye(r3star_k2)         )*B34starmatcT_k2,(F1*U1star_k2)' * U1star_k2 )), ...
                   U2_n * tens2mat(Btil12_n,2) * kron( B34matcT_n' * kron(U4_n'*U4star_k2, U3_n'*U3star_k2)*B34starmatcT_k2, U1_n'*U1star_k2) ...
                   + dtn*(1-nu)/2 * (F2* U2_k1 * tens2mat(Btil12_k1,2) * kron( B34matcT_k1'*kron(U4_k1'*U4star_k2,U3_k1'*U3star_k2)     *B34starmatcT_k2,     U1_k1'*U1star_k2 ) ...
                                       + U2_k1 * tens2mat(Btil12_k1,2) * kron( B34matcT_k1'*kron(U4_k1'*U4star_k2,(F3*U3_k1)'*U3star_k2)*B34starmatcT_k2,     U1_k1'*U1star_k2 )...
                                       + U2_k1 * tens2mat(Btil12_k1,2) * kron( B34matcT_k1'*kron((F4*U4_k1)'*U4star_k2,U3_k1'*U3star_k2)*B34starmatcT_k2,     U1_k1'*U1star_k2 )...
                                       + U2_k1 * tens2mat(Btil12_k1,2) * kron( B34matcT_k1'*kron(U4_k1'*U4star_k2,     U3_k1'*U3star_k2)*B34starmatcT_k2,(F1*U1_k1)'*U1star_k2 )) );

        K3_k2 = sylvesterf( (speye(N3)-dtn*nu*F3), ...
                    -dtn*nu*( kron( B12starmatcT_k2' * kron( speye(r2star_k2), U1star_k2'*F1'*U1star_k2 )*B12starmatcT_k2, speye(r4star_k2) ) ...
                            + kron( B12starmatcT_k2' * kron( U2star_k2'*F2'*U2star_k2, speye(r1star_k2) )*B12starmatcT_k2, speye(r4star_k2) ) ...
                            + kron( B12starmatcT_k2' * kron( speye(r2star_k2), speye(r1star_k2)         )*B12starmatcT_k2, (F4*U4star_k2)'*U4star_k2 )), ...
                   U3_n * tens2mat(Btil34_n,1) * kron( B12matcT_n' * kron(U2_n'*U2star_k2, U1_n'*U1star_k2)*B12starmatcT_k2, U4_n'*U4star_k2) ...
                   + dtn*(1-nu)/2 * (F3* U3_k1 * tens2mat(Btil34_k1,1) * kron( B12matcT_k1'*kron(U2_k1'*U2star_k2, U1_k1'*U1star_k2)     *B12starmatcT_k2,     U4_k1' *U4star_k2 ) ...
                                       + U3_k1 * tens2mat(Btil34_k1,1) * kron( B12matcT_k1'*kron(U2_k1'*U2star_k2, (F1*U1_k1)'*U1star_k2)*B12starmatcT_k2,     U4_k1' *U4star_k2 ) ...
                                       + U3_k1 * tens2mat(Btil34_k1,1) * kron( B12matcT_k1'*kron((F2*U2_k1)'*U2star_k2, U1_k1'*U1star_k2)*B12starmatcT_k2,     U4_k1' *U4star_k2 ) ...
                                       + U3_k1 * tens2mat(Btil34_k1,1) * kron( B12matcT_k1'*kron(U2_k1'*U2star_k2, U1_k1'*U1star_k2)     *B12starmatcT_k2, (F4*U4_k1)'*U4star_k2 )) );


        K4_k2 = sylvesterf( (speye(N4)-dtn*nu*F4), ...
                    -dtn*nu*( kron( B12starmatcT_k2' * kron( speye(r2star_k2), U1star_k2'*F1'*U1star_k2 )*B12starmatcT_k2, speye(r3star_k2) ) ...
                            + kron( B12starmatcT_k2' * kron( U2star_k2'*F2'*U2star_k2, speye(r1star_k2) )*B12starmatcT_k2, speye(r3star_k2) ) ...
                            + kron( B12starmatcT_k2' * kron( speye(r2star_k2), speye(r1star_k2)         )*B12starmatcT_k2, (F3*U3star_k2)'*U3star_k2 )), ...
                   U4_n * tens2mat(Btil34_n,2) * kron( B12matcT_n' * kron(U2_n'*U2star_k2, U1_n'*U1star_k2)*B12starmatcT_k2, U3_n'*U3star_k2) ...
                   + dtn*(1-nu)/2 * (F4* U4_k1 * tens2mat(Btil34_k1,2) * kron( B12matcT_k1'*kron(U2_k1'*U2star_k2, U1_k1'*U1star_k2)     *B12starmatcT_k2,     U3_k1' *U3star_k2 ) ...
                                       + U4_k1 * tens2mat(Btil34_k1,2) * kron( B12matcT_k1'*kron(U2_k1'*U2star_k2, (F1*U1_k1)'*U1star_k2)*B12starmatcT_k2,     U3_k1' *U3star_k2 ) ...
                                       + U4_k1 * tens2mat(Btil34_k1,2) * kron( B12matcT_k1'*kron((F2*U2_k1)'*U2star_k2, U1_k1'*U1star_k2)*B12starmatcT_k2,     U3_k1' *U3star_k2 ) ...
                                       + U4_k1 * tens2mat(Btil34_k1,2) * kron( B12matcT_k1'*kron(U2_k1'*U2star_k2, U1_k1'*U1star_k2)     *B12starmatcT_k2, (F3*U3_k1)'*U3star_k2 )) );

        [U1hat_k2,U2hat_k2,U3hat_k2,U4hat_k2,r1hat_k2,r2hat_k2,r3hat_k2,r4hat_k2] = redaugk4(K1_k2,K2_k2,K3_k2,K4_k2,[U1_k1,U1_n],[U2_k1,U2_n],[U3_k1,U3_n],[U4_k1,U4_n]);

        % B steps (2)

        M1 = speye(r34star_k2); 
        A1 = -dtn*(nu)*U2hat_k2'*(F2*U2hat_k2);
        H = speye(r1hat_k2); 
        A2 = B34starmatcT_k2'*kron(speye(r4star_k2), -dtn*(nu)* U3star_k2'*(F3*U3star_k2))*B34starmatcT_k2+ B34starmatcT_k2'*kron(-dtn*(nu)* U4star_k2'*(F4*U4star_k2), speye(r3star_k2))*B34starmatcT_k2; 
        M = speye(r2hat_k2);
        H3 = speye(r34star_k2);
        A3 = -dtn*(nu)*U1hat_k2'*(F1*U1hat_k2);
        B =   ttm(Btil12_n, {U1hat_k2'*U1_n, U2hat_k2'* U2_n, B34starmatcT_k2' * kron( U3star_k2'*U3_n, U4star_k2'*U4_n) * B34matcT_n},[1,2,3])+...
            ( ttm(Btil12_k1, {               U1hat_k2'*   U1_k1, U2hat_k2'*   U2_k1, B34starmatcT_k2' * kron(dtn*(1-nu)/2 * U3star_k2'*F3*U3_k1, U4star_k2'   *U4_k1) * B34matcT_k1},[1,2,3])+...
              ttm(Btil12_k1, {dtn*(1-nu)/2 * U1hat_k2'*F1*U1_k1, U2hat_k2'*   U2_k1, B34starmatcT_k2' * kron(               U3star_k2'   *U3_k1, U4star_k2'   *U4_k1) * B34matcT_k1},[1,2,3])+ ...
              ttm(Btil12_k1, {               U1hat_k2'*   U1_k1, U2hat_k2'*   U2_k1, B34starmatcT_k2' * kron(dtn*(1-nu)/2 * U3star_k2'   *U3_k1, U4star_k2'*F4*U4_k1) * B34matcT_k1},[1,2,3])+...
              ttm(Btil12_k1, {dtn*(1-nu)/2 * U1hat_k2'*   U1_k1, U2hat_k2'*F2*U2_k1, B34starmatcT_k2' * kron(               U3star_k2'   *U3_k1, U4star_k2'   *U4_k1) * B34matcT_k1},[1,2,3])); %r1hatxr2hatxr34
        [~,Btil12hat_k2] = Simoncini_V2(M1,A1,H,A2,M,H3,A3,B);        
        [~,~,r12hat_k2] = size(Btil12hat_k2); 
        Btil12hatmat = tens2mat(Btil12hat_k2,3) ;
        [B12hatmat_k2,~] = qr(Btil12hatmat',0);

        M1 = speye(r12star_k2); 
        A1 = -dtn*(nu)*U4hat_k2'*(F4*U4hat_k2);
        H = speye(r3hat_k2); 
        A2 = B12starmatcT_k2'*kron(speye(r2star_k2), -dtn*(nu)* U1star_k2'*(F1*U1star_k2))*B12starmatcT_k2+ B12starmatcT_k2'*kron(-dtn*(nu)* U2star_k2'*(F2*U2star_k2), speye(r1star_k2))*B12starmatcT_k2;
        M = speye(r4hat_k2);
        H3 = speye(r12star_k2);
        A3 = -dtn*(nu)*U3hat_k2'*(F3*U3hat_k2);
        B =   ttm(Btil34_n,  {U3hat_k2'*U3_n, U4hat_k2'* U4_n, B12starmatcT_k2' * kron( U1star_k2'*U1_n, U2star_k2'*U2_n) * B12matcT_n},[1,2,3])+...
            ( ttm(Btil34_k1, {               U3hat_k2'   *U3_k1, U4hat_k2'*   U4_k1, B12starmatcT_k2'* kron( dtn*(1-nu)/2 * U1star_k2'*F1*U1_k1, U2star_k2'*   U2_k1) * B12matcT_k1},[1,2,3])+...
              ttm(Btil34_k1, {dtn*(1-nu)/2 * U3hat_k2'*F3*U3_k1, U4hat_k2'*   U4_k1, B12starmatcT_k2'* kron(                U1star_k2'   *U1_k1, U2star_k2'*   U2_k1) * B12matcT_k1},[1,2,3])+...
              ttm(Btil34_k1, {               U3hat_k2'   *U3_k1, U4hat_k2'*   U4_k1, B12starmatcT_k2'* kron( dtn*(1-nu)/2 * U1star_k2'   *U1_k1, U2star_k2'*F2*U2_k1) * B12matcT_k1},[1,2,3])+...
              ttm(Btil34_k1, {dtn*(1-nu)/2 * U3hat_k2'   *U3_k1, U4hat_k2'*F4*U4_k1, B12starmatcT_k2'* kron(                U1star_k2'   *U1_k1, U2star_k2'*   U2_k1) * B12matcT_k1},[1,2,3])); %r1hatxr2hatxr34
        [~,Btil34hat_k2] = Simoncini_V2(M1,A1,H,A2,M,H3,A3,B);        
        [~,~,r34hat_k2] = size(Btil34hat_k2); 
        Btil34hatmat = tens2mat(Btil34hat_k2,3) ;
        [B34hatmat_k2,~] = qr(Btil34hatmat',0);

        B12hat_k2 = reshape(B12hatmat_k2, [r1hat_k2,r2hat_k2,r12hat_k2]);
        B34hat_k2 = reshape(B34hatmat_k2, [r3hat_k2,r4hat_k2,r34hat_k2]);

        % B1234 equation
        A =  speye(r34hat_k2) - (B12hatmat_k2'*kron(dtn*nu*speye(r2hat_k2),U1hat_k2'*(F1*U1hat_k2))*B12hatmat_k2+...
                                 B12hatmat_k2'*kron(dtn*nu*U2hat_k2'*(F2*U2hat_k2),speye(r1hat_k2))*B12hatmat_k2);

        B = -dtn*nu*B34hatmat_k2'*kron(speye(r4hat_k2),((F3*U3hat_k2)'*U3hat_k2))*B34hatmat_k2 ...
            -dtn*nu*B34hatmat_k2'*kron(((F4*U4hat_k2)'*U4hat_k2),speye(r3hat_k2))*B34hatmat_k2;


        B1234hat_k2 = sylvesterf(A ,B ,...
            B12hatmat_k2'*kron(U2hat_k2'*U2_n,U1hat_k2'*U1_n)*B12matcT_n*B1234_n*B34matcT_n'*kron(U4_n'*U4hat_k2,U3_n'*U3hat_k2)*B34hatmat_k2 + ...
            dtn*(1-nu)/2*( B12hatmat_k2'*kron(U2hat_k2'*   U2_k1, U1hat_k2'*F1*U1_k1)*B12matcT_k1*B1234_k1*B34matcT_k1'*kron(U4_k1'*   U4hat_k2,U3_k1'*   U3hat_k2)*B34hatmat_k2+...
                           B12hatmat_k2'*kron(U2hat_k2'*   U2_k1, U1hat_k2'*   U1_k1)*B12matcT_k1*B1234_k1*B34matcT_k1'*kron(U4_k1'*   U4hat_k2,U3_k1'*F3*U3hat_k2)*B34hatmat_k2+...
                           B12hatmat_k2'*kron(U2hat_k2'*F2*U2_k1, U1hat_k2'*   U1_k1)*B12matcT_k1*B1234_k1*B34matcT_k1'*kron(U4_k1'*   U4hat_k2,U3_k1'*   U3hat_k2)*B34hatmat_k2+...
                           B12hatmat_k2'*kron(U2hat_k2'*   U2_k1, U1hat_k2'*   U1_k1)*B12matcT_k1*B1234_k1*B34matcT_k1'*kron(U4_k1'*F4*U4hat_k2,U3_k1'*   U3hat_k2)*B34hatmat_k2) );
    

        %TRUNCATE 
        Uhats_2 = {[],[],[],full(U1hat_k2),full(U2hat_k2),full(U3hat_k2),full(U4hat_k2)};
        Bhats_2 = {full(B1234hat_k2),full(B12hat_k2),full(B34hat_k2),[],[],[],[]};

        ht_u0 = htensor(ht_u0.children,ht_u0.dim2ind, Uhats_2, Bhats_2);
        opts.max_rank = min([N1,N2,N3,N4]);
        opts.rel_eps = tol;
        opts.abs_eps = tol;
        ht_u0 = truncate_nonorthog(ht_u0,opts);

        % k1 vals
        B1234_k2 = ht_u0.B{1}; 
        B12_k2 = ht_u0.B{2}; 
        B34_k2 = ht_u0.B{3};
        U1_k2 = ht_u0.U{4};
        U2_k2 = ht_u0.U{5};
        U3_k2 = ht_u0.U{6};
        U4_k2 = ht_u0.U{7};

        B12matcT_k2 = tens2mat(B12_k2,3)';
        B34matcT_k2 = tens2mat(B34_k2,3)';
        
        [~,r1_k2] = size(U1_k2);
        [~,r2_k2] = size(U2_k2);
        [~,r3_k2] = size(U3_k2);
        [~,r4_k2] = size(U4_k2);
        [r12_k2,r34_k2] = size(B1234_k2);

        Btil12mat_k2 = tens2mat(B12_k2,3)'* B1234_k2 ;
        Btil34mat_k2 = tens2mat(B34_k2,3)'* B1234_k2'; %' wuz here
        
        Btil12_k2 = reshape(Btil12mat_k2,[r1_k2,r2_k2,r34_k2]);
        Btil34_k2 = reshape(Btil34mat_k2,[r3_k2,r4_k2,r12_k2]);


        Uk2s = {[],[],[],full(U1_k2),full(U2_k2),full(U3_k2),full(U4_k2)};
        Bk2s = {full(B1234_k2),full(B12_k2),full(B34_k2),[],[],[],[]};
        
        htk2 = htensor(ht_u0.children,ht_u0.dim2ind, Uk2s, Bk2s);
      

        [K1_nn_ddag,K2_nn_ddag,K3_nn_ddag,K4_nn_ddag] = Kup4Dfull(dtn, N1,N2,N3,N4,r1_n,r2_n,r3_n,r4_n,r12_n,r34_n,F1,F2,F3,F4,U1_n,U2_n,U3_n,U4_n,U1_n,U2_n,U3_n,U4_n,B12_n,B34_n,B12matcT_n,B34matcT_n,Btil12_n,Btil34_n);
        [U1hat_nn_ddag,U2hat_nn_ddag,U3hat_nn_ddag,U4hat_nn_ddag,r1hat_nn_ddag,r2hat_nn_ddag,r3hat_nn_ddag,r4hat_nn_ddag] = redaugk4(K1_nn_ddag,K2_nn_ddag,K3_nn_ddag,K4_nn_ddag,U1_n,U2_n,U3_n,U4_n);
        
        [r12hat_nn_ddag,B12hatmat_nn_ddag] = Bupfull(dtn,r1hat_nn_ddag,r2hat_nn_ddag,r34_n,r3_n,r4_n,F1,F2,F3,F4,U1_n,U2_n,U3_n,U4_n,U1hat_nn_ddag,U2hat_nn_ddag, U3_n, U4_n, B34_n,B34matcT_n,Btil12_n);
        [r34hat_nn_ddag,B34hatmat_nn_ddag] = Bupfull(dtn,r3hat_nn_ddag,r4hat_nn_ddag,r12_n,r1_n,r2_n,F3,F4,F1,F2,U3_n,U4_n,U1_n,U2_n,U3hat_nn_ddag,U4hat_nn_ddag, U1_n, U2_n, B12_n,B12matcT_n,Btil34_n);              
 
        B12hat_nn_ddag = reshape(B12hatmat_nn_ddag, [r1hat_nn_ddag,r2hat_nn_ddag,r12hat_nn_ddag]);
        B34hat_nn_ddag = reshape(B34hatmat_nn_ddag, [r3hat_nn_ddag,r4hat_nn_ddag,r34hat_nn_ddag]);

        % B1234 equation

        A =  speye(r34hat_nn_ddag) - (B12hatmat_nn_ddag'*kron(dtn*speye(r2hat_nn_ddag),U1hat_nn_ddag'*(F1*U1hat_nn_ddag))*B12hatmat_nn_ddag+...
                                 B12hatmat_nn_ddag'*kron(dtn*U2hat_nn_ddag'*(F2*U2hat_nn_ddag),speye(r1hat_nn_ddag))*B12hatmat_nn_ddag);

        B = -dtn*B34hatmat_nn_ddag'*kron(speye(r4hat_nn_ddag),((F3*U3hat_nn_ddag)'*U3hat_nn_ddag))*B34hatmat_nn_ddag ...
            -dtn*B34hatmat_nn_ddag'*kron(((F4*U4hat_nn_ddag)'*U4hat_nn_ddag),speye(r3hat_nn_ddag))*B34hatmat_nn_ddag;

        B1234hat_nn_ddag = sylvesterf(A, B ,...
            B12hatmat_nn_ddag'*kron(U2hat_nn_ddag'*U2_n,U1hat_nn_ddag'*U1_n)*B12matcT_n*B1234_n*B34matcT_n'*kron(U4_n'*U4hat_nn_ddag,U3_n'*U3hat_nn_ddag)*B34hatmat_nn_ddag );
        
       
        Unns_ddag = {[],[],[],full(U1hat_nn_ddag),full(U2hat_nn_ddag),full(U3hat_nn_ddag),full(U4hat_nn_ddag)};
        Bnns_ddag = {full(B1234hat_nn_ddag),full(B12hat_nn_ddag),full(B34hat_nn_ddag),[],[],[],[]};

        htk3_ddag = htensor(ht_u0.children,ht_u0.dim2ind,Unns_ddag,Bnns_ddag);

        htaug2 = plus(plus(plus(htn,htk1),htk2),htk3_ddag);
        htaug2 = truncate_nonorthog(htaug2,opts);

        B1234star_k3 = htaug2.B{1}; 
        B12star_k3 = htaug2.B{2}; 
        B34star_k3 = htaug2.B{3};
        U1star_k3 = htaug2.U{4};
        U2star_k3 = htaug2.U{5};
        U3star_k3 = htaug2.U{6};
        U4star_k3 = htaug2.U{7};

        B12starmatcT_k3 = tens2mat(B12star_k3,3)';
        B34starmatcT_k3 = tens2mat(B34star_k3,3)';

        [~,r1star_k3] = size(U1star_k3);
        [~,r2star_k3] = size(U2star_k3);
        [~,r3star_k3] = size(U3star_k3);
        [~,r4star_k3] = size(U4star_k3);
        [r12star_k3,r34star_k3] = size(B1234star_k3);

        %% Stage 3
        %sol(tn)+dt*B1*D(sol(t(k=1)) + dt*B2*Dsol(t(k=2)) + %dt*nu*D(sol(t(k=3))

        %K steps (3)

         K1_k3 = sylvesterf( (speye(N1)-dtn*nu*F1), ...
                    -dtn*nu*( kron( B34starmatcT_k3' * kron( speye(r4star_k3), U3star_k3'*F3'*U3star_k3 )*B34starmatcT_k3,speye(r2star_k3) ) ...
                            + kron( B34starmatcT_k3' * kron( U4star_k3'*F4'*U4star_k3, speye(r3star_k3) )*B34starmatcT_k3,speye(r2star_k3) )...
                            + kron( B34starmatcT_k3' * kron( speye(r4star_k3), speye(r3star_k3)         )*B34starmatcT_k3,(F2*U2star_k3)' * U2star_k3 )), ...
                    U1_n * tens2mat(Btil12_n,1) * kron( B34matcT_n' * kron(U4_n'*U4star_k3, U3_n'*U3star_k3)*B34starmatcT_k3, U2_n'*U2star_k3) ...
                   + B1*dtn * (F1* U1_k1 * tens2mat(Btil12_k1,1) *   kron( B34matcT_k1'*kron(     U4_k1'*U4star_k3,     U3_k1'*U3star_k3)*B34starmatcT_k3,U2_k1'*U2star_k3) ...
                                 + U1_k1 * tens2mat(Btil12_k1,1) *   kron( B34matcT_k1'*kron(     U4_k1'*U4star_k3,(F3*U3_k1)'*U3star_k3)*B34starmatcT_k3,U2_k1'*U2star_k3) ...
                                 + U1_k1 * tens2mat(Btil12_k1,1) *   kron( B34matcT_k1'*kron((F4*U4_k1)'*U4star_k3,     U3_k1'*U3star_k3)*B34starmatcT_k3,U2_k1'*U2star_k3)...
                                 + U1_k1 * tens2mat(Btil12_k1,1) *   kron( B34matcT_k1'*kron(     U4_k1'*U4star_k3,     U3_k1'*U3star_k3)*B34starmatcT_k3,(F2*U2_k1)'*U2star_k3 ) ) ...
                   + B2*dtn * (F1* U1_k2 * tens2mat(Btil12_k2,1) *   kron( B34matcT_k2'*kron(     U4_k2'*U4star_k3,     U3_k2'*U3star_k3)*B34starmatcT_k3,U2_k2'*U2star_k3) ...
                                 + U1_k2 * tens2mat(Btil12_k2,1) *   kron( B34matcT_k2'*kron(     U4_k2'*U4star_k3,(F3*U3_k2)'*U3star_k3)*B34starmatcT_k3,U2_k2'*U2star_k3) ...
                                 + U1_k2 * tens2mat(Btil12_k1,1) *   kron( B34matcT_k2'*kron((F4*U4_k2)'*U4star_k3,     U3_k2'*U3star_k3)*B34starmatcT_k3,U2_k2'*U2star_k3)...
                                 + U1_k2 * tens2mat(Btil12_k1,1) *   kron( B34matcT_k2'*kron(     U4_k2'*U4star_k3,     U3_k2'*U3star_k3)*B34starmatcT_k3,(F2*U2_k2)'*U2star_k3 ) ));

         K2_k3 = sylvesterf( (speye(N2)-dtn*nu*F2), ...
                    -dtn*nu*( kron( B34starmatcT_k3' * kron( speye(r4star_k3), U3star_k3'*F3'*U3star_k3 )*B34starmatcT_k3,speye(r1star_k3) ) ...
                            + kron( B34starmatcT_k3' * kron( U4star_k3'*F4'*U4star_k3, speye(r3star_k3) )*B34starmatcT_k3,speye(r1star_k3) )...
                            + kron( B34starmatcT_k3' * kron( speye(r4star_k3), speye(r3star_k3)         )*B34starmatcT_k3,(F1*U1star_k3)' * U1star_k3 )) , ...
                    U2_n * tens2mat(Btil12_n,1) * kron( B34matcT_n' * kron(U4_n'*U4star_k3, U3_n'*U3star_k3)*B34starmatcT_k3, U1_n'*U1star_k3) ...
                   + B1*dtn * (F2* U2_k1 * tens2mat(Btil12_k1,2) *   kron( B34matcT_k1'*kron(U4_k1'*U4star_k3,     U3_k1'*U3star_k3)*B34starmatcT_k3,U1_k1'*U1star_k3) ...
                                 + U2_k1 * tens2mat(Btil12_k1,2) *   kron( B34matcT_k1'*kron(U4_k1'*U4star_k3,(F3*U3_k1)'*U3star_k3)*B34starmatcT_k3,U1_k1'*U1star_k3) ...
                                 + U2_k1 * tens2mat(Btil12_k1,2) *   kron( B34matcT_k1'*kron((F4*U4_k1)'*U4star_k3,U3_k1'*U3star_k3)*B34starmatcT_k3,U1_k1'*U1star_k3)...
                                 + U2_k1 * tens2mat(Btil12_k1,2) *   kron( B34matcT_k1'*kron(U4_k1'*U4star_k3,     U3_k1'*U3star_k3)*B34starmatcT_k3,(F1*U1_k1)'*U1star_k3 ) ) ...
                   + B2*dtn * (F2* U2_k2 * tens2mat(Btil12_k2,2) *   kron( B34matcT_k2'*kron(U4_k2'*U4star_k3,     U3_k2'*U3star_k3)*B34starmatcT_k3,U1_k2'*U1star_k3) ...
                                 + U2_k2 * tens2mat(Btil12_k2,2) *   kron( B34matcT_k2'*kron(U4_k2'*U4star_k3,(F3*U3_k2)'*U3star_k3)*B34starmatcT_k3,U1_k2'*U1star_k3) ...
                                 + U2_k2 * tens2mat(Btil12_k2,2) *   kron( B34matcT_k2'*kron((F4*U4_k2)'*U4star_k3,U3_k2'*U3star_k3)*B34starmatcT_k3,U1_k2'*U1star_k3)...
                                 + U2_k2 * tens2mat(Btil12_k2,2) *   kron( B34matcT_k2'*kron(U4_k2'*U4star_k3,     U3_k2'*U3star_k3)*B34starmatcT_k3,(F1*U1_k2)'*U1star_k3 ) ));


        K3_k3 = sylvesterf( (speye(N3)-dtn*nu*F3), ...
                    -dtn*nu*( kron( B12starmatcT_k3' * kron( speye(r2star_k3), U1star_k3'*F1'*U1star_k3 )*B12starmatcT_k3,speye(r4star_k3) ) ...
                            + kron( B12starmatcT_k3' * kron( U2star_k3'*F2'*U2star_k3, speye(r1star_k3) )*B12starmatcT_k3,speye(r4star_k3) )...
                            + kron( B12starmatcT_k3' * kron( speye(r2star_k3), speye(r1star_k3)         )*B12starmatcT_k3,(F4*U4star_k3)' * U4star_k3 )), ...
                    U3_n * tens2mat(Btil34_n,1) * kron( B12matcT_n' * kron(U2_n'*U2star_k3, U1_n'*U1star_k3)*B12starmatcT_k3, U4_n'*U4star_k3) ...
                   + B1*dtn * (F3* U3_k1 * tens2mat(Btil34_k1,1) *   kron( B12matcT_k1'*kron(U2_k1'*U2star_k3,     U1_k1'*U1star_k3)*B12starmatcT_k3,    U4_k1'*U4star_k3) ...
                                 + U3_k1 * tens2mat(Btil34_k1,1) *   kron( B12matcT_k1'*kron(U2_k1'*U2star_k3,(F1*U1_k1)'*U1star_k3)*B12starmatcT_k3,    U4_k1'*U4star_k3) ...
                                 + U3_k1 * tens2mat(Btil34_k1,1) *   kron( B12matcT_k1'*kron((F2*U2_k1)'*U2star_k3,U1_k1'*U1star_k3)*B12starmatcT_k3,    U4_k1'*U4star_k3)...
                                 + U3_k1 * tens2mat(Btil34_k1,1) *   kron( B12matcT_k1'*kron(U2_k1'*U2star_k3,     U1_k1'*U1star_k3)*B12starmatcT_k3,(F4*U4_k1)'*U4star_k3) ) ...
                   + B2*dtn * (F3* U3_k2 * tens2mat(Btil34_k2,1) *   kron( B12matcT_k2'*kron(U2_k2'*U2star_k3,     U1_k2'*U1star_k3)*B12starmatcT_k3,    U4_k2'*U4star_k3) ...
                                 + U3_k2 * tens2mat(Btil34_k2,1) *   kron( B12matcT_k2'*kron(U2_k2'*U2star_k3,(13*U1_k2)'*U1star_k3)*B12starmatcT_k3,    U4_k2'*U4star_k3) ...
                                 + U3_k2 * tens2mat(Btil34_k2,1) *   kron( B12matcT_k2'*kron((F2*U2_k2)'*U2star_k3,U1_k2'*U1star_k3)*B12starmatcT_k3,    U4_k2'*U4star_k3)...
                                 + U3_k2 * tens2mat(Btil34_k2,1) *   kron( B12matcT_k2'*kron(U2_k2'*U2star_k3,     U1_k2'*U1star_k3)*B12starmatcT_k3,(F4*U4_k2)'*U4star_k3) ));


        K4_k3 = sylvesterf( (speye(N4)-dtn*nu*F4), ...
                    -dtn*nu*( kron( B12starmatcT_k3' * kron( speye(r2star_k3), U1star_k3'*F1'*U1star_k3 )*B12starmatcT_k3,speye(r3star_k3) ) ...
                            + kron( B12starmatcT_k3' * kron( U2star_k3'*F2'*U2star_k3, speye(r1star_k3) )*B12starmatcT_k3,speye(r3star_k3) )...
                            + kron( B12starmatcT_k3' * kron( speye(r2star_k3), speye(r1star_k3)         )*B12starmatcT_k3,(F3*U3star_k3)' * U3star_k3 )), ...
                    U4_n * tens2mat(Btil34_n,2) * kron( B12matcT_n' * kron(U2_n'*U2star_k3, U1_n'*U1star_k3)*B12starmatcT_k3, U3_n'*U3star_k3) ...
                   + B1*dtn * (F4* U4_k1 * tens2mat(Btil34_k1,2) *   kron( B12matcT_k1'*kron(U2_k1'*U2star_k3,     U1_k1'*U1star_k3)*B12starmatcT_k3,    U3_k1'*U3star_k3) ...
                                 + U4_k1 * tens2mat(Btil34_k1,2) *   kron( B12matcT_k1'*kron(U2_k1'*U2star_k3,(F1*U1_k1)'*U1star_k3)*B12starmatcT_k3,    U3_k1'*U3star_k3) ...
                                 + U4_k1 * tens2mat(Btil34_k1,2) *   kron( B12matcT_k1'*kron((F2*U2_k1)'*U2star_k3,U1_k1'*U1star_k3)*B12starmatcT_k3,    U3_k1'*U3star_k3)...
                                 + U4_k1 * tens2mat(Btil34_k1,2) *   kron( B12matcT_k1'*kron(U2_k1'*U2star_k3,     U1_k1'*U1star_k3)*B12starmatcT_k3,(F3*U3_k1)'*U3star_k3 ) ) ...
                   + B2*dtn * (F4* U4_k2 * tens2mat(Btil34_k2,2) *   kron( B12matcT_k2'*kron(U2_k2'*U2star_k3,     U1_k2'*U1star_k3)*B12starmatcT_k3,    U3_k2'*U3star_k3) ...
                                 + U4_k2 * tens2mat(Btil34_k2,2) *   kron( B12matcT_k2'*kron(U2_k2'*U2star_k3,(F1*U1_k2)'*U1star_k3)*B12starmatcT_k3,    U3_k2'*U3star_k3) ...
                                 + U4_k2 * tens2mat(Btil34_k2,2) *   kron( B12matcT_k2'*kron((F2*U2_k2)'*U2star_k3,U1_k2'*U1star_k3)*B12starmatcT_k3,    U3_k2'*U3star_k3)...
                                 + U4_k2 * tens2mat(Btil34_k2,2) *   kron( B12matcT_k2'*kron(U2_k2'*U2star_k3,     U1_k2'*U1star_k3)*B12starmatcT_k3,(F3*U3_k2)'*U3star_k3) ));

       [U1hat_k3,U2hat_k3,U3hat_k3,U4hat_k3,r1hat_k3,r2hat_k3,r3hat_k3,r4hat_k3] = redaugk4(K1_k3,K2_k3,K3_k3,K4_k3,[U1_k2,U1_k1,U1_n],[U2_k2,U2_k1,U2_n],[U3_k2,U3_k1,U3_n],[U4_k2,U4_k1,U4_n]);

       % B steps (3)

        M1 = speye(r34star_k3); 
        A1 = -dtn*(nu)*U2hat_k3'*(F2*U2hat_k3);
        H = speye(r1hat_k3); 
        A2 = B34starmatcT_k3'*kron(speye(r4star_k3), -dtn*(nu)* U3star_k3'*(F3*U3star_k3))*B34starmatcT_k3+B34starmatcT_k3'*kron(-dtn*(nu)* U4star_k3'*(F4*U4star_k3),speye(r4star_k3) )*B34starmatcT_k3; 
        M = speye(r2hat_k3);
        H3 = speye(r34star_k3);
        A3 = -dtn*(nu)*U1hat_k3'*(F1*U1hat_k3);
        B =   ttm(Btil12_n,  {         U1hat_k3'*   U1_n,  U2hat_k3'*    U2_n,  B34starmatcT_k3' * kron(         U3star_k3'   *U3_n,  U4star_k3'*   U4_n ) * B34matcT_n },[1,2,3])+...
            ( ttm(Btil12_k1, {         U1hat_k3'*   U1_k1, U2hat_k3'*    U2_k1, B34starmatcT_k3' * kron(B1*dtn * U3star_k3'*F3*U3_k1, U4star_k3'*   U4_k1) * B34matcT_k1},[1,2,3])+...
              ttm(Btil12_k1, {B1*dtn * U1hat_k3'*F1*U1_k1, U2hat_k3'*    U2_k1, B34starmatcT_k3' * kron(         U3star_k3'   *U3_k1, U4star_k3'*   U4_k1) * B34matcT_k1},[1,2,3])+...
              ttm(Btil12_k1, {         U1hat_k3'*   U1_k1, U2hat_k3'*    U2_k1, B34starmatcT_k3' * kron(B1*dtn * U3star_k3'*   U3_k1, U4star_k3'*F4*U4_k1) * B34matcT_k1},[1,2,3])+...
              ttm(Btil12_k1, {B1*dtn * U1hat_k3'*   U1_k1, U2hat_k3'*F2* U2_k1, B34starmatcT_k3' * kron(         U3star_k3'   *U3_k1, U4star_k3'*   U4_k1) * B34matcT_k1},[1,2,3]))+...
            ( ttm(Btil12_k2, {         U1hat_k3'*   U1_k2, U2hat_k3'*    U2_k2, B34starmatcT_k3' * kron(B2*dtn * U3star_k3'*F3*U3_k2, U4star_k3'*   U4_k2) * B34matcT_k2},[1,2,3])+...
              ttm(Btil12_k2, {B2*dtn * U1hat_k3'*F1*U1_k2, U2hat_k3'*    U2_k2, B34starmatcT_k3' * kron(         U3star_k3'   *U3_k2, U4star_k3'*   U4_k2) * B34matcT_k2},[1,2,3])+...
              ttm(Btil12_k2, {         U1hat_k3'*   U1_k2, U2hat_k3'*    U2_k2, B34starmatcT_k3' * kron(B1*dtn * U3star_k3'*   U3_k2, U4star_k3'*F4*U4_k2) * B34matcT_k2},[1,2,3])+...
              ttm(Btil12_k2, {B1*dtn * U1hat_k3'*   U1_k2, U2hat_k3'*F2* U2_k2, B34starmatcT_k3' * kron(         U3star_k3'   *U3_k2, U4star_k3'*   U4_k2) * B34matcT_k2},[1,2,3])); 
        [~,Btil12hat_k3] = Simoncini_V2(M1,A1,H,A2,M,H3,A3,B);        
        [~,~,r12hat_k3] = size(Btil12hat_k3); 
        Btil12hatmat_k3 = tens2mat(Btil12hat_k3,3) ;
        [B12hatmat_k3,~] = qr(Btil12hatmat_k3',0);

        % LEFT OFF HERE

        M1 = speye(r12star_k3); 
        A1 = -dtn*(nu)*U4hat_k3'*(F4*U4hat_k3);
        H = speye(r3hat_k3); 
        A2 = B12starmatcT_k3'*kron(speye(r2star_k3), -dtn*(nu)* U1star_k3'*(F1*U1star_k3))*B12starmatcT_k3 + B12starmatcT_k3'*kron(-dtn*(nu)* U1star_k3'*(F1*U1star_k3),speye(r1star_k3))*B12starmatcT_k3;
        M = speye(r4hat_k3);
        H3 = speye(r12star_k3);
        A3 = -dtn*(nu)*U3hat_k3'*(F3*U3hat_k3);
        B =   ttm(Btil34_n,  {         U3hat_k3'*   U3_n,  U4hat_k3'*    U4_n,  B12starmatcT_k3'* kron(         U1star_k3'*   U1_n,  U2star_k3'*   U2_n)  * B12matcT_n},[1,2,3])+...
            ( ttm(Btil34_k1, {         U3hat_k3'   *U3_k1, U4hat_k3'*    U4_k1, B12starmatcT_k3'* kron(B1*dtn * U1star_k3'*F1*U1_k1, U2star_k3'*   U2_k1) * B12matcT_k1},[1,2,3])+...
              ttm(Btil34_k1, {B1*dtn * U3hat_k3'*F3*U3_k1, U4hat_k3'*    U4_k1, B12starmatcT_k3'* kron(         U1star_k3'   *U1_k1, U2star_k3'*   U2_k1) * B12matcT_k1},[1,2,3])+...
              ttm(Btil34_k1, {         U3hat_k3'   *U3_k1, U4hat_k3'*    U4_k1, B12starmatcT_k3'* kron(B1*dtn * U1star_k3'   *U1_k1, U2star_k3'*F2*U2_k1) * B12matcT_k1},[1,2,3])+...
              ttm(Btil34_k1, {B1*dtn * U3hat_k3'   *U3_k1, U4hat_k3'*F4* U4_k1, B12starmatcT_k3'* kron(         U1star_k3'   *U1_k1, U2star_k3'*   U2_k1) * B12matcT_k1},[1,2,3]))+...
            ( ttm(Btil34_k2, {         U3hat_k3'   *U3_k2, U4hat_k3'*    U4_k2, B12starmatcT_k3'* kron(B2*dtn * U1star_k3'*F1*U1_k2, U2star_k3'*   U2_k2) * B12matcT_k2},[1,2,3])+...
              ttm(Btil34_k2, {B2*dtn * U3hat_k3'*F3*U3_k2, U4hat_k3'*    U4_k2, B12starmatcT_k3'* kron(         U1star_k3'   *U1_k2, U2star_k3'*   U2_k2) * B12matcT_k2},[1,2,3])+...
              ttm(Btil34_k2, {         U3hat_k3'   *U3_k2, U4hat_k3'*    U4_k2, B12starmatcT_k3'* kron(B2*dtn * U1star_k3'   *U1_k2, U2star_k3'*F2*U2_k2) * B12matcT_k2},[1,2,3])+...
              ttm(Btil34_k2, {B2*dtn * U3hat_k3'   *U3_k2, U4hat_k3'*F4* U4_k2, B12starmatcT_k3'* kron(         U1star_k3'   *U1_k2, U2star_k3'*   U2_k2) * B12matcT_k2},[1,2,3])); 
        [~,Btil34hat_k3] = Simoncini_V2(M1,A1,H,A2,M,H3,A3,B);        
        [~,~,r34hat_k3] = size(Btil34hat_k3); 
        Btil34hatmat_k3 = tens2mat(Btil34hat_k3,3) ;
        [B34hatmat_k3,~] = qr(Btil34hatmat_k3',0);

        B12hat_k3 = reshape(B12hatmat_k3, [r1hat_k3,r2hat_k3,r12hat_k3]);
        B34hat_k3 = reshape(B34hatmat_k3, [r3hat_k3,r4hat_k3,r34hat_k3]);

        % B1234 equation
        
        A =  speye(r12hat_k3) - (B12hatmat_k3'*kron(dtn*nu*speye(r2hat_k3),U1hat_k3'*(F1*U1hat_k3))*B12hatmat_k3+...
                                 B12hatmat_k3'*kron(dtn*nu*U2hat_k3'*(F2*U2hat_k3),speye(r1hat_k3))*B12hatmat_k3);

        B = -dtn*nu*B34hatmat_k3'*kron(speye(r4hat_k3),((F3*U3hat_k3)'*U3hat_k3))*B34hatmat_k3 ...
            -dtn*nu*B34hatmat_k3'*kron(((F4*U4hat_k3)'*U4hat_k3),speye(r3hat_k3))*B34hatmat_k3;

        B1234hat_k3 = sylvesterf(A ,B ,...
            B12hatmat_k3'*kron(U2hat_k3'*U2_n,U1hat_k3'*U1_n)*B12matcT_n*B1234_n*B34matcT_n'*kron(U4_n'*U4hat_k3,U3_n'*U3hat_k3)*B34hatmat_k3 + ...
            B1*dtn    *( B12hatmat_k3'*kron(U2hat_k3'*   U2_k1, U1hat_k3'*F1*U1_k1)*B12matcT_k1*B1234_k1*B34matcT_k1'*kron(U4_k1'*   U4hat_k3,U3_k1'*   U3hat_k3)*B34hatmat_k3+...
                         B12hatmat_k3'*kron(U2hat_k3'*   U2_k1, U1hat_k3'*   U1_k1)*B12matcT_k1*B1234_k1*B34matcT_k1'*kron(U4_k1'*   U4hat_k3,U3_k1'*F3*U3hat_k3)*B34hatmat_k3+...
                         B12hatmat_k3'*kron(U2hat_k3'*F2*U2_k1, U1hat_k3'*   U1_k1)*B12matcT_k1*B1234_k1*B34matcT_k1'*kron(U4_k1'*   U4hat_k3,U3_k1'*   U3hat_k3)*B34hatmat_k3+...
                         B12hatmat_k3'*kron(U2hat_k3'*   U2_k1, U1hat_k3'*   U1_k1)*B12matcT_k1*B1234_k1*B34matcT_k1'*kron(U4_k1'*F4*U4hat_k3,U3_k1'*   U3hat_k3)*B34hatmat_k3)+... 
            B2*dtn    *( B12hatmat_k3'*kron(U2hat_k3'*   U2_k2, U1hat_k3'*F1*U1_k2)*B12matcT_k2*B1234_k2*B34matcT_k2'*kron(U4_k2'*   U4hat_k3,U3_k2'*   U3hat_k3)*B34hatmat_k3+...
                         B12hatmat_k3'*kron(U2hat_k3'*   U2_k2, U1hat_k3'*   U1_k2)*B12matcT_k2*B1234_k2*B34matcT_k2'*kron(U4_k2'*   U4hat_k3,U3_k2'*F3*U3hat_k3)*B34hatmat_k3+...
                         B12hatmat_k3'*kron(U2hat_k3'*F2*U2_k2, U1hat_k3'*   U1_k2)*B12matcT_k2*B1234_k2*B34matcT_k2'*kron(U4_k2'*   U4hat_k3,U3_k2'*   U3hat_k3)*B34hatmat_k3+...
                         B12hatmat_k3'*kron(U2hat_k3'*   U2_k2, U1hat_k3'*   U1_k2)*B12matcT_k2*B1234_k2*B34matcT_k2'*kron(U4_k2'*F4*U4hat_k3,U3_k2'*   U3hat_k3)*B34hatmat_k3));
    
        %TRUNCATE 
        Uhats_3 = {[],[],[],full(U1hat_k3),full(U2hat_k3),full(U3hat_k3),full(U4hat_k3)};
        Bhats_3 = {full(B1234hat_k3),full(B12hat_k3),full(B34hat_k3),[],[],[],[]};
    
        ht_u0 = htensor(ht_u0.children,ht_u0.dim2ind, Uhats_3, Bhats_3);
        opts.max_rank = min([N1,N2,N3,N4]);
        opts.rel_eps = tol;
        opts.abs_eps = tol;
        ht_u0 = truncate_nonorthog(ht_u0,opts);

        %  vals
        B1234_n = ht_u0.B{1};
        B12_n = ht_u0.B{2};
        B34_n = ht_u0.B{3};
        U1_n = ht_u0.U{4};
        U2_n = ht_u0.U{5};  
        U3_n = ht_u0.U{6};
        U4_n = ht_u0.U{7};
    
        B12matcT_n = tens2mat(B12_n,3)';
        B34matcT_n = tens2mat(B34_n,3)';
        
        U1star_k1 = U1_n;
        U2star_k1 = U2_n;
        U3star_k1 = U3_n;  
        U4star_k1 = U4_n;
        B12star_k1 = B12_n;
        B34star_k1 = B34_n;
        B12starmatcT_k1 = tens2mat(B12star_k1,3)';
        B34starmatcT_k1 = tens2mat(B34star_k1,3)';
    
        [~,r1_n] = size(U1_n);
        [~,r2_n] = size(U2_n);
        [~,r3_n] = size(U3_n);
        [~,r4_n] = size(U4_n);
    
        r1star_k1 = r1_n;
        r2star_k1 = r2_n;
        r3star_k1 = r3_n;
        r4star_k1 = r4_n;
        [r12_n,r34_n] = size(B1234_n);
        r12star_k1 = r12_n;
        r34star_k1 = r34_n;
        
        Btil12mat_n = tens2mat(B12_n,3)'* B1234_n ;
        Btil34mat_n = tens2mat(B34_n,3)'* B1234_n' ;
    
        Btil12_n = reshape(Btil12mat_n,[r1_n,r2_n,r34_n]) ;
        Btil34_n = reshape(Btil34mat_n,[r3_n,r4_n,r12_n]) ;

        rankvec(n,:) = [r1_n,r2_n,r3_n,r4_n];

        % disp([r12_n,r34_n])

        %diffusion check
        format shortG
        t = full(ht_u0);

        %Plot slice

        slice = t(:,:,20,20);

        % Create coordinate grids
        [x,y] = ndgrid(1:N1, 1:N2);

        % % Plot 
        % figure(1)
        % surf(x, y, slice);
        % axis([0 40 0 40 -150e-3 150e-3]);
        % view(30,15)
        % % view(10, 0);     % camera on +y axis, looking in x-z plane
        % shading interp;        % smooth shading
        % colorbar;
        % xlabel('y');
        % ylabel('x');
        % zlabel('Value');
        % drawnow
        % 
        % if n ~= 2
        %     m0 = m;
        % end
        % 
        % m = max(abs(t(:)));
        % mvec(n-1) = m;
        % 
        % if n ~= 2
        %     mrat = m / (m0);   % or m/m0 depending on your intention
        %     fprintf('mrat = %.5e\n', mrat);
        % end
        % 
        % fprintf('max  = %.5e\n', m);

    end

    L1error(p) = dx1*dx2*dx3*dx4*sum( sum(sum( sum( abs(full(ht_u0) - uexact)))))/L^4;
    disp(L1error(p))
end

%% PLots
% close all;
% figure
% 
% plot(2:Nt,mvec,LineWidth=1.5)
% title("Max over Iteration")


figure
plot(tvals,rankvec(:,1),'b--',LineWidth=1.5);hold on;
plot(tvals,rankvec(:,2),'m-.',LineWidth=1.5)
plot(tvals,rankvec(:,3),'g-',LineWidth=1.5);
plot(tvals,rankvec(:,3),'k.',LineWidth=1.5);hold off;
legend("r1","r2","r3","avg",Location= 'best')
xlabel("Time t")
ylabel("Rank")
title("Rank Test")


hold off;loglog(lambdavec,L1error,'b',LineWidth=1.5);hold on;
loglog(lambdavec(10:end-10),lambdavec(10:end-10).^3*.000013, 'k--', LineWidth=1.5)
title("4D Algorithm Error - Rank 3 IC",FontSize=13)
xlabel('$\lambda, \Delta t =  \lambda / \Big({1/\mathrm{dx}_1 + 1/\mathrm{dx}_2 + 1/\mathrm{dx}_3 + 1/\mathrm{dx}_4}\Big)$', ...
       'Interpreter','latex', FontSize=13);
ylabel("L1 Norm of Difference from Exact Solution ",FontSize = 13)
legend("Error Using Backward Euler", "Slope 1$", Interpreter ='Latex',Location = 'best',FontSize = 13)





