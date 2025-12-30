%---------------------
% Date: 11/30/2025
% Author: Paolo Bosques-Paulet
% HT Algorithm for 4D diffusion eqn w/ 1st order accuracy
%---------------------

close all;clear variables; 

% Add Tensorlab and Htucker toolbox to path!

% Params and EQ Def

% ut = d1* ux1x1 + d2 * ux2x2 + d3* ux3x3
d1 = 1/10;
d2 = 1/10;
d3 = 1/10;
d4 = 1/10;

N1 = 40;
N2 = 40;
N3 = 40;
N4 = 40;

tf = .5;%15 %.5;
L = 2*pi;    %2*pi;

% Cell centeredd grid in space ---

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
% u0 = 0.8*exp(-15*((X1-6.5).^2+(X2-6.5).^2+(X3-6.5).^2))+0.5*exp(-15*((X1-7.5).^2+(X2-7.5).^2+(X3-7.5).^2));
% 
% u0 = 0.8*exp(-15*(X1-1)).^2+0.5*exp(-15*(X1+1).^2);
% %
% % Accuracy Test (Rank1)
% u0 = sin(2*pi/L*X1).*sin(2*pi/L*X2).*sin(2*pi/L*X3).*sin(2*pi/L*X4);
% uexact =  exp(-4*pi^2/L^2 * tf *(d1+d2+d3+d4)).*sin(2*pi/L*X1).*sin(2*pi/L*X2).*sin(2*pi/L*X3).*sin(2*pi/L*X4);
% 
% % Accuracy Test (Rank2)
% u0 = sin(2*pi/L*X1).*sin(2*pi/L*X2).*sin(2*pi/L*X3).*sin(2*pi/L*X4)+...
%     sin(2*2*pi/L*X1).*sin(2*2*pi/L*X2).*sin(2*2*pi/L*X3).*sin(2*2*pi/L*X4);
% 
% uexact = exp(-4*pi^2/L^2 * tf *(d1+d2+d3+d4)).*sin(2*pi/L*X1).*sin(2*pi/L*X2).*sin(2*pi/L*X3).*sin(2*pi/L*X4)+...
%     exp(-4*4*pi^2/L^2 * tf *(d1+d2+d3+d4)).*sin(2*2*pi/L*X1).*sin(2*2*pi/L*X2).*sin(2*2*pi/L*X3).*sin(2*2*pi/L*X4);
% 
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
lambdavec = 0.1:0.05:3;
dtvec = zeros(size(lambdavec));
L1error = zeros(size(lambdavec));
Linferror = zeros(size(lambdavec));

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
    
    U1star_n = U1_n;
    U2star_n = U2_n;
    U3star_n = U3_n;  
    U4star_n = U4_n;
    B12star_n = B12_n;
    B34star_n = B34_n;
    B12starmatcT_n = tens2mat(B12star_n,3)';
    B34starmatcT_n = tens2mat(B34star_n,3)';

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
    [~,r1_n] = size(U1_n);
    [~,r2_n] = size(U2_n);
    [~,r3_n] = size(U3_n);
    [~,r4_n] = size(U4_n);

    r1star_n = r1_n;
    r2star_n = r2_n;
    r3star_n = r3_n;
    r4star_n = r4_n;
    [r12_n,r34_n] = size(B1234_n);
    r12star_n = r12_n;
    r34star_n = r34_n;
    
    rankvec(1,:) = [r1_n,r2_n,r3_n,r4_n];
    
    Nt = numel(tvals);
    
    Btil12mat_n = tens2mat(B12_n,3)'* B1234_n ;
    Btil34mat_n = tens2mat(B34_n,3)'* B1234_n' ;% ' wuz here

    Btil12_n = reshape(Btil12mat_n,[r1_n,r2_n,r34_n]) ;
    Btil34_n = reshape(Btil34mat_n,[r3_n,r4_n,r12_n]) ;
   
    mvec = zeros(1,length(2:Nt));

    for n = 2:Nt 
       
        disp(tvals(n))
        dtn = tvals(n)-tvals(n-1);
        %Ks        
        

       [K1_nn,K2_nn,K3_nn,K4_nn] = Kupfull(dtn, N1,N2,N3,N4,r1star_n,r2star_n,r3star_n,r4star_n,r12star_n,r34star_n,F1,F2,F3,F4,U1_n,U2_n,U3_n,U4_n,U1star_n,U2star_n,U3star_n,U4star_n,B12_n,B34_n,B12starmatcT_n,B34starmatcT_n,Btil12_n,Btil34_n);
       %                           ksteps(dtn, N1,N2,N3,N4,r1star,  r2star,  r3star,  r4star,  r12star,  r34star,  F1,F2,F3,F4,U1_n,U2_n,U3_n,U4_n,U1star,   U2star, U3star,  U4star  ,B12_n,B34_n,B12starmatcT,B34starmatcT,Btil12,Btil34)
       [U1hat_nn,U2hat_nn,U3hat_nn,U4hat_nn,r1hat_nn,r2hat_nn,r3hat_nn,r4hat_nn] = redaugk4(K1_nn,K2_nn,K3_nn,K4_nn,U1_n,U2_n,U3_n,U4_n);
        
       
       [r12hat_nn,B12hatmat_nn] = Bupfull(dtn,r1hat_nn,r2hat_nn,r34star_n,r3star_n,r4star_n,F1,F2,F3,F4,U1_n,U2_n,U3_n,U4_n,U1hat_nn,U2hat_nn, U3star_n, U4star_n, B34_n,B34starmatcT_n,Btil12_n);
       [r34hat_nn,B34hatmat_nn] = Bupfull(dtn,r3hat_nn,r4hat_nn,r12star_n,r1star_n,r2star_n,F3,F4,F1,F2,U3_n,U4_n,U1_n,U2_n,U3hat_nn,U4hat_nn, U1star_n, U2star_n, B12_n,B12starmatcT_n,Btil34_n);
   
                            %     Bup(dtn,rahat,   rbhat,   rcstar,   rc1star, rc2star, Fa, Fb, Fc1,Fc2,Ua_n,Ub_n,Uc1_n,Uc2_n,Uahat, Ubhat,      Uc1star,   Uc2star ,   Bc_n, BcstarmatcT,    Btilp) 
 
        B12hat_nn = reshape(B12hatmat_nn, [r1hat_nn,r2hat_nn,r12hat_nn]);
        B34hat_nn = reshape(B34hatmat_nn, [r3hat_nn,r4hat_nn,r34hat_nn]);

        % B1234 equation
        
        %AX + XB = C
        A =  speye(r34hat_nn) - (B12hatmat_nn'*kron(dtn*speye(r2hat_nn),U1hat_nn'*(F1*U1hat_nn))*B12hatmat_nn+...
                                 B12hatmat_nn'*kron(dtn*U2hat_nn'*(F2*U2hat_nn),speye(r1hat_nn))*B12hatmat_nn);

        B = -dtn*B34hatmat_nn'*kron(speye(r4hat_nn),((F3*U3hat_nn)'*U3hat_nn))*B34hatmat_nn ...
            -dtn*B34hatmat_nn'*kron(((F4*U4hat_nn)'*U4hat_nn),speye(r3hat_nn))*B34hatmat_nn;

        B1234hat_nn = sylvesterf(A, B ,...
            B12hatmat_nn'*kron(U2hat_nn'*U2_n,U1hat_nn'*U1_n)*B12matcT_n*B1234_n*B34matcT_n'*kron(U4_n'*U4hat_nn,U3_n'*U3hat_nn)*B34hatmat_nn );


        %TRUNCATE 
        Uhats = {[],[],[],full(U1hat_nn),full(U2hat_nn),full(U3hat_nn),full(U4hat_nn)};
        Bhats = {full(B1234hat_nn),full(B12hat_nn),full(B34hat_nn),[],[],[],[]};
    
        ht_u0 = htensor(ht_u0.children,ht_u0.dim2ind, Uhats, Bhats);
        opts.max_rank = min([N1,N2,N3,N4]);
        opts.rel_eps = tol;
        opts.abs_eps = tol;
        ht_u0 = truncate_nonorthog(ht_u0,opts);
        
        % ns
        B1234_n = ht_u0.B{1}; 
        B12_n = ht_u0.B{2}; 
        B34_n = ht_u0.B{3};
        U1_n = ht_u0.U{4};
        U2_n = ht_u0.U{5};
        U3_n = ht_u0.U{6};
        U4_n = ht_u0.U{7};

        B12matcT_n = tens2mat(B12_n,3)';
        B34matcT_n = tens2mat(B34_n,3)';
        
        % Approximate update bases
        U1star_n = U1_n;
        U2star_n = U2_n;
        U3star_n = U3_n;  
        U4star_n = U4_n;
        B12star_n = B12_n;
        B34star_n = B34_n;
        B12starmatcT_n = tens2mat(B12star_n,3)';
        B34starmatcT_n = tens2mat(B34star_n,3)';
        
        [~,r1_n] = size(U1_n);
        [~,r2_n] = size(U2_n);
        [~,r3_n] = size(U3_n);
        [~,r4_n] = size(U4_n);
        [r12_n,r34_n] = size(B1234_n);

        r1star_n = r1_n;
        r2star_n = r2_n;
        r3star_n = r3_n;
        r4star_n = r4_n;

        Btil12mat_n = tens2mat(B12_n,3)'* B1234_n ;
        Btil34mat_n = tens2mat(B34_n,3)'* B1234_n'; %' wuz here
        
        Btil12_n = reshape(Btil12mat_n,[r1_n,r2_n,r34_n]);
        Btil34_n = reshape(Btil34mat_n,[r3_n,r4_n,r12_n]);

        disp([r1_n,r2_n,r3_n,r4_n])
        rankvec(n,:) = [r1_n,r2_n,r3_n,r4_n];

       
       
        % Create coordinate grids
        [x,y] = ndgrid(1:N1, 1:N2);

        % % Plot 
        %diffusion check
        % format shortG
        % t = full(ht_u0);
        % slice = t(:,:,20,20);
        % figure(1)
        % surf(x, y, slice);
        % axis([0 40 0 40 -60e-3 60e-3]);
        % view(30,15)
        % % view(10, 0);     % camera on +y axis, looking in x-z plane
        % shading interp;        % smooth shading
        % colorbar;
        % xlabel('y');
        % ylabel('x');
        % zlabel('Value');
        % drawnow
        % % 
        % % if n ~= 2
        %     m0 = m;
        % end
        % 
        % m = max(abs(t(:)));
        % mvec(n-1) = m;
        % 
        % % if n ~= 2
        % %     mrat = m / (m0);   % or m/m0 depending on your intention
        % %     fprintf('mrat = %.5e\n', mrat);
        % % end
        % 
        % fprintf('max  = %.5e\n', m);
        
    end

    L1error(p) = dx1*dx2*dx3*dx4*sum( sum(sum( sum( abs(full(ht_u0) - uexact)))))/L^4;
    Linferror(p) = max(max(max(max(abs(full(ht_u0)-uexact)))));
    disp(L1error(p))
end

%% PLots
% close all;
figure

plot(2:Nt,mvec,LineWidth=1.5)
title("Max over Iteration")


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
loglog(lambdavec(10:end-10),lambdavec(10:end-10)*.0013, 'k--', LineWidth=1.5)
title("4D Algorithm Error - Rank 3 IC",FontSize=13)
xlabel('$\lambda, \Delta t =  \lambda / \Big({1/\mathrm{dx}_1 + 1/\mathrm{dx}_2 + 1/\mathrm{dx}_3 + 1/\mathrm{dx}_4}\Big)$', ...
       'Interpreter','latex', FontSize=13);
ylabel("L1 Norm of Difference from Exact Solution ",FontSize = 13)
legend("Error Using Backward Euler", "Slope 1$", Interpreter ='Latex',Location = 'best',FontSize = 13)

