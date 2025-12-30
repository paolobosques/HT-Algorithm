function [rphat,Bphatmat_nn] = Bup2l(dtn,rahat,rbhat,rcstar,rc2star, Fa, Fc1, Ua_n,Ub_n,Uc1_n,Uc2_n, Uahat,Ubhat,Uc1star, Uc2star ,Bc_n, BcstarmatcT,Btilp) 

% Combined core update for 2 laplacian 4d diffusion eqn

        M1 = speye(rcstar); % r34xr34
        A1 = speye(rbhat); % r2hatxr2hat
        H = speye(rahat); %r1hatxr1hat
        A2 = BcstarmatcT'*kron(speye(rc2star), -dtn* Uc1star'*(Fc1*Uc1star))*BcstarmatcT; %r34xr34
        M = speye(rbhat);%r2hatxr2hat
        H3 = speye(rcstar);%r34xr34
        A3 = -dtn*Uahat'*(Fa*Uahat);%r1hatxr1hat
        B =  ttm(Btilp, {Uahat'*Ua_n, Ubhat'* Ub_n, BcstarmatcT' * kron( Uc1star'*Uc1_n, Uc2star'*Uc2_n) * tens2mat(Bc_n,3)'},[1,2,3]); %r1hatxr2hatxr34
        [~,Btilphat_nn] = Simoncini_V2(M1,A1,H,A2,M,H3,A3,B);        
        [~,~,rphat] = size(Btilphat_nn); %this sets r12hat = r34_star (assuming r34_star < r1hat*r2hat)
        Btilphatmat = tens2mat(Btilphat_nn,3) ;
        [Bphatmat_nn,~] = qr(Btilphatmat',0);
end