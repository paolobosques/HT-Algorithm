function [rphat,Bphatmat_nn] = Bupfull(dtn,rahat,rbhat,rcstar,rc1star,rc2star, Fa, Fb, Fc1, Fc2, Ua_n,Ub_n,Uc1_n,Uc2_n, Uahat,Ubhat,Uc1star, Uc2star ,Bc_n, BcstarmatcT,Btilp) 

        %Example case nodes: a = 1, b = 2, p (parent) = 12, c = 34 (c1
        %=3,c2 = 4)
        M1 = speye(rcstar); % r34xr34
        A1 = -dtn*Ubhat'*(Fb*Ubhat);%r2hatxr2hat
        H = speye(rahat); %r1hatxr1hat
        A2 = BcstarmatcT'*kron(speye(rc2star), -dtn* Uc1star'*(Fc1*Uc1star))*BcstarmatcT + BcstarmatcT'*kron(-dtn * Uc2star'*(Fc2*Uc2star), speye(rc1star))*BcstarmatcT; %r34xr34 
        M = speye(rbhat);%r2hatxr2hat
        H3 = speye(rcstar);%r34xr34
        A3 = -dtn*Uahat'*(Fa*Uahat);%r1hatxr1hat
        B =  ttm(Btilp, {Uahat'*Ua_n, Ubhat'* Ub_n, BcstarmatcT' * kron( Uc1star'*Uc1_n, Uc2star'*Uc2_n) * tens2mat(Bc_n,3)'},[1,2,3]); %r1hatxr2hatxr34
        [~,Btilphat_nn] = Simoncini_V2(M1,A1,H,A2,M,H3,A3,B);        
        [~,~,rphat] = size(Btilphat_nn); 
        Btilphatmat = tens2mat(Btilphat_nn,3) ;
        [Bphatmat_nn,~] = qr(Btilphatmat',0);

end