function [V1hat,V2hat,V3hat,V4hat,rhat1,rhat2,rhat3,rhat4] = redaugk4(K1new,K2new,K3new,K4new,V1n,V2n,V3n,V4n)

    %Reduced augmentation allowing for varying ranks for 4D problems
    %(applied to ksteps)

    Knews = {K1new,K2new,K3new,K4new};
    Vns = {V1n,V2n,V3n,V4n};
    rhats = zeros(1,length(Knews));
    Vhats = cell(1,length(Knews));

    
    for j = 1:length(Knews)
        [Vddag,~] = qr(Knews{j},0); %QR each k
        Vaug = [Vddag,Vns{j}];
        [Q,R] = qr(Vaug,0);
        [U,s,~] = svd(full(R),0);
        s = diag(s); %Takes diagonal and puts into vector
        stil = s(s > 1.0e-12); % For matlab e-13 machine db precision
        rhats(j) = numel(stil);
        Util = U(:,1:rhats(j));
        Vhats{j} = Q*Util;
    end

    rhat1 = rhats(1);
    rhat2 = rhats(2);
    rhat3 = rhats(3);
    rhat4 = rhats(4);

    V1hat = Vhats{1};
    V2hat = Vhats{2};
    V3hat = Vhats{3};
    V4hat = Vhats{4};
  
end