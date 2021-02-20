function [out, rout, Wi, L2, L1] = ANFIS_eval(in,fs,cons_param)

nSet = size(fs,1);
nIn = size(fs,2);
nRules = nSet^nIn;
x = in(1);
y = in(2);


actv_in = zeros(nSet*nIn,1);

for i = 1:nSet
	actv_in(i) = fs{i,1}(x);
	actv_in(i+nSet) = fs{i,2}(y);
end
L1 = actv_in;

%(Layer 2) T-norm prodotto
act_Tnorm = actv_in(nSet+1:nSet*nIn)*actv_in(1:nSet)';
L2 = act_Tnorm;
%act_Tnorm = L2;
%(Layer3) activation strenght normalizzate 
Wi(1:nRules) = act_Tnorm(1:nRules)/sum(act_Tnorm,'all');

%(Layer 4) calcolo delle uscite secondo TSK (uscita di tipo linare)
p = cons_param(:,1);
q = cons_param(:,2);
r = cons_param(:,3);

rout = Wi'.* (p.*x + q.*y + r);

%(Layer 5 sommo)
out = sum(rout);

end

