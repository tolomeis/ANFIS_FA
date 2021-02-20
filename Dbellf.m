function [Da, Db, Dc] = Dbellf(x,a,b,c)

% Dx = 2*b*abs((x-c)/a)^(2*b-1)...
% 	/ ...
% 	(a*(1 + abs((x-c)/a)^(2*b))^2);
dist = ((x-c)^2)/(a^2);

Da =  2*b * dist^b...
	/ ...
	(a * (1 + dist^b )^2 );


Db = -  dist^b * log(dist)...
	/ ...
	((1 + dist^b )^2);

Dc = 2*b* dist^b...
	/...
	( (x-c) * ( dist^b + 1 )^2 );

end
