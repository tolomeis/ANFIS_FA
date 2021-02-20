lambdaV = logspace(2,4,50);
pefor = zeros(100,1);



for il = 1:100
	lambda = lambdaV(il);
	ANFIS_main;
	perfor(il) = err;

end

figure(2)
plot(1:50,perfor(1:50));
hold on
grid on
