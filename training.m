%% HYBRID LEARNING
% dati x,y,pts

nit = 100;
st = 20;

fig = figure(1);
fig.WindowState = 'maximized';
tiledlayout(3,3)
nexttile([2 3]);
surf(x,y,f(x,y),f(x,y),'FaceAlpha',0.4)
nexttile(9)
h = animatedline('Marker','o');
h2 = animatedline('Marker','*','Color','red');
figure(1)

t = f(x,y);
prec_err = 100;



for kit = 1:nit
	tic;
	%% FORWARD STEP
	Wi = zeros(nPoints, nRules);
	out_fwstep = zeros(nPoints,1);

	for i = 1:nPoints
		[out_fwstep(i), ~, pes] = ANFIS_eval([Xt(i) Yt(i)],inSet,cons_param);
		Wi(i,:) = pes;
	end
	
	A = [];
	for i = 1:nRules
		A = [A, [Xt.*Wi(:,i), Yt.*Wi(:,i), Wi(:,i)]];
	end
	
	outvect = pts;

	X = zeros(nRules*3, 1);
	S = lambda*eye(nRules*3);
	for i = 1:nPoints
		S = S - (S*A(i,:)'*A(i,:)*S)/(1+A(i,:)*S*A(i,:)');
		X = X + S*A(i,:)'*(outvect(i)-A(i,:)*X);
	end
	
	%PARAM = inv(A'*A)*A'*outvect;
	ptemp = zeros(3,nRules);
	ptemp(1:nRules*3) = X;
	cons_param = ptemp';
	
	%% PLOT
	%Plotto punti
	z = zeros(size(x,1), size(x,2));

	for i = 1:np
		z(i) = ANFIS_eval([x(i), y(i)], inSet, cons_param);
	end
	err = sum((z(1:np)-t(1:np)).^2)/np;
	
	
	% Errore su training
	zt = zeros(nPoints,1);
	for i = 1:nPoints
		zt(i) = ANFIS_eval([Xt(i), Yt(i)], inSet, cons_param);
	end
	err_train = sum((zt(1:nPoints)-pts(1:nPoints)).^2)/nPoints;
	
	%% BACKWARD STEP
	
	for i = 1:nPoints
		[out, wf, Wi, L2, L1] = ANFIS_eval([Xt(i) Yt(i)],inSet,cons_param);
		%Parametri conseguenti (di uscita) aggiornati per un singolo punto di
		%training. Calcolo il nuovo errore e aggiorno gli antecedenti
% 		out = out_fwstep(i);
		er = pts(i) - out; %delta layer 5
		delta5 = er;
		
		p = cons_param(:,1);
		q = cons_param(:,2);
		r = cons_param(:,3);
		delta4 =  (p*Xt(i) + q*Yt(i) + r).*delta5;  %deltaLayer4 = derivata della funz. di layer 4 risp a Wi
		
		%layer 3 (normalizza), un neurone ha ingresso il W non normalizzato
		delta3 = zeros(nRules, nRules);
		%delta3(rispetto_a_in_k,neurone_j)
		for j = 1:nRules
			delta3(:,j) = L2(j)/sum(L2, 'all')^2;
			delta3(j,j) = ((sum(L2, 'all') - L2(j))/sum(L2, 'all')^2);
			delta3(:,j) = delta3(:,j)*delta4(j);
		end

		
		
		%L2 moltiplica due a due

		delta2 = zeros(nRules,2);
		for j = 1:nRules
			delta2(j,1) = L1(floor((j-1)/nSet) +1);
			delta2(j,2) = L1(mod(j-1,nSet)+1);
		end
		delta2(:,1) = delta2(:,1).*sum(delta3,2);
		delta2(:,2) = delta2(:,2).*sum(delta3,2);


		delta1 =zeros(nSet*nIn, 3);

		a = prem_param(:,1:2);
		b = prem_param(:,3:4);
		c = prem_param(:,5:6);
		for j = 1:nSet
			delta1(j,:) = Dbellf(Xt(i),a(j,1),b(j,1),c(j,1)); %*Xt(i);			
% 			delta1(j,:) = delta1(j,:).*sum(delta2(1+(j-1)*nSet : j*nSet, 1));	
			
			delta1(j+nSet,:) = Dbellf(Yt(i),a(j,2),b(j,2),c(j,2));%*Yt(i);
% 			delta1(j+nSet,:) = delta1(j+nSet,:).*sum(delta2(j:nSet:(nSet^nIn - nSet + j),2));
		end
		delta1(1,:) = delta1(1,:).*(delta2(1,1) + delta2(2,1));
		delta1(2,:) = delta1(2,:).*(delta2(3,1) + delta2(4,1));
		delta1(3,:) = delta1(3,:).*(delta2(1,2) + delta2(3,2));
		delta1(4,:) = delta1(4,:).*(delta2(2,2) + delta2(4,2));
		
		etanorm = [etaa etab etac]./sqrt(sum(delta1.^2));

		%aqg parametri
		for j = 1:nSet*nIn
% 			deltaNorm = delta1(j,:)/norm(delta1(j,:));
			a(j) = a(j) + etanorm(1)*delta1(j,1);
			b(j) = b(j) + etanorm(2)*delta1(j,2);
			c(j) = c(j) + etanorm(3)*delta1(j,3);
		end
		prem_param = [a b c];
	end


	%% plot
	%rigerero FS
	inSet = cell(nSet,nIn);
	
	a = prem_param(:,1:2);
	b = prem_param(:,3:4);
	c = prem_param(:,5:6);
	for j = 1:nIn
		nexttile(j+6);
		hold on; grid on
		unplot(nSet)
		for i = 1:nSet
			inSet{i,j} = @(t)bellf(t,a(i,j),b(i,j),c(i,j));
			plot(in(:,j), arrayfun( inSet{i,j},in(:,j)));
		end
		drawnow
		title(['Fuzzy set modificati ingresso ' num2str(j)])
	end
	
	zt = zeros(nPoints,1);
	for i = 1:nPoints
		zt(i) = ANFIS_eval([Xt(i), Yt(i)], inSet, cons_param);
	end
	err_train_b = sum((zt(1:nPoints)-pts(1:nPoints)).^2)/nPoints;
	
	disp(num2str(err_train_b- err_train));
	
	nexttile(9)
	hold on;
	addpoints(h,kit,err);
	addpoints(h2,kit,err_train);
	grid on
	title('Andamento MSE');
	drawnow
	
	figure(1)
	nexttile(1, [2 3]);
	if kit > 1
		unplot(1)
	end
	hold on
	surf(x,y,z)
	axis([-1.5  1.5	-1.5 1 -4 5]);
	title("Iteration "+kit+" out of "+nit);
	drawnow

	%% 	end
	
	if(kit == 1) prec_err = err;
	else
		if(abs(err - prec_err) < deltae_stp) && (kit > 5)
			break
		else
			prec_err = (prec_err*0.5 + 1.5*err)/2;
		end
	end
	
end
% figure(1)
% nexttile([2 3]);
%t = f(x,y);
er = sum((z(1:np)-t(1:np)).^2)/np;
title("MSE = " + num2str(er)+" in "+num2str(kit)+" epoche");
