%% PARAMETRI DI INIT
clc; clear all;
close all;
addpath('functs')

% FUNZIONI
f = @(x,y)1*x.^3 - 2*x.*y.^2; % Sella
% f = @(x,y)10*x.*exp(-x.^2-y.^2);
% f = @(x,y)2*sin(pi*x) + 2*cos(pi*y); % dossi
% f = @(X,Y) pi*Y.*sin(pi*X) - pi*X.*cos(pi*Y);

nPoints = 400;		% numero campioni per training
sigma = 0.2;		% Sigma rumore sul campionamento

nSet = 2;			% numero di fuzzy set in cui suddividere un ingresso
etaa = 0.001;		% learning rate backprop
etab = 0.001;
etac = 0.0005;

lambda = 0.5e2;		% parametro inizializzazione min quadrati ricorsivi
nit = 100;			% numero iterazioni massime
er_stop = 0.1;		% MSE di stop
deltae_stp = 1e-4;	% Delta Errore di stop

%% Plot funzione
[x, y] = meshgrid(-1.5:0.05:1.5);
np = size(x,1)^2;

Xt = (rand(nPoints,1)-0.5)*3;
Yt = (rand(nPoints,1)-0.5)*3;
pts = f(Xt,Yt) + sigma*randn([nPoints,1]);


figure(1)

surf(x,y,f(x,y),f(x,y))
hold on
plot3(Xt, Yt, pts,'r*')

grid on

%% Init ANFIS
nIn = 2;		%n di ingressi
in = [-1.5:0.05:1.5; -1.5:0.05:1.5]';
nRules = nSet^(nIn);

% Parametri globali per ogni ingresso
inMax = [max(x,[],'all') max(y,[],'all')];
inMin = [min(x,[],'all') min(y,[],'all')];
deltaIn = 1/(nSet-1) * (inMax - inMin);

% Parametri di ogni fuzzy set
a = 0.5*repmat(deltaIn,[nSet 1]);	% metà della larghezza del set
b = 2*repmat(deltaIn,[nSet 1]);		% pendenza, con a==b ho pendenza 2
c = zeros(nSet, nIn);				% centro della campana

inSet = cell(nSet,nIn);

for j = 1:nIn
	figure(j+1); hold on; grid on
	for i = 1:nSet
		c(i,j) = inMin(j) + (i-1)*deltaIn(j);
		inSet{i,j} = @(t)bellf(t,a(i,j),b(i,j),c(i,j));
		plot(in(:,j), arrayfun( inSet{i,j},in(:,j)));
	end
	title(['Fuzzy set per ingresso ' num2str(j)])
end
prem_param = [a b c];


%Funzione retta, 2 ingressi x e y, parametri p,q,r
p = 0.5*rand(nRules,1);
q = 0.5*rand(nRules,1);
r = 0.5*rand(nRules,1);
cons_param = [p q r];

%% Training e valutazione
close all
training;
plot3(Xt, Yt, pts,'r*');



