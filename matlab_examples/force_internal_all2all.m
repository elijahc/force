% FORCE_INTERNAL_ALL2ALL.m
%
% This function generates the sum of 4 sine waves in figure 2D using the arcitecture of figure 1C (all-to-all
% connectivity) with the RLS learning rule.  The all-2-all connectivity allows a large optimization, in that we can
% maintain a single inverse correlation matrix for the network.  It's also not as a hard a learning problem as internal
% learning with sparse connectivity because there are no approximations of the eigenvectors of the correlation matrices,
% as there would be if this was sparse internal learning.  Note that there is no longer a feedback loop from the output
% unit.
%
% written by David Sussillo

disp('Clearing workspace.');
clear;

linewidth = 3;
fontsize = 14;
fontweight = 'bold';

N = 1000;
p = 1.0;
g = 1.5;				% g greater than 1 leads to chaotic networks.
alpha = 1.0e-0;
nsecs = 1440;
dt = 0.1;
learn_every = 2;

scale = 1.0/sqrt(p*N);
M = randn(N,N)*g*scale;

nRec2Out = N;
wo = zeros(nRec2Out,1);
dw = zeros(nRec2Out,1);

disp(['   N: ', num2str(N)]);
disp(['   g: ', num2str(g)]);
disp(['   p: ', num2str(p)]);
disp(['   nRec2Out: ', num2str(nRec2Out)]);
disp(['   alpha: ', num2str(alpha,3)]);
disp(['   nsecs: ', num2str(nsecs)]);
disp(['   learn_every: ', num2str(learn_every)]);


simtime = 0:dt:nsecs-dt;
simtime_len = length(simtime);
simtime2 = 1*nsecs:dt:2*nsecs-dt;

amp = 0.7;				% amp lower because wf is implied as all ones, which is half the strength of wf.
freq = 1/60;
ft = (amp/1.0)*sin(1.0*pi*freq*simtime) + ...
     (amp/2.0)*sin(2.0*pi*freq*simtime) + ...
     (amp/6.0)*sin(3.0*pi*freq*simtime) + ...
     (amp/3.0)*sin(4.0*pi*freq*simtime);
ft = ft/1.5;

ft2 = (amp/1.0)*sin(1.0*pi*freq*simtime2) + ...
      (amp/2.0)*sin(2.0*pi*freq*simtime2) + ...
      (amp/6.0)*sin(3.0*pi*freq*simtime2) + ...
      (amp/3.0)*sin(4.0*pi*freq*simtime2);
ft2 = ft2/1.5;


wo_len = zeros(1,simtime_len);    
M1_len = zeros(1,simtime_len);
rPr_len = zeros(1,simtime_len);
zt = zeros(1,simtime_len);
zpt = zeros(1,simtime_len);

x0 = 0.5*randn(N,1);
z0 = 0.5*randn(1,1);


x = x0; 
r = tanh(x);
xp = x0;
z = z0; 

figure;
ti = 0;
P = (1.0/alpha)*eye(nRec2Out);
for t = simtime
    ti = ti+1;	
    
    if mod(ti, nsecs/2) == 0
	disp(['time: ' num2str(t,3) '.']);
	subplot 211;
	plot(simtime, ft, 'linewidth', linewidth, 'color', 'green');
	hold on;
	plot(simtime, zt, 'linewidth', linewidth, 'color', 'red');
	title('training', 'fontsize', fontsize, 'fontweight', fontweight);
	legend('f', 'z');	
	xlabel('time', 'fontsize', fontsize, 'fontweight', fontweight);
	ylabel('f and z', 'fontsize', fontsize, 'fontweight', fontweight);
	hold off;
	
	subplot 212;
	plot(simtime, wo_len, 'linewidth', linewidth);
	xlabel('time', 'fontsize', fontsize, 'fontweight', fontweight);
	ylabel('|w|', 'fontsize', fontsize, 'fontweight', fontweight);
	legend('|w|');
	pause(0.5);	
    end
    
    % sim, so x(t) and r(t) are created.
    x = (1.0-dt)*x + M*(r*dt);
    r = tanh(x);
    z = wo'*r;
    
    if mod(ti, learn_every) == 0
	% update inverse correlation matrix
	k = P*r;
	rPr = r'*k;
	c = 1.0/(1.0 + rPr);
	P = P - k*(k'*c);
	
	% update the error for the linear readout
	e = z-ft(ti);
	
	% update the output weights
	dw = -e*k*c;
	wo = wo + dw;
	
	% update the internal weight matrix using the output's error
	M = M + repmat(dw', N, 1);
    end

    % Store the output of the system.
    zt(ti) = z;
    wo_len(ti) = sqrt(wo'*wo);	
end
error_avg = sum(abs(zt-ft))/simtime_len;
disp(['Training MAE: ' num2str(error_avg,3)]);
disp(['Now testing... please wait.']);    

% Now test. 
ti = 0;
for t = simtime				% don't want to subtract time in indices 
    ti = ti+1;    
    
    % sim, so x(t) and r(t) are created.
    x = (1.0-dt)*x + M*(r*dt);
    r = tanh(x);
    z = wo'*r;

    zpt(ti) = z;
end
error_avg = sum(abs(zpt-ft2))/simtime_len;
disp(['Testing MAE: ' num2str(error_avg,3)]);


figure;
subplot 211;
plot(simtime, ft, 'linewidth', linewidth, 'color', 'green');
hold on;
plot(simtime, zt, 'linewidth', linewidth, 'color', 'red');
title('training', 'fontsize', fontsize, 'fontweight', fontweight);
xlabel('time', 'fontsize', fontsize, 'fontweight', fontweight);
hold on;
ylabel('f and z', 'fontsize', fontsize, 'fontweight', fontweight);
legend('f', 'z');


subplot 212;
hold on;
plot(simtime2, ft2, 'linewidth', linewidth, 'color', 'green'); 
axis tight;
plot(simtime2, zpt, 'linewidth', linewidth, 'color', 'red');
axis tight;
title('simulation', 'fontsize', fontsize, 'fontweight', fontweight);
xlabel('time', 'fontsize', fontsize, 'fontweight', fontweight);
ylabel('f and z', 'fontsize', fontsize, 'fontweight', fontweight);
legend('f', 'z');
	

