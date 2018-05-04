% FORCE_INTERNAL_SPARSE.m
%
% This function generates the sum of 4 sine waves in figure 2D using the architecture of figure 1C (sparse connectivity)
% with the RLS learning rule.  We only implement one loop, otherwise we'd be here all week.  Literally.  This is because
% in the case of sparse connectivity, we don't have the option (and the optimization) to use the same inverse
% correlation matrix for all neurons, as we did n force_internal_all2all.m.  Rather, we'd have to have order(N) NxN
% inverse correlation matrices, and clearly that won't fly.  
%
% So implementing only one loop, the script demonstrates that a separate feedback loop that is not attached to the
% output unit can be trained using the output's error, even though the output and the control unit (the one that feed's
% its signal back into the network) do not share a single pre-synaptic input.  This principle is used heavily in both
% architectures in figure 1B and 1C for the real examples shown in the paper.
%
% written by David Sussillo

disp('Clearing workspace.');
clear;

linewidth = 3;
fontsize = 14;
fontweight = 'bold';

N = 2000;
p = 0.1;
g = 1.5;				% g greater than 1 leads to chaotic networks.
alpha = 1.0;
nsecs = 1440;
dt = 0.1;
learn_every = 2;

scale = 1.0/sqrt(p*N);
M = sprandn(N,N,p)*g*scale;
M = full(M);

nRec2Out = N/2;
nRec2Control = N/2;

% Allow output and control units to start with different ICs.  If you set beta greater than zero, then y will look
% different than z but still drive the network with the appropriate frequency content (because it will be driven with
% z).  A value of beta = 0 shows that the learning rules produce extremely similar signals for both z(t) and y(t),
% despite having no common pre-synaptic inputs.  Keep in mind that the vector norm of the output weights is 0.1-0.2 when
% finished, so if you make beta too big, things will eventually go crazy and learning won't converge.
%beta = 0.1;				
beta = 0.0;
wo = beta*randn(nRec2Out,1)/sqrt(N/2);			% synaptic strengths from internal pool to output unit
dwo = zeros(nRec2Out,1);
wc = beta*randn(nRec2Control, 1)/sqrt(N/2);		% synaptic strengths from internal pool to control unit
dwc = zeros(nRec2Control, 1);

wf = 2.0*(rand(N,1)-0.5);		% the feedback now comes from the control unit as opposed to the output

% Deliberatley set the pre-synaptic neurons to nonoverlapping between the output and control units.
zidxs = 1:round(N/2);			
yidxs = round(N/2)+1:N;			 

disp(['   N: ', num2str(N)]);
disp(['   g: ', num2str(g)]);
disp(['   p: ', num2str(p)]);
disp(['   nRec2Out: ', num2str(nRec2Out)]);
disp(['   nRec2Control: ', num2str(nRec2Control)]);
disp(['   alpha: ', num2str(alpha,3)]);
disp(['   nsecs: ', num2str(nsecs)]);
disp(['   learn_every: ', num2str(learn_every)]);


simtime = 0:dt:nsecs-dt;
simtime_len = length(simtime);
simtime2 = 1*nsecs:dt:2*nsecs-dt;

amp = 1.3;
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
wc_len = zeros(1,simtime_len);
zt = zeros(1,simtime_len);
yt = zeros(1,simtime_len);
zpt = zeros(1,simtime_len);
ypt = zeros(1,simtime_len);

x0 = 0.5*randn(N,1);
z0 = 0.5*randn(1,1);
y0 = 0.5*randn(1,1);

x = x0; 
r = tanh(x);
z = z0; 
y = y0;

figure;
ti = 0;
Pz = (1.0/alpha)*eye(nRec2Out);
Py = (1.0/alpha)*eye(nRec2Control);
for t = simtime
    ti = ti+1;	
    
    if mod(ti, nsecs/2) == 0
	disp(['time: ' num2str(t,3) '.']);
	subplot 211;
	plot(simtime, ft, 'linewidth', linewidth, 'color', 'green');
	hold on;
	plot(simtime, zt, 'linewidth', linewidth, 'color', 'red');
	plot(simtime, yt, 'linewidth', linewidth, 'color', 'magenta'); 
	title('training', 'fontsize', fontsize, 'fontweight', fontweight);
	xlabel('time', 'fontsize', fontsize, 'fontweight', fontweight);
	ylabel('f, z and y', 'fontsize', fontsize, 'fontweight', fontweight);
	legend('f', 'z', 'y');
	hold off;
	
	subplot 212;
	plot(simtime, wo_len, 'linewidth', linewidth); 
	hold on;
	plot(simtime, wc_len, 'linewidth', linewidth, 'color', 'green'); 
	hold off;
	xlabel('time', 'fontsize', fontsize, 'fontweight', fontweight);
	ylabel('|w_o|, |w_c|', 'fontsize', fontsize, 'fontweight', fontweight);
	legend('|w_o|', '|w_c| ');	

	pause(0.5);	
    end
    
    % sim, so x(t) and r(t) are created.
    x = (1.0-dt)*x + M*(r*dt) + wf*(y*dt); % note the y here.
    r = tanh(x);
    rz = r(zidxs);			% the neurons that project to the output
    ry = r(yidxs);			% the neurons that project to the control unit
    z = wo'*rz;
    y = wc'*ry;
    
    if mod(ti, learn_every) == 0
	% update inverse correlation matrix for the output unit
	kz = Pz*rz;
	rPrz = rz'*kz;
	cz = 1.0/(1.0 + rPrz);
	Pz = Pz - kz*(kz'*cz);    
	% update the error for the linear readout
	e = z-ft(ti);
	% update the output weights
	dwo = -e*kz*cz;
	wo = wo + dwo;

	% update inverse correlation matrix for the control unit
	ky = Py*ry;
	rPry = ry'*ky;
	cy = 1.0/(1.0 + rPry);
	Py = Py - ky*(ky'*cy);    

	%%% NOTE WE USE THE OUTPUT'S ERROR %%%
	% update the output weights
	dwc = -e*ky*cy;
	wc = wc + dwc;	
    end
    
    % Store the output of the system.
    zt(ti) = z;
    yt(ti) = y;
    wo_len(ti) = sqrt(wo'*wo);	
    wc_len(ti) = sqrt(wc'*wc);	
end
error_avg = sum(abs(zt-ft))/simtime_len;
disp(['Training MAE: ' num2str(error_avg,3)]);    
disp(['Now testing... please wait.']);    


% Now test. 
ti = 0;
for t = simtime				% don't want to subtract time in indices
    ti = ti+1;    
    
    % sim, so x(t) and r(t) are created.
    x = (1.0-dt)*x + M*(r*dt) + wf*(y*dt); % note the y here.
    r = tanh(x);
    rz = r(zidxs);			% the neurons that project to the output
    ry = r(yidxs);			% the neurons that project to the control unit
    z = wo'*rz;
    y = wc'*ry;
    
    zpt(ti) = z;
    ypt(ti) = y;
end
error_avg = sum(abs(zpt-ft2))/simtime_len;
disp(['Testing MAE: ' num2str(error_avg,3)]);


figure;
subplot 211;
plot(simtime, ft, 'linewidth', linewidth, 'color', 'green');
hold on;
plot(simtime, zt, 'linewidth', linewidth, 'color', 'red');
plot(simtime, yt, 'linewidth', linewidth, 'color', 'magenta'); 
title('training', 'fontsize', fontsize, 'fontweight', fontweight);
xlabel('time', 'fontsize', fontsize, 'fontweight', fontweight);
hold on;
axis tight;
ylabel('f, z and y', 'fontsize', fontsize, 'fontweight', fontweight);
legend('f', 'z', 'y');


subplot 212;
hold on;
plot(simtime2, ft2, 'linewidth', linewidth, 'color', 'green'); 
plot(simtime2, zpt, 'linewidth', linewidth, 'color', 'red');
plot(simtime2, ypt, 'linewidth', linewidth, 'color', 'magenta'); 
axis tight;
title('simulation', 'fontsize', fontsize, 'fontweight', fontweight);
xlabel('time', 'fontsize', fontsize, 'fontweight', fontweight);
ylabel('f, z and y', 'fontsize', fontsize, 'fontweight', fontweight);
legend('f', 'z', 'y');
	

