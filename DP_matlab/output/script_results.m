%% analysis of optimal policies for Folsom reservoir (CA)
%
% MatteoG 03/02/2017

clear
clc

% load historical data
s_hist = load('../data/storHist_011095_300916.txt','-ascii') ;
r_hist = load('../data/relHist_011095_300916.txt','-ascii') ;

%% DDP results
load workspaceDDP.mat

Ny = length(sys_param.simulation.q)/sys_param.algorithm.T ;
D = repmat( sys_param.simulation.w, Ny, 1);

% historical performance
G_hist = (max( D - r_hist, 0 )).^2 ;
J_hist = mean( G_hist )

figure; 
subplot(211); plot(s(1:end-1), 'r', 'LineWidth',2); 
hold on; plot(s_hist, 'k', 'LineWidth',2); 
legend('simulated DDP', 'observed'); grid on;
axis([1 length(s_hist) 0 1000]); ylabel('storage (TAF)'); 
subplot(212); plot(r(2:end), 'r', 'LineWidth',2); 
hold on; plot(r_hist, 'k',  'LineWidth',2);
hold on; plot(D, 'g:',  'LineWidth',2);
legend('simulated DDP','observed','demand'); grid on;
axis([1 length(r_hist) 0 200]); ylabel('release (TAF/day)'); 


%% save DDP results (storage, rel. decision, release, deficit^2)
X = [s(1:end-1) u(1:end-1) r(2:end) G] ; %
save -ascii resultDDP.txt X

%% SDP results
load workspaceSDP.mat

Ny = length(sys_param.simulation.q)/sys_param.algorithm.T ;
D = repmat( sys_param.simulation.w, Ny, 1);

% historical performance
G_hist = (max( D - r_hist, 0 )).^2 ;
J_hist = mean( G_hist )

figure; 
subplot(211); plot(s(1:end-1), 'r', 'LineWidth',2); 
hold on; plot(s_hist, 'k', 'LineWidth',2); 
legend('simulated DDP', 'observed'); grid on;
axis([1 length(s_hist) 0 1000]); ylabel('storage (TAF)'); 
subplot(212); plot(r(2:end), 'r', 'LineWidth',2); 
hold on; plot(r_hist, 'k',  'LineWidth',2);
hold on; plot(D, 'g:',  'LineWidth',2);
legend('simulated DDP','observed','demand'); grid on;
axis([1 length(r_hist) 0 200]); ylabel('release (TAF/day)'); 

%% save SDP results (storage, rel. decision, release, deficit^2)
X = [s(1:end-1) u(1:end-1) r(2:end) G] ; %
save -ascii resultSDP.txt X

%% comparison
load -ascii resultDDP.txt 
load -ascii resultSDP.txt 
load -ascii demand_sim.txt

figure; plot(s_hist, 'Color', [.5 .5 .5], 'LineWidth',2); 
hold on; plot( resultDDP(:,1), 'r', 'LineWidth',2); 
hold on; plot( resultSDP(:,1), 'k', 'LineWidth',2); 
axis([1 length(s_hist) 0 1000]); ylabel('storage (TAF)'); 
legend('observed', 'DDP', 'SDP'); grid on;

G_hist = (max( demand_sim - r_hist, 0 )).^2 ;
J_hist = mean( G_hist )
J_ddp = mean( resultDDP(:,end) )
J_sdp = mean( resultSDP(:,end) )
