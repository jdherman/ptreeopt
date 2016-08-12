

% SDP implemenation to design the optimal operating policy for Folsom
% (adapted from M3O toolbox)


clear all;
clc

cd('/Users/matteo/Poli/research/myCode/GIT/ptreeopt/DP_matlab')
global sys_param;

addpath('sim')
addpath('SDP')

%% Set general system parameters

qq  = load('./data/q_inflow.txt','-ascii') ; % 01/10/1904 - 30/09/2015
ee  = load('./data/evaporation.txt','-ascii') ; % 01/10/1904 - 30/09/2015
w   = load('./data/demand.txt','-ascii') ;

dn = datenum(1904,10,1):datenum(2015,09,30);
dn_sim = datenum(1995,10,1):datenum(2015,09,30); % simulation horizon 01/10/1995 - 30/09/2015
id1 = find( dn == dn_sim(1) );
id2 = find( dn == dn_sim(end) );
qqs = qq(id1:id2);
% remove 29/feb
dv = datevec(dn_sim');
feb29 = dv(:,2).*dv(:,3) == 2*29;
q_ = qqs(~feb29) ;
sys_param.simulation.ev = ee(~feb29) ;
% remove negative value
q_(q_<0)= eps ;
sys_param.simulation.q = q_ ;

% structure with system paramenters
sys_param.simulation.s_in  = 458.6 ; % initial storage on 1/10/1995
sys_param.simulation.w     = [w(1:151); w(153:end)]  ; % remove 29feb
sys_param.simulation.EnvF  = 0  ;  % environmnetal flow (not considered)
sys_param.simulation.delta = 1; % daily time-step



%% set SDP optimization

% discretization of variables' domain (Hoa Binh case: ns=68, nu=17, nq=101)
grids.discr_s = [0:20:1000]' ; % taf
grids.discr_u = [[0:.25:8],9,[10:20:cfs_to_taf(115000)]]' ; % taf/day 
grids.discr_q = [[0:50],[55:10:155]]' ; % taf/day
sys_param.algorithm = grids;
%[vv, VV] = construct_rel_matrices(grids); % compute daily minimum/maximum release matrixes
load ./data/minRel_table
load ./data/maxRel_table
sys_param.algorithm.min_rel = vv;
sys_param.algorithm.max_rel = VV;

% set algorithm parameters
sys_param.algorithm.name = 'sdp';
sys_param.algorithm.Hend = 0 ; % penalty set to 0
sys_param.algorithm.T = 365 ;    % the period is equal 1 as we assume stationary conditions
sys_param.algorithm.gamma = 1; % set future discount factor
tol = -1;    % accuracy level for termination 
max_it = 10; % maximum iteration for termination 

% Estimate cyclostationary pdf (assuming log-normal distribution)
T = sys_param.algorithm.T ;
Ny = length(q_)/T;
Q = reshape( sys_param.simulation.q, T, Ny );
sys_param.algorithm.q_stat = nan(T,2) ;
for i = 1:365
    qi = Q(i,:) ;
    sys_param.algorithm.q_stat(i,:) = lognfit(qi);
end

%% run SDP optimization
policy.H = opt_sdp(tol, max_it) ;
%policy.H   = load('tempBellman.txt','-ascii') ;
%% run simulation of SDP-policies (01/10/1995-30/09/2015)
q_sim = sys_param.simulation.q ;
s_init = sys_param.simulation.s_in ;
[J, s,u,r, G] = simLake( q_sim, s_init, policy );
disp(J)

figure; 
subplot(211); plot(s, 'LineWidth',2); 
axis([1 7300 0 1000]); ylabel('storage (TAF)');
subplot(212); plot(r, 'LineWidth',2); 
hold on; plot(repmat( sys_param.simulation.w, Ny, 1), 'r:',  'LineWidth',2);
axis([1 7300 0 200]); ylabel('release (TAF/day)'); legend('release','demand');

%% save results (storage, rel. decision, release, deficit^2)
X = [s u r [nan;G]] ; %
save -ascii ./fig/resultSDP.txt X
