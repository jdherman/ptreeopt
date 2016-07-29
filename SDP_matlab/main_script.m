

% SDP implemenation to design the optimal operating policy for Folsom
% (adapted from M3O toolbox)


clear all;
clc

cd('/Users/matteo/Poli/research/myCode/GIT/ptreeopt/SDP_matlab')
global sys_param;

addpath('sim')

%% Configure general system parameters

qq     = load('./data/q_inflow.txt','-ascii') ; % 01/10/1904 - 30/09/2015
ee  = load('./data/evaporation.txt','-ascii') ; % 01/10/1904 - 30/09/2015

% remove 29/feb
dn = datenum(1904,10,1):datenum(2015,09,30);
dv = datevec(dn');
feb29 = dv(:,2).*dv(:,3) == 2*29;
q_ = qq(~feb29) ;
sys_param.simulation.ev = ee(~feb29) ;
% remove negative value
q_(q_<0)= eps ;
sys_param.simulation.q = q_ ;

% structure with system paramenters
sys_param.simulation.h_in  = 30 ; % initial level ??
sys_param.simulation.w     = 50*ones(365,1) ; % irrigation demand ??
sys_param.simulation.A     = 100 ; % surface (maybe useless) ??
sys_param.simulation.EnvF  = 5   ;  % environmnetal flow?? 
sys_param.simulation.delta = 1; % 60*60*24;



%% run SDP optimization

% discretization of variables' domain
grids.discr_s = [0:100:1000]' ;
grids.discr_u = [0:100:1000]' ;
grids.discr_q = [0:450]' ;
sys_param.algorithm = grids;
[vv, VV] = construct_rel_matrices(grids); % compute daily minimum/maximum release matrixes
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

policy.H = opt_sdp(tol, max_it) ;

%% run simulation of SDP-policies
q_sim = sys_param.simulation.q(1:365*5) ;
J = simLake( q, h_in, policy );

