function [ H , idx_u ] = Bellman_sdp( H_ , s_curr )

global sys_param;

%-- Initialization --
discr_s = sys_param.algorithm.discr_s;
discr_u = sys_param.algorithm.discr_u;
discr_q = sys_param.algorithm.discr_q;
n_u = length(sys_param.algorithm.discr_u);
n_q = length(sys_param.algorithm.discr_q);
gamma   = sys_param.algorithm.gamma;
delta = sys_param.simulation.delta;

VV = repmat(sys_param.simulation.VV', 1 , n_u);
vv = repmat(sys_param.simulation.vv', 1 , n_u);

mi_q    = sys_param.algorithm.stat_t(1); % mean of log(disturbance)
sigma_q = sys_param.algorithm.stat_t(2); % std of log(disturbance)

%-- Calculate actual release contrained by min/max release rate --
uu = repmat(discr_u', n_q, 1);
R = min( VV , max( vv , uu ) );

%==========================================================================
% Calculate the state transition; TO BE ADAPTAED ACCORDING TO
% YOUR OWN CASE STUDY

qq = repmat( discr_q, 1, n_u );
s_next = s_curr + delta*( qq - R );

%==========================================================================
% Compute immediate costs and aggregated one; TO BE ADAPTAED ACCORDING TO
% YOUR OWN CASE STUDY
G = (max( sys_param.simulation.wt - R, 0 )).^2 ;

%-- Compute cost-to-go given by Bellman function --
H_ = interp1q( discr_s , H_ , s_next(:) ) ;
H_ = reshape( H_, n_q, n_u )         ;

%-- Compute resolution of Bellman value function --
% compute the probability of occourence of inflow that falls within the
% each bin of descritized inflow level
cdf_q      = logncdf( discr_q , mi_q , sigma_q );  
p_q        = diff(cdf_q);                          
p_diff_ini = 1-sum(p_q);                           
p_diff     = [ p_diff_ini ; p_q];                 

Q     = (G + gamma.*H_)'*p_diff ;
H     = min(Q)                  ;
sens  = eps                     ;
idx_u = find( Q <= H + sens )   ;

end