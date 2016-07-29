function [JJ, HH] = run_sdp()


global sys_param;

%-- Initialization --
q    = sys_param.simulation.q;
h_in = sys_param.simulation.h_in;


%-- Run optimization --
policy.H = opt_sdp( tol, max_it );
HH = policy.H;

%-- Run simulation --
[Jflo, Jirr] = simLake( q, h_in, policy );
JJ(1) = Jflo;
JJ(2) = Jirr;

end