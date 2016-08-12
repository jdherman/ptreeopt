function v = min_release(s,q,dt)

global sys_param;

% time-varying parameters
sF = flood_buffer(dt); 
wt = sys_param.simulation.w(dt);

if (s < sF)
    v = sys_param.simulation.EnvF ;
elseif (s < 975)
    v = max(0.2*(q + s - sF), 0.5*wt) ;
else
    v = max_release(s) ;
end


end