function V = max_release(s)



% rule from http://www.usbr.gov/mp/cvp//cvp-cas/docs/Draft_Findings/130814_tech_memo_flood_control_purpose_hydrology_methods_results.pdf
%  storage = [0, 100, 400, 600, 1000]
%  release = cfs_to_taf(np.array([0, 35000, 40000, 115000, 115000])) # make the last one 130 for future runs
%
% MG_update on 01/02/2017 - deadpool added for storage < 90 

xs = [0, 90, 100, 400, 600, 1000] ;
xr = cfs_to_taf([0, 0, 35000, 40000, 115000, 115000]) ;

V = interp_lin_scalar(xs, xr, s) ;

end