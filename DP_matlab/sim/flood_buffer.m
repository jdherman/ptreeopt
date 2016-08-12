function y = flood_buffer( d )

 tp = [1, 50, 151, 200, 243, 365] ;
 sp = [975, 400, 400, 750, 975, 975] ;
 
 y = interp_lin_scalar( tp, sp, d ) ;