The SDP/DDP runs were using the historical TOCS rule. 

```
In attachment some figures and the trajectories produced by DDP and SDP, organized as (storage, rel. decision, release, deficit^2). for storage and release you have to consider from 1 to end-1, while release and deficit from 2 to end (this is consistent with our notation from Rodolfo-Andrea book).
```

SDP performance: 0.22 (TAF/d)^2

DDP performance: 0.06 (TAF/d)^2

To compare, must require the TOCS rule to hold. Otherwise the policy tree will do better, there is too much flexibility in the flood buffer.


# new update Feb 2017
the value of J (average daily deficit^2) is as follows:
J_hist = 0.3442
J_sdp = 0.3087
J_ddp = 0.1270

columns: storage, target_rel, actual_rel, deficit^2