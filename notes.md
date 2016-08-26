
See lit review in Yang et al. 2016 WRR submission for decision tree (fitting ML) in reservoir ops. Starts with Bessler et al. 2003.

Also see their calculation (in the first paper) of Gini coefficients for each feature variable. The same could apply to ptreeopt.


# FIRST EXPERIMENT
- Historical 1995-2015 period. TDI, day, storage as features; Actions are release demand, hedge 50/80, and flood control. 50 random seeds, tree depths from 2-10, 20,000 NFE.
- This took 12 hours to run using 50 processors in parallel on HPC1.
- The purpose of this is to explore the algorithmic properties and be confident that it actually works.

# SECOND EXPERIMENT
- Climate change scenarios. First what does data look like?


figure 2 folsom map:
- Also add a flow duration curve for years 1955-2015 (C) and subplot (D) which shows the daily releases historically and the fitted sine wave demand. Explain that this is illustrative.

Figure 3: GP operations: crossover, mutation, pruning/collapse. Illustrate with some made-up example trees. Include depth-first list representations in the figure. 

Figure (4?) on algorithmic performance:
- A: J vs. NFE. Need DDP/SDP single performance values. Plot J as a ratio of this.
- B: Boxplot of final (NFE=20000) objective function values, as a ratio, with one box for each tree depth (show that there is not much difference). This is the same as the validation plot -- is there a point to this? Want to show that smaller/simpler trees are better.
- C: Tree similarity measure compared to ending "best" one, for all different depths (show that larger depths have more variability in the resulting policy, something like "equifinality". This exists for MOEAs too.)
- D: Validation with tree depth. For this, we need synthetic scenarios (30/50) and run DDP on each of them to find the performance ratio. No new optimization, just re-evaluate on the synthetic scenarios.