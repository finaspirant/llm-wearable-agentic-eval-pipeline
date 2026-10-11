# Author ratings vs. judge panel -- analysis

seed=20261002  n_resamples=10000
n_cases=20  by_family={'bargaining': 7, 'negotiation': 7, 'persuasion': 6}

## valuation_reasoning

### a) Spearman rho vs. author (+ 95% bootstrap CI)

| scope | target | n | rho | ci_low | ci_high | n_valid_reps |
|---|---|---|---|---|---|---|
| overall | GameTheoreticRigor | 20 | 0.7101 | 0.4745 | 0.8749 | 10000/10000 |
| overall | LiteralGroundedness | 20 | 0.3078 | 0.0658 | 0.5808 | 9575/10000 |
| overall | OpponentResponsiveness | 20 | 0.6710 | 0.3641 | 0.8744 | 10000/10000 |
| overall | judge_mean | 20 | 0.6847 | 0.4296 | 0.8424 | 10000/10000 |
| bargaining | GameTheoreticRigor | 7 | 0.3333 | -0.7500 | 0.9661 | 9605/10000 |
| bargaining | LiteralGroundedness | 7 | -0.1739 | -0.7500 | 0.5292 | 8822/10000 |
| bargaining | OpponentResponsiveness | 7 | 0.2415 | -0.5477 | 0.8367 | 9548/10000 |
| bargaining | judge_mean | 7 | 0.2412 | -0.8126 | 0.9165 | 9792/10000 |
| negotiation | GameTheoreticRigor | 7 | undefined (constant) | n/a | n/a | 0/10000 |
| negotiation | LiteralGroundedness | 7 | undefined (constant) | n/a | n/a | 0/10000 |
| negotiation | OpponentResponsiveness | 7 | 0.6831 | 0.6614 | 1.0000 | 6591/10000 |
| negotiation | judge_mean | 7 | 0.6831 | 0.6614 | 1.0000 | 6547/10000 |
| persuasion | GameTheoreticRigor | 6 | 0.5270 | -0.4472 | 1.0000 | 9543/10000 |
| persuasion | LiteralGroundedness | 6 | 0.4472 | 0.2000 | 1.0000 | 6531/10000 |
| persuasion | OpponentResponsiveness | 6 | 0.5270 | -0.4472 | 1.0000 | 9535/10000 |
| persuasion | judge_mean | 6 | 0.5185 | -0.4472 | 1.0000 | 9560/10000 |

### b) Exact / within-1 agreement vs. author (rounded)

| scope | target | n | exact | within_1 |
|---|---|---|---|---|
| overall | GameTheoreticRigor | 20 | 0.4500 | 0.8500 |
| overall | LiteralGroundedness | 20 | 0.2000 | 0.6500 |
| overall | OpponentResponsiveness | 20 | 0.3500 | 0.9000 |
| overall | judge_mean | 20 | 0.3500 | 0.8000 |
| bargaining | GameTheoreticRigor | 7 | 0.5714 | 0.8571 |
| bargaining | LiteralGroundedness | 7 | 0.2857 | 0.4286 |
| bargaining | OpponentResponsiveness | 7 | 0.2857 | 0.8571 |
| bargaining | judge_mean | 7 | 0.4286 | 0.7143 |
| negotiation | GameTheoreticRigor | 7 | 0.2857 | 0.8571 |
| negotiation | LiteralGroundedness | 7 | 0.2857 | 0.8571 |
| negotiation | OpponentResponsiveness | 7 | 0.2857 | 1.0000 |
| negotiation | judge_mean | 7 | 0.2857 | 0.8571 |
| persuasion | GameTheoreticRigor | 6 | 0.5000 | 0.8333 |
| persuasion | LiteralGroundedness | 6 | 0.0000 | 0.6667 |
| persuasion | OpponentResponsiveness | 6 | 0.5000 | 0.8333 |
| persuasion | judge_mean | 6 | 0.3333 | 0.8333 |

### c) Mean offset (author - target), raw (+ 95% bootstrap CI)

| scope | target | n | mean_offset | ci_low | ci_high | n_valid_reps |
|---|---|---|---|---|---|---|
| overall | GameTheoreticRigor | 20 | -0.7000 | -1.0000 | -0.4000 | 10000/10000 |
| overall | LiteralGroundedness | 20 | -1.2500 | -1.6500 | -0.9000 | 10000/10000 |
| overall | OpponentResponsiveness | 20 | -0.6500 | -0.9500 | -0.3500 | 10000/10000 |
| overall | judge_mean | 20 | -0.8667 | -1.1667 | -0.5667 | 10000/10000 |
| bargaining | GameTheoreticRigor | 7 | -0.5714 | -1.1429 | -0.1429 | 10000/10000 |
| bargaining | LiteralGroundedness | 7 | -1.5714 | -2.4286 | -0.7143 | 10000/10000 |
| bargaining | OpponentResponsiveness | 7 | -0.5714 | -1.2857 | 0.1429 | 10000/10000 |
| bargaining | judge_mean | 7 | -0.9048 | -1.5714 | -0.2857 | 10000/10000 |
| negotiation | GameTheoreticRigor | 7 | -0.8571 | -1.2857 | -0.4286 | 10000/10000 |
| negotiation | LiteralGroundedness | 7 | -0.8571 | -1.2857 | -0.4286 | 10000/10000 |
| negotiation | OpponentResponsiveness | 7 | -0.7143 | -1.0000 | -0.4286 | 10000/10000 |
| negotiation | judge_mean | 7 | -0.8095 | -1.1905 | -0.4286 | 10000/10000 |
| persuasion | GameTheoreticRigor | 6 | -0.6667 | -1.3333 | -0.1667 | 10000/10000 |
| persuasion | LiteralGroundedness | 6 | -1.3333 | -1.6667 | -1.0000 | 10000/10000 |
| persuasion | OpponentResponsiveness | 6 | -0.6667 | -1.3333 | -0.1667 | 10000/10000 |
| persuasion | judge_mean | 6 | -0.8889 | -1.3889 | -0.5000 | 10000/10000 |

### d) Pairwise judge Spearman + mean offset (point estimates only)

| scope | judge_pair | n | spearman_rho | mean_offset |
|---|---|---|---|---|
| overall | GameTheoreticRigor|LiteralGroundedness | 20 | 0.5666 | -0.5500 |
| overall | GameTheoreticRigor|OpponentResponsiveness | 20 | 0.8924 | 0.0500 |
| overall | LiteralGroundedness|OpponentResponsiveness | 20 | 0.5846 | 0.6000 |
| bargaining | GameTheoreticRigor|LiteralGroundedness | 7 | 0.5217 | -1.0000 |
| bargaining | GameTheoreticRigor|OpponentResponsiveness | 7 | 0.7246 | 0.0000 |
| bargaining | LiteralGroundedness|OpponentResponsiveness | 7 | 0.7201 | 1.0000 |
| negotiation | GameTheoreticRigor|LiteralGroundedness | 7 | undefined (constant) | 0.0000 |
| negotiation | GameTheoreticRigor|OpponentResponsiveness | 7 | undefined (constant) | 0.1429 |
| negotiation | LiteralGroundedness|OpponentResponsiveness | 7 | undefined (constant) | 0.1429 |
| persuasion | GameTheoreticRigor|LiteralGroundedness | 6 | 0.5657 | -0.6667 |
| persuasion | GameTheoreticRigor|OpponentResponsiveness | 6 | 1.0000 | 0.0000 |
| persuasion | LiteralGroundedness|OpponentResponsiveness | 6 | 0.5657 | 0.6667 |

## horizon_strategy_planning

### a) Spearman rho vs. author (+ 95% bootstrap CI)

| scope | target | n | rho | ci_low | ci_high | n_valid_reps |
|---|---|---|---|---|---|---|
| overall | GameTheoreticRigor | 20 | 0.0953 | -0.3376 | 0.5014 | 10000/10000 |
| overall | LiteralGroundedness | 20 | 0.3221 | -0.2308 | 0.7544 | 10000/10000 |
| overall | OpponentResponsiveness | 20 | 0.3901 | -0.0100 | 0.6961 | 9999/10000 |
| overall | judge_mean | 20 | 0.2862 | -0.1966 | 0.6794 | 10000/10000 |
| bargaining | GameTheoreticRigor | 7 | 0.0000 | -0.5477 | 0.5477 | 6691/10000 |
| bargaining | LiteralGroundedness | 7 | 0.1400 | -0.9354 | 0.9661 | 9045/10000 |
| bargaining | OpponentResponsiveness | 7 | 0.0000 | -0.7303 | 0.7500 | 9013/10000 |
| bargaining | judge_mean | 7 | 0.1373 | -0.8126 | 0.9354 | 9938/10000 |
| negotiation | GameTheoreticRigor | 7 | 0.4583 | -0.2415 | 0.9354 | 9898/10000 |
| negotiation | LiteralGroundedness | 7 | 0.4677 | -0.3416 | 0.9860 | 9743/10000 |
| negotiation | OpponentResponsiveness | 7 | 0.7100 | 0.2336 | 1.0000 | 9778/10000 |
| negotiation | judge_mean | 7 | 0.6063 | -0.1667 | 0.9354 | 9922/10000 |
| persuasion | GameTheoreticRigor | 6 | -0.2236 | -1.0000 | 0.7071 | 9002/10000 |
| persuasion | LiteralGroundedness | 6 | 0.6124 | 0.2000 | 1.0000 | 8404/10000 |
| persuasion | OpponentResponsiveness | 6 | 0.2236 | -0.7071 | 1.0000 | 8925/10000 |
| persuasion | judge_mean | 6 | 0.1050 | -0.8165 | 0.9045 | 9058/10000 |

### b) Exact / within-1 agreement vs. author (rounded)

| scope | target | n | exact | within_1 |
|---|---|---|---|---|
| overall | GameTheoreticRigor | 20 | 0.3500 | 0.7500 |
| overall | LiteralGroundedness | 20 | 0.0500 | 0.4500 |
| overall | OpponentResponsiveness | 20 | 0.3500 | 0.8500 |
| overall | judge_mean | 20 | 0.2500 | 0.8000 |
| bargaining | GameTheoreticRigor | 7 | 0.2857 | 0.8571 |
| bargaining | LiteralGroundedness | 7 | 0.1429 | 0.7143 |
| bargaining | OpponentResponsiveness | 7 | 0.1429 | 0.8571 |
| bargaining | judge_mean | 7 | 0.1429 | 1.0000 |
| negotiation | GameTheoreticRigor | 7 | 0.2857 | 0.7143 |
| negotiation | LiteralGroundedness | 7 | 0.0000 | 0.2857 |
| negotiation | OpponentResponsiveness | 7 | 0.2857 | 0.8571 |
| negotiation | judge_mean | 7 | 0.2857 | 0.7143 |
| persuasion | GameTheoreticRigor | 6 | 0.5000 | 0.6667 |
| persuasion | LiteralGroundedness | 6 | 0.0000 | 0.3333 |
| persuasion | OpponentResponsiveness | 6 | 0.6667 | 0.8333 |
| persuasion | judge_mean | 6 | 0.3333 | 0.6667 |

### c) Mean offset (author - target), raw (+ 95% bootstrap CI)

| scope | target | n | mean_offset | ci_low | ci_high | n_valid_reps |
|---|---|---|---|---|---|---|
| overall | GameTheoreticRigor | 20 | -0.6000 | -1.0000 | -0.1500 | 10000/10000 |
| overall | LiteralGroundedness | 20 | -1.4500 | -1.8000 | -1.0500 | 10000/10000 |
| overall | OpponentResponsiveness | 20 | -0.5000 | -0.9000 | -0.1000 | 10000/10000 |
| overall | judge_mean | 20 | -0.8500 | -1.2167 | -0.4500 | 10000/10000 |
| bargaining | GameTheoreticRigor | 7 | 0.0000 | -0.7143 | 0.8571 | 10000/10000 |
| bargaining | LiteralGroundedness | 7 | -0.8571 | -1.5714 | -0.1429 | 10000/10000 |
| bargaining | OpponentResponsiveness | 7 | -0.1429 | -0.8571 | 0.7143 | 10000/10000 |
| bargaining | judge_mean | 7 | -0.3333 | -1.0000 | 0.4286 | 10000/10000 |
| negotiation | GameTheoreticRigor | 7 | -1.0000 | -1.5714 | -0.4286 | 10000/10000 |
| negotiation | LiteralGroundedness | 7 | -1.8571 | -2.2857 | -1.4286 | 10000/10000 |
| negotiation | OpponentResponsiveness | 7 | -0.8571 | -1.2857 | -0.4286 | 10000/10000 |
| negotiation | judge_mean | 7 | -1.2381 | -1.7143 | -0.7619 | 10000/10000 |
| persuasion | GameTheoreticRigor | 6 | -0.8333 | -1.5000 | -0.1667 | 10000/10000 |
| persuasion | LiteralGroundedness | 6 | -1.6667 | -2.0000 | -1.3333 | 10000/10000 |
| persuasion | OpponentResponsiveness | 6 | -0.5000 | -1.1667 | 0.0000 | 10000/10000 |
| persuasion | judge_mean | 6 | -1.0000 | -1.5000 | -0.5000 | 10000/10000 |

### d) Pairwise judge Spearman + mean offset (point estimates only)

| scope | judge_pair | n | spearman_rho | mean_offset |
|---|---|---|---|---|
| overall | GameTheoreticRigor|LiteralGroundedness | 20 | 0.5573 | -0.8500 |
| overall | GameTheoreticRigor|OpponentResponsiveness | 20 | 0.7387 | 0.1000 |
| overall | LiteralGroundedness|OpponentResponsiveness | 20 | 0.4441 | 0.9500 |
| bargaining | GameTheoreticRigor|LiteralGroundedness | 7 | 0.0000 | -0.8571 |
| bargaining | GameTheoreticRigor|OpponentResponsiveness | 7 | 0.6455 | -0.1429 |
| bargaining | LiteralGroundedness|OpponentResponsiveness | 7 | 0.0000 | 0.7143 |
| negotiation | GameTheoreticRigor|LiteralGroundedness | 7 | 0.9354 | -0.8571 |
| negotiation | GameTheoreticRigor|OpponentResponsiveness | 7 | 0.8391 | 0.1429 |
| negotiation | LiteralGroundedness|OpponentResponsiveness | 7 | 0.7246 | 1.0000 |
| persuasion | GameTheoreticRigor|LiteralGroundedness | 6 | 0.3651 | -0.8333 |
| persuasion | GameTheoreticRigor|OpponentResponsiveness | 6 | 0.7667 | 0.3333 |
| persuasion | LiteralGroundedness|OpponentResponsiveness | 6 | 0.4564 | 1.1667 |

## outcome_consistency

### a) Spearman rho vs. author (+ 95% bootstrap CI)

| scope | target | n | rho | ci_low | ci_high | n_valid_reps |
|---|---|---|---|---|---|---|
| overall | GameTheoreticRigor | 20 | undefined (constant) | n/a | n/a | 0/10000 |
| overall | LiteralGroundedness | 20 | undefined (constant) | n/a | n/a | 0/10000 |
| overall | OpponentResponsiveness | 20 | undefined (constant) | n/a | n/a | 0/10000 |
| overall | judge_mean | 20 | undefined (constant) | n/a | n/a | 0/10000 |
| bargaining | GameTheoreticRigor | 7 | undefined (constant) | n/a | n/a | 0/10000 |
| bargaining | LiteralGroundedness | 7 | undefined (constant) | n/a | n/a | 0/10000 |
| bargaining | OpponentResponsiveness | 7 | undefined (constant) | n/a | n/a | 0/10000 |
| bargaining | judge_mean | 7 | undefined (constant) | n/a | n/a | 0/10000 |
| negotiation | GameTheoreticRigor | 7 | undefined (constant) | n/a | n/a | 0/10000 |
| negotiation | LiteralGroundedness | 7 | undefined (constant) | n/a | n/a | 0/10000 |
| negotiation | OpponentResponsiveness | 7 | undefined (constant) | n/a | n/a | 0/10000 |
| negotiation | judge_mean | 7 | undefined (constant) | n/a | n/a | 0/10000 |
| persuasion | GameTheoreticRigor | 6 | undefined (constant) | n/a | n/a | 0/10000 |
| persuasion | LiteralGroundedness | 6 | undefined (constant) | n/a | n/a | 0/10000 |
| persuasion | OpponentResponsiveness | 6 | undefined (constant) | n/a | n/a | 0/10000 |
| persuasion | judge_mean | 6 | undefined (constant) | n/a | n/a | 0/10000 |

### b) Exact / within-1 agreement vs. author (rounded)

| scope | target | n | exact | within_1 |
|---|---|---|---|---|
| overall | GameTheoreticRigor | 20 | 0.6000 | 1.0000 |
| overall | LiteralGroundedness | 20 | 1.0000 | 1.0000 |
| overall | OpponentResponsiveness | 20 | 0.6500 | 0.9500 |
| overall | judge_mean | 20 | 0.6500 | 1.0000 |
| bargaining | GameTheoreticRigor | 7 | 0.2857 | 1.0000 |
| bargaining | LiteralGroundedness | 7 | 1.0000 | 1.0000 |
| bargaining | OpponentResponsiveness | 7 | 0.4286 | 1.0000 |
| bargaining | judge_mean | 7 | 0.4286 | 1.0000 |
| negotiation | GameTheoreticRigor | 7 | 0.8571 | 1.0000 |
| negotiation | LiteralGroundedness | 7 | 1.0000 | 1.0000 |
| negotiation | OpponentResponsiveness | 7 | 0.8571 | 0.8571 |
| negotiation | judge_mean | 7 | 0.8571 | 1.0000 |
| persuasion | GameTheoreticRigor | 6 | 0.6667 | 1.0000 |
| persuasion | LiteralGroundedness | 6 | 1.0000 | 1.0000 |
| persuasion | OpponentResponsiveness | 6 | 0.6667 | 1.0000 |
| persuasion | judge_mean | 6 | 0.6667 | 1.0000 |

### c) Mean offset (author - target), raw (+ 95% bootstrap CI)

| scope | target | n | mean_offset | ci_low | ci_high | n_valid_reps |
|---|---|---|---|---|---|---|
| overall | GameTheoreticRigor | 20 | 0.4000 | 0.2000 | 0.6000 | 10000/10000 |
| overall | LiteralGroundedness | 20 | 0.0000 | 0.0000 | 0.0000 | 10000/10000 |
| overall | OpponentResponsiveness | 20 | 0.4000 | 0.1500 | 0.6500 | 10000/10000 |
| overall | judge_mean | 20 | 0.2667 | 0.1167 | 0.4167 | 10000/10000 |
| bargaining | GameTheoreticRigor | 7 | 0.7143 | 0.2857 | 1.0000 | 10000/10000 |
| bargaining | LiteralGroundedness | 7 | 0.0000 | 0.0000 | 0.0000 | 10000/10000 |
| bargaining | OpponentResponsiveness | 7 | 0.5714 | 0.2857 | 0.8571 | 10000/10000 |
| bargaining | judge_mean | 7 | 0.4286 | 0.1905 | 0.6190 | 10000/10000 |
| negotiation | GameTheoreticRigor | 7 | 0.1429 | 0.0000 | 0.4286 | 10000/10000 |
| negotiation | LiteralGroundedness | 7 | 0.0000 | 0.0000 | 0.0000 | 10000/10000 |
| negotiation | OpponentResponsiveness | 7 | 0.2857 | 0.0000 | 0.8571 | 10000/10000 |
| negotiation | judge_mean | 7 | 0.1429 | 0.0000 | 0.4286 | 10000/10000 |
| persuasion | GameTheoreticRigor | 6 | 0.3333 | 0.0000 | 0.6667 | 10000/10000 |
| persuasion | LiteralGroundedness | 6 | 0.0000 | 0.0000 | 0.0000 | 10000/10000 |
| persuasion | OpponentResponsiveness | 6 | 0.3333 | 0.0000 | 0.6667 | 10000/10000 |
| persuasion | judge_mean | 6 | 0.2222 | 0.0000 | 0.4444 | 10000/10000 |

### d) Pairwise judge Spearman + mean offset (point estimates only)

| scope | judge_pair | n | spearman_rho | mean_offset |
|---|---|---|---|---|
| overall | GameTheoreticRigor|LiteralGroundedness | 20 | undefined (constant) | -0.4000 |
| overall | GameTheoreticRigor|OpponentResponsiveness | 20 | 0.8885 | 0.0000 |
| overall | LiteralGroundedness|OpponentResponsiveness | 20 | undefined (constant) | 0.4000 |
| bargaining | GameTheoreticRigor|LiteralGroundedness | 7 | undefined (constant) | -0.7143 |
| bargaining | GameTheoreticRigor|OpponentResponsiveness | 7 | 0.7303 | -0.1429 |
| bargaining | LiteralGroundedness|OpponentResponsiveness | 7 | undefined (constant) | 0.5714 |
| negotiation | GameTheoreticRigor|LiteralGroundedness | 7 | undefined (constant) | -0.1429 |
| negotiation | GameTheoreticRigor|OpponentResponsiveness | 7 | 1.0000 | 0.1429 |
| negotiation | LiteralGroundedness|OpponentResponsiveness | 7 | undefined (constant) | 0.2857 |
| persuasion | GameTheoreticRigor|LiteralGroundedness | 6 | undefined (constant) | -0.3333 |
| persuasion | GameTheoreticRigor|OpponentResponsiveness | 6 | 1.0000 | 0.0000 |
| persuasion | LiteralGroundedness|OpponentResponsiveness | 6 | undefined (constant) | 0.3333 |

## outcome_consistency rating distributions (n=20)

| rater | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| author | 0 | 0 | 0 | 0 | 20 |
| GameTheoreticRigor | 0 | 0 | 0 | 8 | 12 |
| LiteralGroundedness | 0 | 0 | 0 | 0 | 20 |
| OpponentResponsiveness | 0 | 0 | 1 | 6 | 13 |
