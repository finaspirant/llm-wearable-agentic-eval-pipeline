# Stratified GLEE PIA -- post-hoc analysis

## 1. Integrity

Mismatches between stored and recomputed kappa: **0**

## 2. Aggregates -- with_degenerate

### kappa_overall

| scope | unweighted_mean | member_weighted_mean | pooled | n_clusters |
|---|---|---|---|---|
| overall | 0.0428 | 0.0923 | 0.1768 | 105 |
| bargaining | -0.0128 | 0.0641 | 0.1288 | 35 |
| negotiation | -0.0286 | -0.0009 | 0.0548 | 35 |
| persuasion | 0.1700 | 0.1700 | 0.1964 | 35 |

### valuation_reasoning

| scope | unweighted_mean | member_weighted_mean | pooled | n_clusters |
|---|---|---|---|---|
| overall | 0.0606 | 0.1066 | 0.2759 | 105 |
| bargaining | -0.0170 | 0.0655 | 0.1415 | 35 |
| negotiation | 0.0233 | 0.0408 | 0.0237 | 35 |
| persuasion | 0.1757 | 0.1757 | 0.2369 | 35 |

### horizon_strategy_planning

| scope | unweighted_mean | member_weighted_mean | pooled | n_clusters |
|---|---|---|---|---|
| overall | -0.0360 | 0.0106 | 0.1112 | 105 |
| bargaining | -0.0199 | 0.0725 | 0.1307 | 35 |
| negotiation | -0.1151 | -0.0809 | 0.0449 | 35 |
| persuasion | 0.0270 | 0.0270 | 0.0927 | 35 |

### concession_handling

| scope | unweighted_mean | member_weighted_mean | pooled | n_clusters |
|---|---|---|---|---|
| overall | -0.0976 | -0.0423 | 0.0865 | 66 |
| bargaining | -0.0769 | 0.0255 | 0.1034 | 32 |
| negotiation | -0.1171 | -0.1079 | 0.0114 | 34 |
| persuasion | n/a | n/a | n/a | 0 |

### outcome_consistency

| scope | unweighted_mean | member_weighted_mean | pooled | n_clusters |
|---|---|---|---|---|
| overall | 0.1414 | 0.1867 | 0.2338 | 105 |
| bargaining | 0.0480 | 0.0841 | 0.1395 | 35 |
| negotiation | 0.0689 | 0.1043 | 0.1392 | 35 |
| persuasion | 0.3073 | 0.3073 | 0.2595 | 35 |

## 2. Aggregates -- without_degenerate

### kappa_overall

| scope | unweighted_mean | member_weighted_mean | pooled | n_clusters |
|---|---|---|---|---|
| overall | 0.0074 | 0.0581 | 0.1703 | 105 |
| bargaining | -0.0232 | 0.0609 | 0.1277 | 35 |
| negotiation | -0.0665 | -0.0303 | 0.0478 | 35 |
| persuasion | 0.1120 | 0.1120 | 0.1911 | 35 |

### valuation_reasoning

| scope | unweighted_mean | member_weighted_mean | pooled | n_clusters |
|---|---|---|---|---|
| overall | 0.0330 | 0.0897 | 0.2693 | 102 |
| bargaining | -0.0170 | 0.0655 | 0.1415 | 35 |
| negotiation | -0.0683 | -0.0290 | 0.0081 | 32 |
| persuasion | 0.1757 | 0.1757 | 0.2369 | 35 |

### horizon_strategy_planning

| scope | unweighted_mean | member_weighted_mean | pooled | n_clusters |
|---|---|---|---|---|
| overall | -0.0360 | 0.0106 | 0.1112 | 105 |
| bargaining | -0.0199 | 0.0725 | 0.1307 | 35 |
| negotiation | -0.1151 | -0.0809 | 0.0449 | 35 |
| persuasion | 0.0270 | 0.0270 | 0.0927 | 35 |

### concession_handling

| scope | unweighted_mean | member_weighted_mean | pooled | n_clusters |
|---|---|---|---|---|
| overall | -0.0976 | -0.0423 | 0.0865 | 66 |
| bargaining | -0.0769 | 0.0255 | 0.1034 | 32 |
| negotiation | -0.1171 | -0.1079 | 0.0114 | 34 |
| persuasion | n/a | n/a | n/a | 0 |

### outcome_consistency

| scope | unweighted_mean | member_weighted_mean | pooled | n_clusters |
|---|---|---|---|---|
| overall | 0.0609 | 0.1044 | 0.2142 | 96 |
| bargaining | 0.0200 | 0.0762 | 0.1353 | 34 |
| negotiation | 0.0124 | 0.0528 | 0.1268 | 33 |
| persuasion | 0.1640 | 0.1640 | 0.2438 | 29 |

## 3. Degenerate clusters (denominator ~ 0, kappa forced to 1.0)

Count: **12**

| cluster_id | family | dimension | n_judged_members | n_items_used_in_kappa | uniform_score_value |
|---|---|---|---|---|---|
| bargaining_062 | bargaining | outcome_consistency | 2 | 2 | 5 |
| negotiation_002 | negotiation | valuation_reasoning | 10 | 10 | 5 |
| negotiation_002 | negotiation | outcome_consistency | 10 | 10 | 5 |
| negotiation_064 | negotiation | valuation_reasoning | 3 | 3 | 5 |
| negotiation_081 | negotiation | valuation_reasoning | 2 | 2 | 5 |
| negotiation_081 | negotiation | outcome_consistency | 2 | 2 | 5 |
| persuasion_011 | persuasion | outcome_consistency | 10 | 10 | 5 |
| persuasion_015 | persuasion | outcome_consistency | 10 | 10 | 5 |
| persuasion_103 | persuasion | outcome_consistency | 10 | 10 | 5 |
| persuasion_113 | persuasion | outcome_consistency | 10 | 10 | 5 |
| persuasion_184 | persuasion | outcome_consistency | 10 | 10 | 5 |
| persuasion_188 | persuasion | outcome_consistency | 10 | 10 | 5 |

## 4. Cluster bootstrap (primary aggregate: unweighted mean)

seed=20261002  n_resamples=10000

| stat | scope | ci_low | ci_high | n_valid_reps | fraction_kappa_gt_0 |
|---|---|---|---|---|---|
| kappa_overall | overall | 0.0115 | 0.0765 | 10000/10000 | 0.9972 |
| kappa_overall | bargaining | -0.0721 | 0.0465 | 10000/10000 | 0.3434 |
| kappa_overall | negotiation | -0.0891 | 0.0437 | 10000/10000 | 0.1918 |
| kappa_overall | persuasion | 0.1298 | 0.2113 | 10000/10000 | 1.0000 |
| valuation_reasoning | overall | 0.0112 | 0.1146 | 10000/10000 | 0.9930 |
| valuation_reasoning | bargaining | -0.0953 | 0.0616 | 10000/10000 | 0.3394 |
| valuation_reasoning | negotiation | -0.0822 | 0.1477 | 10000/10000 | 0.6377 |
| valuation_reasoning | persuasion | 0.1106 | 0.2433 | 10000/10000 | 1.0000 |
| horizon_strategy_planning | overall | -0.0730 | 0.0012 | 10000/10000 | 0.0289 |
| horizon_strategy_planning | bargaining | -0.1023 | 0.0632 | 10000/10000 | 0.3223 |
| horizon_strategy_planning | negotiation | -0.1670 | -0.0618 | 10000/10000 | 0.0000 |
| horizon_strategy_planning | persuasion | -0.0242 | 0.0767 | 10000/10000 | 0.8554 |
| concession_handling | overall | -0.1441 | -0.0511 | 10000/10000 | 0.0000 |
| concession_handling | bargaining | -0.1581 | 0.0057 | 10000/10000 | 0.0343 |
| concession_handling | negotiation | -0.1629 | -0.0691 | 10000/10000 | 0.0000 |
| concession_handling | persuasion | n/a | n/a | 0/10000 | n/a |
| outcome_consistency | overall | 0.0854 | 0.2011 | 10000/10000 | 1.0000 |
| outcome_consistency | bargaining | -0.0291 | 0.1346 | 10000/10000 | 0.8771 |
| outcome_consistency | negotiation | -0.0216 | 0.1751 | 10000/10000 | 0.9213 |
| outcome_consistency | persuasion | 0.1955 | 0.4297 | 10000/10000 | 1.0000 |

## 5. Cluster bootstrap, 12 degenerate entries excluded

seed=20261002  n_resamples=10000

| stat | scope | ci_low | ci_high | n_valid_reps | fraction_kappa_gt_0 |
|---|---|---|---|---|---|
| kappa_overall | overall | -0.0220 | 0.0364 | 10000/10000 | 0.6933 |
| kappa_overall | bargaining | -0.0882 | 0.0410 | 10000/10000 | 0.2453 |
| kappa_overall | negotiation | -0.1104 | -0.0207 | 10000/10000 | 0.0022 |
| kappa_overall | persuasion | 0.0702 | 0.1530 | 10000/10000 | 1.0000 |
| valuation_reasoning | overall | -0.0071 | 0.0735 | 10000/10000 | 0.9465 |
| valuation_reasoning | bargaining | -0.0953 | 0.0616 | 10000/10000 | 0.3394 |
| valuation_reasoning | negotiation | -0.1320 | -0.0016 | 10000/10000 | 0.0227 |
| valuation_reasoning | persuasion | 0.1106 | 0.2433 | 10000/10000 | 1.0000 |
| horizon_strategy_planning | overall | -0.0730 | 0.0012 | 10000/10000 | 0.0289 |
| horizon_strategy_planning | bargaining | -0.1023 | 0.0632 | 10000/10000 | 0.3223 |
| horizon_strategy_planning | negotiation | -0.1670 | -0.0618 | 10000/10000 | 0.0000 |
| horizon_strategy_planning | persuasion | -0.0242 | 0.0767 | 10000/10000 | 0.8554 |
| concession_handling | overall | -0.1441 | -0.0511 | 10000/10000 | 0.0000 |
| concession_handling | bargaining | -0.1581 | 0.0057 | 10000/10000 | 0.0343 |
| concession_handling | negotiation | -0.1629 | -0.0691 | 10000/10000 | 0.0000 |
| concession_handling | persuasion | n/a | n/a | 0/10000 | n/a |
| outcome_consistency | overall | 0.0230 | 0.0991 | 10000/10000 | 0.9996 |
| outcome_consistency | bargaining | -0.0430 | 0.0836 | 10000/10000 | 0.7288 |
| outcome_consistency | negotiation | -0.0530 | 0.0783 | 10000/10000 | 0.6410 |
| outcome_consistency | persuasion | 0.1013 | 0.2270 | 10000/10000 | 1.0000 |

## 6. Judge agreement diagnostics

### valuation_reasoning

**bargaining** -- n_scores=702, distribution={'1': 0, '2': 58, '3': 260, '4': 236, '5': 148}, frac_ge_4=0.5470, frac_eq_5=0.2108, per_judge_mean={'GameTheoreticRigor': '3.3547', 'LiteralGroundedness': '4.1838', 'OpponentResponsiveness': '3.4872'}

| judge_pair | n_items_paired | exact_agreement | within_1_agreement | spearman_rho | cohens_kappa |
|---|---|---|---|---|---|
| GameTheoreticRigor|LiteralGroundedness | 234 | 0.3419 | 0.8077 | 0.6570 | 0.1589 |
| GameTheoreticRigor|OpponentResponsiveness | 234 | 0.5897 | 0.9701 | 0.5914 | 0.3559 |
| LiteralGroundedness|OpponentResponsiveness | 234 | 0.2692 | 0.8120 | 0.3909 | 0.0536 |

**negotiation** -- n_scores=663, distribution={'1': 0, '2': 2, '3': 16, '4': 116, '5': 529}, frac_ge_4=0.9729, frac_eq_5=0.7979, per_judge_mean={'GameTheoreticRigor': '4.8281', 'LiteralGroundedness': '4.9955', 'OpponentResponsiveness': '4.4796'}

| judge_pair | n_items_paired | exact_agreement | within_1_agreement | spearman_rho | cohens_kappa |
|---|---|---|---|---|---|
| GameTheoreticRigor|LiteralGroundedness | 221 | 0.8416 | 0.9910 | -0.0287 | -0.0086 |
| GameTheoreticRigor|OpponentResponsiveness | 221 | 0.6335 | 0.9910 | 0.5349 | 0.2341 |
| LiteralGroundedness|OpponentResponsiveness | 221 | 0.5520 | 0.9321 | 0.1228 | 0.0018 |

**persuasion** -- n_scores=1050, distribution={'1': 0, '2': 2, '3': 74, '4': 288, '5': 686}, frac_ge_4=0.9276, frac_eq_5=0.6533, per_judge_mean={'GameTheoreticRigor': '4.4600', 'LiteralGroundedness': '4.9086', 'OpponentResponsiveness': '4.3686'}

| judge_pair | n_items_paired | exact_agreement | within_1_agreement | spearman_rho | cohens_kappa |
|---|---|---|---|---|---|
| GameTheoreticRigor|LiteralGroundedness | 350 | 0.5857 | 0.9371 | 0.3046 | 0.0998 |
| GameTheoreticRigor|OpponentResponsiveness | 350 | 0.7486 | 1.0000 | 0.7345 | 0.5666 |
| LiteralGroundedness|OpponentResponsiveness | 350 | 0.5371 | 0.9114 | 0.2857 | 0.1250 |

### horizon_strategy_planning

**bargaining** -- n_scores=702, distribution={'1': 0, '2': 50, '3': 363, '4': 228, '5': 61}, frac_ge_4=0.4117, frac_eq_5=0.0869, per_judge_mean={'GameTheoreticRigor': '3.1581', 'LiteralGroundedness': '3.8718', 'OpponentResponsiveness': '3.2521'}

| judge_pair | n_items_paired | exact_agreement | within_1_agreement | spearman_rho | cohens_kappa |
|---|---|---|---|---|---|
| GameTheoreticRigor|LiteralGroundedness | 234 | 0.3248 | 0.9530 | 0.6830 | 0.0416 |
| GameTheoreticRigor|OpponentResponsiveness | 234 | 0.6838 | 1.0000 | 0.6324 | 0.4077 |
| LiteralGroundedness|OpponentResponsiveness | 234 | 0.3889 | 0.9316 | 0.5387 | 0.1062 |

**negotiation** -- n_scores=663, distribution={'1': 3, '2': 111, '3': 182, '4': 278, '5': 89}, frac_ge_4=0.5535, frac_eq_5=0.1342, per_judge_mean={'GameTheoreticRigor': '3.1584', 'LiteralGroundedness': '4.2670', 'OpponentResponsiveness': '3.1086'}

| judge_pair | n_items_paired | exact_agreement | within_1_agreement | spearman_rho | cohens_kappa |
|---|---|---|---|---|---|
| GameTheoreticRigor|LiteralGroundedness | 221 | 0.1312 | 0.7602 | 0.7098 | -0.1431 |
| GameTheoreticRigor|OpponentResponsiveness | 221 | 0.7195 | 0.9955 | 0.8105 | 0.5942 |
| LiteralGroundedness|OpponentResponsiveness | 221 | 0.1357 | 0.7104 | 0.7063 | -0.1026 |

**persuasion** -- n_scores=1050, distribution={'1': 1, '2': 84, '3': 312, '4': 463, '5': 190}, frac_ge_4=0.6219, frac_eq_5=0.1810, per_judge_mean={'GameTheoreticRigor': '3.6000', 'LiteralGroundedness': '4.2486', 'OpponentResponsiveness': '3.3143'}

| judge_pair | n_items_paired | exact_agreement | within_1_agreement | spearman_rho | cohens_kappa |
|---|---|---|---|---|---|
| GameTheoreticRigor|LiteralGroundedness | 350 | 0.4229 | 0.8943 | 0.5298 | 0.1482 |
| GameTheoreticRigor|OpponentResponsiveness | 350 | 0.4914 | 0.9686 | 0.5896 | 0.2505 |
| LiteralGroundedness|OpponentResponsiveness | 350 | 0.2400 | 0.8229 | 0.6226 | -0.0044 |

### concession_handling

**bargaining** -- n_scores=612, distribution={'1': 1, '2': 37, '3': 214, '4': 266, '5': 94}, frac_ge_4=0.5882, frac_eq_5=0.1536, per_judge_mean={'GameTheoreticRigor': '3.4216', 'LiteralGroundedness': '4.1765', 'OpponentResponsiveness': '3.4363'}

| judge_pair | n_items_paired | exact_agreement | within_1_agreement | spearman_rho | cohens_kappa |
|---|---|---|---|---|---|
| GameTheoreticRigor|LiteralGroundedness | 204 | 0.2941 | 0.9510 | 0.6876 | 0.0054 |
| GameTheoreticRigor|OpponentResponsiveness | 204 | 0.5980 | 0.9657 | 0.5975 | 0.3728 |
| LiteralGroundedness|OpponentResponsiveness | 204 | 0.3284 | 0.8824 | 0.5758 | 0.0517 |

**negotiation** -- n_scores=633, distribution={'1': 4, '2': 76, '3': 93, '4': 231, '5': 229}, frac_ge_4=0.7267, frac_eq_5=0.3618, per_judge_mean={'GameTheoreticRigor': '3.6019', 'LiteralGroundedness': '4.8341', 'OpponentResponsiveness': '3.4313'}

| judge_pair | n_items_paired | exact_agreement | within_1_agreement | spearman_rho | cohens_kappa |
|---|---|---|---|---|---|
| GameTheoreticRigor|LiteralGroundedness | 211 | 0.1469 | 0.6967 | 0.4021 | -0.0411 |
| GameTheoreticRigor|OpponentResponsiveness | 211 | 0.5877 | 0.9479 | 0.6813 | 0.4060 |
| LiteralGroundedness|OpponentResponsiveness | 211 | 0.1896 | 0.6114 | 0.1651 | 0.0080 |

**persuasion** -- n_scores=0, distribution={'1': 0, '2': 0, '3': 0, '4': 0, '5': 0}, frac_ge_4=n/a, frac_eq_5=n/a, per_judge_mean={'GameTheoreticRigor': 'n/a', 'LiteralGroundedness': 'n/a', 'OpponentResponsiveness': 'n/a'}

| judge_pair | n_items_paired | exact_agreement | within_1_agreement | spearman_rho | cohens_kappa |
|---|---|---|---|---|---|
| GameTheoreticRigor|LiteralGroundedness | 0 | n/a | n/a | n/a | n/a |
| GameTheoreticRigor|OpponentResponsiveness | 0 | n/a | n/a | n/a | n/a |
| LiteralGroundedness|OpponentResponsiveness | 0 | n/a | n/a | n/a | n/a |

### outcome_consistency

**bargaining** -- n_scores=702, distribution={'1': 2, '2': 6, '3': 30, '4': 247, '5': 417}, frac_ge_4=0.9459, frac_eq_5=0.5940, per_judge_mean={'GameTheoreticRigor': '4.2222', 'LiteralGroundedness': '4.8761', 'OpponentResponsiveness': '4.4786'}

| judge_pair | n_items_paired | exact_agreement | within_1_agreement | spearman_rho | cohens_kappa |
|---|---|---|---|---|---|
| GameTheoreticRigor|LiteralGroundedness | 234 | 0.4017 | 0.9359 | 0.3589 | 0.0724 |
| GameTheoreticRigor|OpponentResponsiveness | 234 | 0.6709 | 0.9957 | 0.6333 | 0.4296 |
| LiteralGroundedness|OpponentResponsiveness | 234 | 0.5812 | 0.9786 | 0.3492 | 0.1442 |

**negotiation** -- n_scores=663, distribution={'1': 0, '2': 0, '3': 18, '4': 129, '5': 516}, frac_ge_4=0.9729, frac_eq_5=0.7783, per_judge_mean={'GameTheoreticRigor': '4.5566', 'LiteralGroundedness': '5.0000', 'OpponentResponsiveness': '4.6968'}

| judge_pair | n_items_paired | exact_agreement | within_1_agreement | spearman_rho | cohens_kappa |
|---|---|---|---|---|---|
| GameTheoreticRigor|LiteralGroundedness | 221 | 0.6063 | 0.9502 | n/a | 0.0000 |
| GameTheoreticRigor|OpponentResponsiveness | 221 | 0.7466 | 0.9955 | 0.6141 | 0.4657 |
| LiteralGroundedness|OpponentResponsiveness | 221 | 0.7285 | 0.9683 | n/a | 0.0000 |

**persuasion** -- n_scores=1050, distribution={'1': 0, '2': 2, '3': 14, '4': 94, '5': 940}, frac_ge_4=0.9848, frac_eq_5=0.8952, per_judge_mean={'GameTheoreticRigor': '4.7800', 'LiteralGroundedness': '4.9971', 'OpponentResponsiveness': '4.8571'}

| judge_pair | n_items_paired | exact_agreement | within_1_agreement | spearman_rho | cohens_kappa |
|---|---|---|---|---|---|
| GameTheoreticRigor|LiteralGroundedness | 350 | 0.8057 | 0.9771 | 0.1315 | 0.0093 |
| GameTheoreticRigor|OpponentResponsiveness | 350 | 0.8886 | 0.9971 | 0.6893 | 0.5902 |
| LiteralGroundedness|OpponentResponsiveness | 350 | 0.8829 | 0.9829 | 0.1620 | 0.0188 |

## 7. Shuffled-cluster floor

| stat | family | real | shuffled_mean | ci_low | ci_high | n_valid_perms |
|---|---|---|---|---|---|---|
| kappa_overall | bargaining | -0.0128 | 0.0321 | -0.0028 | 0.0716 | 1000/1000 |
| kappa_overall | negotiation | -0.0286 | 0.0179 | -0.0201 | 0.0604 | 1000/1000 |
| kappa_overall | persuasion | 0.1700 | 0.1789 | 0.1557 | 0.2042 | 1000/1000 |
| valuation_reasoning | bargaining | -0.0170 | 0.0350 | -0.0165 | 0.0905 | 1000/1000 |
| valuation_reasoning | negotiation | 0.0233 | 0.0427 | -0.0430 | 0.1328 | 1000/1000 |
| valuation_reasoning | persuasion | 0.1757 | 0.2034 | 0.1853 | 0.2193 | 1000/1000 |
| horizon_strategy_planning | bargaining | -0.0199 | 0.0244 | -0.0284 | 0.0876 | 1000/1000 |
| horizon_strategy_planning | negotiation | -0.1151 | -0.0498 | -0.0876 | -0.0112 | 1000/1000 |
| horizon_strategy_planning | persuasion | 0.0270 | 0.0534 | 0.0407 | 0.0641 | 1000/1000 |
| concession_handling | bargaining | -0.0769 | 0.0023 | -0.0437 | 0.0590 | 1000/1000 |
| concession_handling | negotiation | -0.1171 | -0.0778 | -0.1167 | -0.0376 | 1000/1000 |
| concession_handling | persuasion | n/a | n/a | n/a | n/a | 0/1000 |
| outcome_consistency | bargaining | 0.0480 | 0.0710 | 0.0040 | 0.1469 | 1000/1000 |
| outcome_consistency | negotiation | 0.0689 | 0.1516 | 0.0659 | 0.2422 | 1000/1000 |
| outcome_consistency | persuasion | 0.3073 | 0.2800 | 0.2115 | 0.3486 | 1000/1000 |

## 8. Cluster dependence (shared games)

| family | n_clusters | distinct_games | mean_clusters_per_game | share_clusters_sharing_a_game |
|---|---|---|---|---|
| bargaining | 35 | 139 | 1.6043 | 0.9429 |
| negotiation | 35 | 125 | 1.7680 | 1.0000 |
| persuasion | 35 | 246 | 1.0976 | 0.3429 |

## 9. Cluster-game connected components (resampling unit)

| family | n_clusters | n_components | largest_component_size | component_sizes (desc) |
|---|---|---|---|---|
| bargaining | 35 | 7 | 14 | [14, 7, 6, 4, 2, 1, 1] |
| negotiation | 35 | 5 | 21 | [21, 4, 4, 3, 3] |
| persuasion | 35 | 29 | 2 | [2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] |

## 10. Cluster-level vs. game-level bootstrap (both: 12 degenerate entries excluded)

cluster-level: seed=20261002 n_resamples=10000  |  game-level: seed=20261002 n_resamples=10000

| stat | scope | cluster_ci_low | cluster_ci_high | game_ci_low | game_ci_high | cluster_frac>0 | game_frac>0 |
|---|---|---|---|---|---|---|---|
| kappa_overall | overall | -0.0220 | 0.0364 | -0.0431 | 0.0791 | 0.6933 | 0.6215 |
| kappa_overall | bargaining | -0.0882 | 0.0410 | -0.1176 | 0.1045 | 0.2453 | 0.3659 |
| kappa_overall | negotiation | -0.1104 | -0.0207 | -0.1493 | 0.0538 | 0.0022 | 0.0860 |
| kappa_overall | persuasion | 0.0702 | 0.1530 | 0.0698 | 0.1539 | 1.0000 | 1.0000 |
| valuation_reasoning | overall | -0.0071 | 0.0735 | -0.0370 | 0.1301 | 0.9465 | 0.8147 |
| valuation_reasoning | bargaining | -0.0953 | 0.0616 | -0.1454 | 0.1713 | 0.3394 | 0.4214 |
| valuation_reasoning | negotiation | -0.1320 | -0.0016 | -0.1594 | 0.0597 | 0.0227 | 0.1178 |
| valuation_reasoning | persuasion | 0.1106 | 0.2433 | 0.1148 | 0.2440 | 1.0000 | 1.0000 |
| horizon_strategy_planning | overall | -0.0730 | 0.0012 | -0.0783 | 0.0292 | 0.0289 | 0.1206 |
| horizon_strategy_planning | bargaining | -0.1023 | 0.0632 | -0.0983 | 0.1385 | 0.3223 | 0.3920 |
| horizon_strategy_planning | negotiation | -0.1670 | -0.0618 | -0.2119 | 0.0119 | 0.0000 | 0.0334 |
| horizon_strategy_planning | persuasion | -0.0242 | 0.0767 | -0.0198 | 0.0780 | 0.8554 | 0.8671 |
| concession_handling | overall | -0.1441 | -0.0511 | -0.1553 | -0.0138 | 0.0000 | 0.0098 |
| concession_handling | bargaining | -0.1581 | 0.0057 | -0.1737 | 0.0665 | 0.0343 | 0.3313 |
| concession_handling | negotiation | -0.1629 | -0.0691 | -0.1825 | -0.0357 | 0.0000 | 0.0055 |
| concession_handling | persuasion | n/a | n/a | n/a | n/a | n/a | n/a |
| outcome_consistency | overall | 0.0230 | 0.0991 | 0.0167 | 0.1173 | 0.9996 | 0.9978 |
| outcome_consistency | bargaining | -0.0430 | 0.0836 | -0.0572 | 0.1080 | 0.7288 | 0.6707 |
| outcome_consistency | negotiation | -0.0530 | 0.0783 | -0.0666 | 0.1030 | 0.6410 | 0.7212 |
| outcome_consistency | persuasion | 0.1013 | 0.2270 | 0.0953 | 0.2314 | 1.0000 | 1.0000 |

## 11. Game-level bootstrap audit (estimator cross-check + CI containment/width checks)

| stat | scope | unweighted_mean | member_weighted_mean | pooled | in_cluster_ci | in_game_ci | game_width>=cluster_width |
|---|---|---|---|---|---|---|---|
| kappa_overall | overall | 0.0074 | 0.0581 | 0.1703 | True | True | True |
| kappa_overall | bargaining | -0.0232 | 0.0609 | 0.1277 | True | True | True |
| kappa_overall | negotiation | -0.0665 | -0.0303 | 0.0478 | True | True | True |
| kappa_overall | persuasion | 0.1120 | 0.1120 | 0.1911 | True | True | True |
| valuation_reasoning | overall | 0.0330 | 0.0897 | 0.2693 | True | True | True |
| valuation_reasoning | bargaining | -0.0170 | 0.0655 | 0.1415 | True | True | True |
| valuation_reasoning | negotiation | -0.0683 | -0.0290 | 0.0081 | True | True | True |
| valuation_reasoning | persuasion | 0.1757 | 0.1757 | 0.2369 | True | True | False |
| horizon_strategy_planning | overall | -0.0360 | 0.0106 | 0.1112 | True | True | True |
| horizon_strategy_planning | bargaining | -0.0199 | 0.0725 | 0.1307 | True | True | True |
| horizon_strategy_planning | negotiation | -0.1151 | -0.0809 | 0.0449 | True | True | True |
| horizon_strategy_planning | persuasion | 0.0270 | 0.0270 | 0.0927 | True | True | False |
| concession_handling | overall | -0.0976 | -0.0423 | 0.0865 | True | True | True |
| concession_handling | bargaining | -0.0769 | 0.0255 | 0.1034 | True | True | True |
| concession_handling | negotiation | -0.1171 | -0.1079 | 0.0114 | True | True | True |
| concession_handling | persuasion | n/a | n/a | n/a | None | None | None |
| outcome_consistency | overall | 0.0609 | 0.1044 | 0.2142 | True | True | True |
| outcome_consistency | bargaining | 0.0200 | 0.0762 | 0.1353 | True | True | True |
| outcome_consistency | negotiation | 0.0124 | 0.0528 | 0.1268 | True | True | True |
| outcome_consistency | persuasion | 0.1640 | 0.1640 | 0.2438 | True | True | True |

## 12. Offset-robust agreement (per-judge-centered ICC(3,1) + Krippendorff's alpha, ordinal)

| dimension | family | icc_3_1_consistency | krippendorff_alpha_ordinal | n_items |
|---|---|---|---|---|
| valuation_reasoning | bargaining | 0.5397 | 0.4625 | 234 |
| valuation_reasoning | negotiation | 0.2776 | 0.1296 | 221 |
| valuation_reasoning | persuasion | 0.4705 | 0.4097 | 350 |
| horizon_strategy_planning | bargaining | 0.6214 | 0.5235 | 234 |
| horizon_strategy_planning | negotiation | 0.7015 | 0.6271 | 221 |
| horizon_strategy_planning | persuasion | 0.5828 | 0.5252 | 350 |
| concession_handling | bargaining | 0.5885 | 0.5465 | 204 |
| concession_handling | negotiation | 0.4121 | 0.3473 | 211 |
| concession_handling | persuasion | n/a | n/a | 0 |
| outcome_consistency | bargaining | 0.5381 | 0.4115 | 234 |
| outcome_consistency | negotiation | 0.3009 | 0.2131 | 221 |
| outcome_consistency | persuasion | 0.3658 | -0.0399 | 350 |

## 13. Leave-one-judge-out (LiteralGroundedness removed)

remaining_judges=['GameTheoreticRigor', 'OpponentResponsiveness']

| stat | scope | unweighted_mean | member_weighted_mean | pooled | n_clusters |
|---|---|---|---|---|---|
| kappa_overall | overall | 0.3047 | 0.3458 | 0.4593 | 105 |
| kappa_overall | bargaining | 0.2521 | 0.3113 | 0.3841 | 35 |
| kappa_overall | negotiation | 0.2674 | 0.3055 | 0.4057 | 35 |
| kappa_overall | persuasion | 0.3944 | 0.3944 | 0.4640 | 35 |
| valuation_reasoning | overall | 0.2730 | 0.3324 | 0.4961 | 105 |
| valuation_reasoning | bargaining | 0.2406 | 0.2744 | 0.3502 | 35 |
| valuation_reasoning | negotiation | 0.0830 | 0.1353 | 0.1712 | 35 |
| valuation_reasoning | persuasion | 0.4956 | 0.4956 | 0.5645 | 35 |
| horizon_strategy_planning | overall | 0.2932 | 0.2884 | 0.4053 | 105 |
| horizon_strategy_planning | bargaining | 0.3174 | 0.3443 | 0.4067 | 35 |
| horizon_strategy_planning | negotiation | 0.4078 | 0.4416 | 0.5931 | 35 |
| horizon_strategy_planning | persuasion | 0.1544 | 0.1544 | 0.2416 | 35 |
| concession_handling | overall | 0.2124 | 0.2666 | 0.3988 | 66 |
| concession_handling | bargaining | 0.1977 | 0.2797 | 0.3623 | 32 |
| concession_handling | negotiation | 0.2262 | 0.2540 | 0.4002 | 34 |
| concession_handling | persuasion | n/a | n/a | n/a | 0 |
| outcome_consistency | overall | 0.3704 | 0.4314 | 0.5369 | 105 |
| outcome_consistency | bargaining | 0.2434 | 0.3431 | 0.4171 | 35 |
| outcome_consistency | negotiation | 0.3347 | 0.3636 | 0.4582 | 35 |
| outcome_consistency | persuasion | 0.5332 | 0.5332 | 0.5860 | 35 |

2-rater shuffled-cluster floor (1,000 perms, same seed):

| stat | family | real | shuffled_mean | ci_low | ci_high | n_valid_perms |
|---|---|---|---|---|---|---|
| kappa_overall | bargaining | 0.2521 | 0.2889 | 0.2250 | 0.3567 | 1000/1000 |
| kappa_overall | negotiation | 0.2674 | 0.3272 | 0.2755 | 0.3839 | 1000/1000 |
| kappa_overall | persuasion | 0.3944 | 0.4186 | 0.3914 | 0.4440 | 1000/1000 |
| valuation_reasoning | bargaining | 0.2406 | 0.2430 | 0.1415 | 0.3547 | 1000/1000 |
| valuation_reasoning | negotiation | 0.0830 | 0.1217 | 0.0211 | 0.2254 | 1000/1000 |
| valuation_reasoning | persuasion | 0.4956 | 0.5279 | 0.5037 | 0.5467 | 1000/1000 |
| horizon_strategy_planning | bargaining | 0.3174 | 0.3284 | 0.2178 | 0.4347 | 1000/1000 |
| horizon_strategy_planning | negotiation | 0.4078 | 0.4997 | 0.4086 | 0.5911 | 1000/1000 |
| horizon_strategy_planning | persuasion | 0.1544 | 0.1874 | 0.1668 | 0.2036 | 1000/1000 |
| concession_handling | bargaining | 0.1977 | 0.2610 | 0.1536 | 0.3771 | 1000/1000 |
| concession_handling | negotiation | 0.2262 | 0.2839 | 0.1906 | 0.3765 | 1000/1000 |
| concession_handling | persuasion | n/a | n/a | n/a | n/a | 0/1000 |
| outcome_consistency | bargaining | 0.2434 | 0.3293 | 0.2247 | 0.4402 | 1000/1000 |
| outcome_consistency | negotiation | 0.3347 | 0.4002 | 0.2970 | 0.5062 | 1000/1000 |
| outcome_consistency | persuasion | 0.5332 | 0.5405 | 0.4635 | 0.6071 | 1000/1000 |

## 14. Variance decomposition of judge-mean score (share explained by cluster membership)

| dimension | scope | real_eta_sq | shuffled_mean | ci_low | ci_high | n_valid_perms | n_members | n_clusters |
|---|---|---|---|---|---|---|---|---|
| valuation_reasoning | overall | 0.5370 | 0.5107 | 0.4919 | 0.5328 | 1000/1000 | 805 | 105 |
| valuation_reasoning | bargaining | 0.1697 | 0.1449 | 0.0888 | 0.2156 | 1000/1000 | 234 | 35 |
| valuation_reasoning | negotiation | 0.2402 | 0.1572 | 0.0985 | 0.2425 | 1000/1000 | 221 | 35 |
| valuation_reasoning | persuasion | 0.1649 | 0.0978 | 0.0614 | 0.1445 | 1000/1000 | 350 | 35 |
| horizon_strategy_planning | overall | 0.3036 | 0.1618 | 0.1329 | 0.1963 | 1000/1000 | 805 | 105 |
| horizon_strategy_planning | bargaining | 0.1561 | 0.1461 | 0.0896 | 0.2160 | 1000/1000 | 234 | 35 |
| horizon_strategy_planning | negotiation | 0.5184 | 0.1543 | 0.0922 | 0.2254 | 1000/1000 | 221 | 35 |
| horizon_strategy_planning | persuasion | 0.1655 | 0.0985 | 0.0605 | 0.1494 | 1000/1000 | 350 | 35 |
| concession_handling | overall | 0.2929 | 0.1951 | 0.1497 | 0.2414 | 1000/1000 | 415 | 66 |
| concession_handling | bargaining | 0.2239 | 0.1534 | 0.0923 | 0.2261 | 1000/1000 | 204 | 32 |
| concession_handling | negotiation | 0.2902 | 0.1576 | 0.0981 | 0.2334 | 1000/1000 | 211 | 34 |
| concession_handling | persuasion | n/a | n/a | n/a | n/a | 0/1000 | 0 | 0 |
| outcome_consistency | overall | 0.2949 | 0.2565 | 0.2219 | 0.2991 | 1000/1000 | 805 | 105 |
| outcome_consistency | bargaining | 0.1624 | 0.1484 | 0.0841 | 0.2441 | 1000/1000 | 234 | 35 |
| outcome_consistency | negotiation | 0.2380 | 0.1556 | 0.0930 | 0.2313 | 1000/1000 | 221 | 35 |
| outcome_consistency | persuasion | 0.1701 | 0.0967 | 0.0593 | 0.1484 | 1000/1000 | 350 | 35 |

## 15. Action-only within-cluster baseline (no judge scores)

| cluster_id | family | n_sampled | n_resolved | n_continuous | continuous_sd | n_categorical | n_distinct_labels | modal_label | modal_share | action_spread | spread_source |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bargaining_001 | bargaining | 10 | 10 | 7 | 0.1748 | 3 | 2 | decision:accept | 0.6667 | 0.1748 | continuous_sd |
| bargaining_003 | bargaining | 10 | 10 | 4 | 0.1325 | 6 | 2 | decision:reject | 0.6667 | 0.1325 | continuous_sd |
| bargaining_005 | bargaining | 10 | 10 | 6 | 0.1633 | 4 | 2 | decision:reject | 0.5000 | 0.1633 | continuous_sd |
| bargaining_008 | bargaining | 10 | 10 | 4 | 0.0768 | 6 | 2 | decision:reject | 0.6667 | 0.0768 | continuous_sd |
| bargaining_009 | bargaining | 10 | 10 | 7 | 0.0809 | 3 | 2 | decision:reject | 0.6667 | 0.0809 | continuous_sd |
| bargaining_010 | bargaining | 10 | 10 | 2 | 0.1273 | 8 | 2 | decision:accept | 0.5000 | 0.1273 | continuous_sd |
| bargaining_013 | bargaining | 6 | 6 | 4 | 0.2398 | 2 | 1 | decision:reject | 1.0000 | 0.2398 | continuous_sd |
| bargaining_016 | bargaining | 10 | 10 | 7 | 0.1016 | 3 | 2 | decision:reject | 0.6667 | 0.1016 | continuous_sd |
| bargaining_018 | bargaining | 10 | 10 | 2 | 0.0212 | 8 | 2 | decision:reject | 0.8750 | 0.0212 | continuous_sd |
| bargaining_019 | bargaining | 10 | 10 | 4 | 0.2273 | 6 | 2 | decision:reject | 0.6667 | 0.2273 | continuous_sd |
| bargaining_020 | bargaining | 10 | 10 | 8 | 0.0443 | 2 | 1 | decision:reject | 1.0000 | 0.0443 | continuous_sd |
| bargaining_021 | bargaining | 10 | 10 | 4 | 0.0968 | 6 | 2 | decision:reject | 0.6667 | 0.0968 | continuous_sd |
| bargaining_024 | bargaining | 10 | 10 | 6 | 0.0581 | 4 | 2 | decision:reject | 0.7500 | 0.0581 | continuous_sd |
| bargaining_026 | bargaining | 8 | 8 | 5 | 0.0365 | 3 | 2 | decision:reject | 0.6667 | 0.0365 | continuous_sd |
| bargaining_027 | bargaining | 6 | 6 | 2 | 0.0424 | 4 | 2 | decision:reject | 0.7500 | 0.0424 | continuous_sd |
| bargaining_028 | bargaining | 3 | 3 | 3 | 0.0404 | 0 | 0 | None | n/a | 0.0404 | continuous_sd |
| bargaining_029 | bargaining | 10 | 10 | 7 | 0.0970 | 3 | 2 | decision:reject | 0.6667 | 0.0970 | continuous_sd |
| bargaining_030 | bargaining | 10 | 10 | 4 | 0.0250 | 6 | 2 | decision:reject | 0.6667 | 0.0250 | continuous_sd |
| bargaining_031 | bargaining | 10 | 10 | 4 | 0.0718 | 6 | 2 | decision:reject | 0.8333 | 0.0718 | continuous_sd |
| bargaining_033 | bargaining | 10 | 10 | 4 | 0.1327 | 6 | 2 | decision:reject | 0.8333 | 0.1327 | continuous_sd |
| bargaining_034 | bargaining | 10 | 10 | 6 | 0.1503 | 4 | 1 | decision:reject | 1.0000 | 0.1503 | continuous_sd |
| bargaining_037 | bargaining | 10 | 10 | 6 | 0.0679 | 4 | 1 | decision:reject | 1.0000 | 0.0679 | continuous_sd |
| bargaining_042 | bargaining | 2 | 2 | 0 | n/a | 2 | 1 | decision:reject | 1.0000 | 0.0000 | categorical_1_minus_modal_share |
| bargaining_046 | bargaining | 4 | 4 | 2 | 0.3536 | 2 | 2 | decision:accept | 0.5000 | 0.3536 | continuous_sd |
| bargaining_047 | bargaining | 2 | 2 | 2 | 0.0000 | 0 | 0 | None | n/a | 0.0000 | continuous_sd |
| bargaining_056 | bargaining | 5 | 5 | 3 | 0.1155 | 2 | 1 | decision:accept | 1.0000 | 0.1155 | continuous_sd |
| bargaining_058 | bargaining | 2 | 2 | 1 | n/a | 1 | 1 | decision:reject | 1.0000 | n/a | None |
| bargaining_060 | bargaining | 2 | 2 | 1 | n/a | 1 | 1 | decision:reject | 1.0000 | n/a | None |
| bargaining_061 | bargaining | 2 | 2 | 1 | n/a | 1 | 1 | decision:reject | 1.0000 | n/a | None |
| bargaining_062 | bargaining | 2 | 2 | 1 | n/a | 1 | 1 | decision:reject | 1.0000 | n/a | None |
| bargaining_064 | bargaining | 2 | 2 | 1 | n/a | 1 | 1 | decision:reject | 1.0000 | n/a | None |
| bargaining_066 | bargaining | 2 | 2 | 1 | n/a | 1 | 1 | decision:reject | 1.0000 | n/a | None |
| bargaining_068 | bargaining | 2 | 2 | 1 | n/a | 1 | 1 | decision:reject | 1.0000 | n/a | None |
| bargaining_069 | bargaining | 2 | 2 | 1 | n/a | 1 | 1 | decision:reject | 1.0000 | n/a | None |
| bargaining_070 | bargaining | 2 | 2 | 1 | n/a | 1 | 1 | decision:reject | 1.0000 | n/a | None |
| negotiation_002 | negotiation | 10 | 10 | 0 | n/a | 10 | 2 | decision:rejectoffer | 0.9000 | 0.1000 | categorical_1_minus_modal_share |
| negotiation_005 | negotiation | 10 | 10 | 0 | n/a | 10 | 1 | decision:rejectoffer | 1.0000 | 0.0000 | categorical_1_minus_modal_share |
| negotiation_007 | negotiation | 10 | 10 | 0 | n/a | 10 | 1 | decision:rejectoffer | 1.0000 | 0.0000 | categorical_1_minus_modal_share |
| negotiation_008 | negotiation | 10 | 10 | 0 | n/a | 10 | 1 | decision:rejectoffer | 1.0000 | 0.0000 | categorical_1_minus_modal_share |
| negotiation_012 | negotiation | 10 | 10 | 0 | n/a | 10 | 1 | decision:rejectoffer | 1.0000 | 0.0000 | categorical_1_minus_modal_share |
| negotiation_014 | negotiation | 10 | 10 | 0 | n/a | 10 | 1 | decision:rejectoffer | 1.0000 | 0.0000 | categorical_1_minus_modal_share |
| negotiation_015 | negotiation | 10 | 10 | 0 | n/a | 10 | 1 | decision:rejectoffer | 1.0000 | 0.0000 | categorical_1_minus_modal_share |
| negotiation_016 | negotiation | 10 | 10 | 0 | n/a | 10 | 2 | decision:rejectoffer | 0.9000 | 0.1000 | categorical_1_minus_modal_share |
| negotiation_017 | negotiation | 10 | 10 | 0 | n/a | 10 | 1 | decision:rejectoffer | 1.0000 | 0.0000 | categorical_1_minus_modal_share |
| negotiation_020 | negotiation | 10 | 10 | 0 | n/a | 10 | 2 | decision:rejectoffer | 0.8000 | 0.2000 | categorical_1_minus_modal_share |
| negotiation_022 | negotiation | 10 | 10 | 0 | n/a | 10 | 1 | decision:rejectoffer | 1.0000 | 0.0000 | categorical_1_minus_modal_share |
| negotiation_024 | negotiation | 10 | 10 | 0 | n/a | 10 | 2 | decision:rejectoffer | 0.9000 | 0.1000 | categorical_1_minus_modal_share |
| negotiation_028 | negotiation | 10 | 10 | 0 | n/a | 10 | 1 | decision:rejectoffer | 1.0000 | 0.0000 | categorical_1_minus_modal_share |
| negotiation_029 | negotiation | 9 | 9 | 0 | n/a | 9 | 2 | decision:rejectoffer | 0.8889 | 0.1111 | categorical_1_minus_modal_share |
| negotiation_030 | negotiation | 10 | 10 | 0 | n/a | 10 | 2 | decision:rejectoffer | 0.8000 | 0.2000 | categorical_1_minus_modal_share |
| negotiation_033 | negotiation | 7 | 7 | 0 | n/a | 7 | 2 | decision:rejectoffer | 0.7143 | 0.2857 | categorical_1_minus_modal_share |
| negotiation_034 | negotiation | 10 | 10 | 0 | n/a | 10 | 2 | decision:rejectoffer | 0.7000 | 0.3000 | categorical_1_minus_modal_share |
| negotiation_040 | negotiation | 5 | 5 | 0 | n/a | 5 | 1 | decision:rejectoffer | 1.0000 | 0.0000 | categorical_1_minus_modal_share |
| negotiation_041 | negotiation | 5 | 5 | 0 | n/a | 5 | 2 | decision:rejectoffer | 0.8000 | 0.2000 | categorical_1_minus_modal_share |
| negotiation_044 | negotiation | 4 | 4 | 0 | n/a | 4 | 1 | decision:rejectoffer | 1.0000 | 0.0000 | categorical_1_minus_modal_share |
| negotiation_051 | negotiation | 3 | 3 | 0 | n/a | 3 | 1 | decision:rejectoffer | 1.0000 | 0.0000 | categorical_1_minus_modal_share |
| negotiation_054 | negotiation | 3 | 3 | 0 | n/a | 3 | 1 | decision:rejectoffer | 1.0000 | 0.0000 | categorical_1_minus_modal_share |
| negotiation_055 | negotiation | 3 | 3 | 0 | n/a | 3 | 1 | decision:rejectoffer | 1.0000 | 0.0000 | categorical_1_minus_modal_share |
| negotiation_062 | negotiation | 3 | 3 | 0 | n/a | 3 | 1 | decision:rejectoffer | 1.0000 | 0.0000 | categorical_1_minus_modal_share |
| negotiation_063 | negotiation | 3 | 3 | 0 | n/a | 3 | 1 | decision:rejectoffer | 1.0000 | 0.0000 | categorical_1_minus_modal_share |
| negotiation_064 | negotiation | 3 | 3 | 0 | n/a | 3 | 1 | decision:rejectoffer | 1.0000 | 0.0000 | categorical_1_minus_modal_share |
| negotiation_065 | negotiation | 3 | 3 | 0 | n/a | 3 | 1 | decision:rejectoffer | 1.0000 | 0.0000 | categorical_1_minus_modal_share |
| negotiation_067 | negotiation | 3 | 3 | 0 | n/a | 3 | 1 | decision:rejectoffer | 1.0000 | 0.0000 | categorical_1_minus_modal_share |
| negotiation_069 | negotiation | 3 | 3 | 0 | n/a | 3 | 1 | decision:rejectoffer | 1.0000 | 0.0000 | categorical_1_minus_modal_share |
| negotiation_070 | negotiation | 3 | 3 | 0 | n/a | 3 | 1 | decision:rejectoffer | 1.0000 | 0.0000 | categorical_1_minus_modal_share |
| negotiation_072 | negotiation | 3 | 3 | 0 | n/a | 3 | 1 | decision:rejectoffer | 1.0000 | 0.0000 | categorical_1_minus_modal_share |
| negotiation_076 | negotiation | 2 | 2 | 0 | n/a | 2 | 1 | decision:rejectoffer | 1.0000 | 0.0000 | categorical_1_minus_modal_share |
| negotiation_077 | negotiation | 2 | 2 | 0 | n/a | 2 | 1 | decision:rejectoffer | 1.0000 | 0.0000 | categorical_1_minus_modal_share |
| negotiation_079 | negotiation | 2 | 2 | 0 | n/a | 2 | 1 | decision:rejectoffer | 1.0000 | 0.0000 | categorical_1_minus_modal_share |
| negotiation_081 | negotiation | 2 | 2 | 0 | n/a | 2 | 2 | decision:walkaway | 0.5000 | 0.5000 | categorical_1_minus_modal_share |
| persuasion_011 | persuasion | 10 | 10 | 0 | n/a | 10 | 2 | decision:yes | 0.9000 | 0.1000 | categorical_1_minus_modal_share |
| persuasion_015 | persuasion | 10 | 10 | 0 | n/a | 10 | 2 | decision:yes | 0.7000 | 0.3000 | categorical_1_minus_modal_share |
| persuasion_016 | persuasion | 10 | 10 | 0 | n/a | 10 | 3 | decision:no | 0.8000 | 0.2000 | categorical_1_minus_modal_share |
| persuasion_025 | persuasion | 10 | 10 | 0 | n/a | 10 | 3 | decision:yes | 0.5000 | 0.5000 | categorical_1_minus_modal_share |
| persuasion_030 | persuasion | 10 | 10 | 0 | n/a | 10 | 3 | message:present | 0.7000 | 0.3000 | categorical_1_minus_modal_share |
| persuasion_042 | persuasion | 10 | 10 | 0 | n/a | 10 | 3 | decision:no | 0.7000 | 0.3000 | categorical_1_minus_modal_share |
| persuasion_052 | persuasion | 10 | 10 | 0 | n/a | 10 | 2 | decision:yes | 0.5000 | 0.5000 | categorical_1_minus_modal_share |
| persuasion_058 | persuasion | 10 | 10 | 0 | n/a | 10 | 1 | decision:yes | 1.0000 | 0.0000 | categorical_1_minus_modal_share |
| persuasion_065 | persuasion | 10 | 10 | 0 | n/a | 10 | 2 | decision:yes | 0.6000 | 0.4000 | categorical_1_minus_modal_share |
| persuasion_067 | persuasion | 10 | 10 | 0 | n/a | 10 | 3 | decision:no | 0.6000 | 0.4000 | categorical_1_minus_modal_share |
| persuasion_075 | persuasion | 10 | 10 | 0 | n/a | 10 | 3 | decision:yes | 0.6000 | 0.4000 | categorical_1_minus_modal_share |
| persuasion_076 | persuasion | 10 | 10 | 0 | n/a | 10 | 3 | decision:no | 0.6000 | 0.4000 | categorical_1_minus_modal_share |
| persuasion_084 | persuasion | 10 | 10 | 0 | n/a | 10 | 3 | decision:no | 0.8000 | 0.2000 | categorical_1_minus_modal_share |
| persuasion_103 | persuasion | 10 | 10 | 0 | n/a | 10 | 2 | decision:yes | 0.9000 | 0.1000 | categorical_1_minus_modal_share |
| persuasion_111 | persuasion | 10 | 10 | 0 | n/a | 10 | 3 | decision:yes | 0.7000 | 0.3000 | categorical_1_minus_modal_share |
| persuasion_113 | persuasion | 10 | 10 | 0 | n/a | 10 | 2 | decision:yes | 0.7000 | 0.3000 | categorical_1_minus_modal_share |
| persuasion_114 | persuasion | 10 | 10 | 0 | n/a | 10 | 2 | decision:yes | 0.9000 | 0.1000 | categorical_1_minus_modal_share |
| persuasion_126 | persuasion | 10 | 10 | 0 | n/a | 10 | 2 | decision:no | 0.6000 | 0.4000 | categorical_1_minus_modal_share |
| persuasion_130 | persuasion | 10 | 10 | 0 | n/a | 10 | 3 | decision:no | 0.5000 | 0.5000 | categorical_1_minus_modal_share |
| persuasion_138 | persuasion | 10 | 10 | 0 | n/a | 10 | 3 | decision:no | 0.7000 | 0.3000 | categorical_1_minus_modal_share |
| persuasion_149 | persuasion | 10 | 10 | 0 | n/a | 10 | 3 | decision:yes | 0.7000 | 0.3000 | categorical_1_minus_modal_share |
| persuasion_156 | persuasion | 10 | 10 | 0 | n/a | 10 | 3 | decision:yes | 0.4000 | 0.6000 | categorical_1_minus_modal_share |
| persuasion_171 | persuasion | 10 | 10 | 0 | n/a | 10 | 3 | decision:no | 0.5000 | 0.5000 | categorical_1_minus_modal_share |
| persuasion_184 | persuasion | 10 | 10 | 0 | n/a | 10 | 2 | decision:yes | 0.9000 | 0.1000 | categorical_1_minus_modal_share |
| persuasion_188 | persuasion | 10 | 10 | 0 | n/a | 10 | 3 | decision:yes | 0.7000 | 0.3000 | categorical_1_minus_modal_share |
| persuasion_194 | persuasion | 10 | 10 | 0 | n/a | 10 | 3 | decision:yes | 0.6000 | 0.4000 | categorical_1_minus_modal_share |
| persuasion_196 | persuasion | 10 | 10 | 0 | n/a | 10 | 3 | decision:no | 0.5000 | 0.5000 | categorical_1_minus_modal_share |
| persuasion_204 | persuasion | 10 | 10 | 0 | n/a | 10 | 3 | decision:yes | 0.6000 | 0.4000 | categorical_1_minus_modal_share |
| persuasion_207 | persuasion | 10 | 10 | 0 | n/a | 10 | 3 | decision:no | 0.6000 | 0.4000 | categorical_1_minus_modal_share |
| persuasion_213 | persuasion | 10 | 10 | 0 | n/a | 10 | 3 | decision:yes | 0.5000 | 0.5000 | categorical_1_minus_modal_share |
| persuasion_216 | persuasion | 10 | 10 | 0 | n/a | 10 | 3 | decision:yes | 0.5000 | 0.5000 | categorical_1_minus_modal_share |
| persuasion_219 | persuasion | 10 | 10 | 0 | n/a | 10 | 3 | message:present | 0.4000 | 0.6000 | categorical_1_minus_modal_share |
| persuasion_227 | persuasion | 10 | 10 | 0 | n/a | 10 | 2 | decision:no | 0.6000 | 0.4000 | categorical_1_minus_modal_share |
| persuasion_233 | persuasion | 10 | 10 | 0 | n/a | 10 | 3 | decision:no | 0.7000 | 0.3000 | categorical_1_minus_modal_share |
| persuasion_236 | persuasion | 10 | 10 | 0 | n/a | 10 | 3 | decision:no | 0.6000 | 0.4000 | categorical_1_minus_modal_share |

### Family summary

| family | n_clusters | n_clusters_with_action_spread | mean_action_spread |
|---|---|---|---|
| bargaining | 35 | 26 | 0.1030 |
| negotiation | 35 | 35 | 0.0599 |
| persuasion | 35 | 35 | 0.3486 |

### Shuffled-cluster floor for action_spread (1000 perms)

| family | real | shuffled_mean | ci_low | ci_high | n_valid_perms |
|---|---|---|---|---|---|
| bargaining | 0.1030 | 0.1220 | 0.0919 | 0.1565 | 1000/1000 |
| negotiation | 0.0599 | 0.0659 | 0.0475 | 0.0876 | 1000/1000 |
| persuasion | 0.3486 | 0.4669 | 0.4371 | 0.4914 | 1000/1000 |

## 16. Cluster-membership homogeneity (round / phase / history length)

| family | n_clusters | n_clusters_mixed_phase | mean_round_modal_share | mean_phase_modal_share | mean_history_length_modal_share | mean_history_length_sd |
|---|---|---|---|---|---|---|
| bargaining | 35 | 32 | 0.9200 | 0.6388 | 0.9200 | 0.1340 |
| negotiation | 35 | 0 | 0.9914 | 1.0000 | 0.9914 | 0.0276 |
| persuasion | 35 | 30 | 0.4314 | 0.7343 | 0.4314 | 1.1238 |

## 17. Action spread vs. judge-mean spread (Spearman, per family)

| family | n_clusters_used | spearman_rho | spearman_p |
|---|---|---|---|
| bargaining | 26 | 0.0431 | 0.8345 |
| negotiation | 35 | -0.0130 | 0.9409 |
| persuasion | 35 | 0.5147 | 0.0016 |
