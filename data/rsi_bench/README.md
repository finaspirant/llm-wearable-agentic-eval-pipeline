# data/rsi_bench -- provenance notes

- `author_ratings.csv` (case_id, valuation_reasoning, horizon_strategy_planning,
  outcome_consistency): the author rated 20 randomly chosen judged steps from
  `human_rating_sheet.md`, a sheet that showed no judge scores. The author
  knew the study's hypothesis. Single rater, n=20.
  `human_rating_key.json` (case -> cluster_id/member_id/judge scores) was
  generated at the same time as the sheet but kept in a separate file, and
  was not consulted until after `author_ratings.csv` was produced.
