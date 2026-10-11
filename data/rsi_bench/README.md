# data/rsi_bench -- provenance notes

- `author_ratings.csv` (case_id, valuation_reasoning, horizon_strategy_planning,
  outcome_consistency): the 20 cases in `human_rating_sheet.md` were rated
  blind -- that sheet shows family, game state, valid actions, reasoning,
  and action for each case with NO judge score of any kind visible, and no
  other model's ratings were shown to the rater before or during rating.
  `human_rating_key.json` (case -> cluster_id/member_id/judge scores) was
  generated at the same time as the sheet but kept in a separate file for
  exactly this reason, and was not consulted until after `author_ratings.csv`
  was produced.
