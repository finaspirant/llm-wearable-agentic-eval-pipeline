# AgentTrace Playground

Paste a JSON agent trajectory and get back a PIA (Process Integrity Assessment)
score with per-dimension breakdown (planning quality, error recovery, goal
alignment), recovery score, outcome score, tool-call score, and failure flags.
The app imports PIAScorer and TrajectoryScorer directly from `src/` — no logic
is duplicated. To run locally: `cd apps/playground && streamlit run app.py`.
Deploy target: HuggingFace Spaces (session 13).
