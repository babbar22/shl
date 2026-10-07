# shl

# Grammar Scoring Engine for Spoken Audio (SHL Hiring Assessment 2026)

Predicts a 0-5 MOS grammar score from 45-60 s speech clips.

## Approach
1. Transcribe audio with Whisper-small (16 kHz mono).
2. Features: linguistic/fluency stats, GPT-2 perplexity, LanguageTool error rate, MiniLM sentence embeddings, wav2vec2 audio embeddings.
3. Models: Ridge, RBF-SVR, Gradient Boosting; PCA fitted inside each CV fold; predictions averaged and clipped to [0, 5].

## Results
| Metric | Value |
|---|---|
| Training RMSE | 0.2705 |
| 5-fold CV RMSE | 0.6217 |
| Baseline (predict mean) RMSE | 1.2382 |
| Kaggle public score | 0.4700 |

## Notes
- train.csv and test.csv reuse the same file names in separate folders; audio is matched per split folder to avoid leakage. An early submission affected by this scored 1.1935; after the fix it scored 0.4700.

## Files
- `notebookaf7d34b838.ipynb`: full executed notebook
- `submission.csv`: final predictions
