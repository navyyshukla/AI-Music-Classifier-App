# AI Music Genre Classifier

Upload a song, get a genre, and a straight answer when the model is **not sure**.

The app analyses the **whole song** (evenly spaced 10 s sections), embeds each with a pretrained
**Audio Spectrogram Transformer** (AudioSet), classifies with a small calibrated head, and combines the
sections. Low confidence or disagreement between sections shows "Not sure" instead of a forced label.

Numbers below come from `reports/metrics.json` (567 held-out test tracks, split by track, 13 genres).

| Metric | Value |
|---|---|
| Accuracy / macro-F1 | 68.3% / 0.66 |
| Accuracy by source | GTZAN 84.0%, FMA 64.9% |
| Expected calibration error | 0.04 |
| Accuracy on accepted predictions / coverage | 79.5% on 77.4% of tracks (abstains below 50% confidence) |
| Modern-song check (`reports/ood_summary.json`) | not run yet: needs your own songs in `Data/ood/` |

## Run the app

```bash
git clone https://github.com/navyyshukla/AI-Music-Classifier-App
cd AI-Music-Classifier-App
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # ffmpeg is needed for MP3: brew install ffmpeg
streamlit run app.py
```

## Train it yourself

1. Download into `Data/`:
   - GTZAN → `Data/genres_original/<genre>/*.wav`
   - FMA metadata → `Data/fma_metadata/tracks.csv`, FMA-medium audio → `Data/fma_medium/<NNN>/<NNNNNN>.mp3`
     (<https://github.com/mdeff/fma>)
2. Embed tracks once (resumable, a couple of hours on a laptop for ~8k tracks):
   `python -m genre_classifier.extract --per-class 600`
3. Train, calibrate and evaluate: `python -m genre_classifier.train`
   (writes `models/genre_head.joblib`, `reports/metrics.json`, confusion matrix, reliability diagram)
4. Check on real modern songs: put files in `Data/ood/<genre>/song.mp3` then
   `python -m genre_classifier.evaluate --ood-dir Data/ood`
5. `pytest -q`

## Design

```
audio -> 16 kHz mono -> up to 18 evenly spaced 10 s windows (silence dropped)
      -> AST embedding (768-d) per window -> scaler + logistic regression
      -> mean of logits over windows -> temperature scaling -> probabilities
      -> abstain if confidence < tau or windows disagree (tau tuned on validation data)
```

* **No leakage:** splits are by track (FMA: official artist-aware split; GTZAN: stratified by track);
  `data.check_no_leakage` fails training if an artist spans two splits. The test set is read once.
* **13 genres:** blues, classical, country, disco, electronic, folk, hip-hop, jazz, metal, pop, reggae,
  rock, soul/R&B (GTZAN + FMA, mapping in `genre_classifier/data.py`).
* Small model file (KBs), no Git LFS, no TensorFlow.

## Limitations

* A closed set of 13 genres; real songs are often blends. Treat the output as a guess, not a fact.
* GTZAN is old and has known duplicates/mislabels; FMA labels are user-submitted and noisy.
  Disco, metal and reggae come only from GTZAN's small sample.
* Calibration is measured on in-distribution data. The modern-song check is the honest test of
  generalisation; extend it before trusting the thresholds.
* The AST encoder needs ~1 GB RAM; first load downloads ~350 MB of weights.

## What went wrong in v1

The first version (kept in `legacy/`) reported ~96% accuracy because augmented spectrogram images of the
same track were split across train and validation. It also analysed only the first 30 seconds, forced a
softmax answer over 10 genres, and kept results inside a button callback so charts vanished on any
interaction. v2 fixes the split, uses a pretrained encoder, analyses the whole song, calibrates and
abstains, and renders results from session state.

## Roadmap

Fine-tune the encoder's top layers; add multi-label tags and more modern data; per-genre abstain
thresholds; compare against the v1 CNN on the same leak-free split.
