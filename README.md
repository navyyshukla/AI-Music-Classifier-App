# AI Music Genre Classifier

Upload a song, get a genre, and a straight answer when the model is **not sure**.

**Live app: <https://ai-music-classifier-app.streamlit.app>**

The app analyses the **whole song** (evenly spaced 10 s sections), embeds each with a pretrained
**Audio Spectrogram Transformer** (AudioSet), classifies with a small calibrated head, and combines the
sections. Low confidence or disagreement between sections shows "Not sure" instead of a forced label.

Numbers below come from `reports/metrics.json` (973 held-out test tracks from GTZAN, FMA and Jamendo,
split by track and artist, 13 genres). Test tracks are never used for training or tuning.

| Metric | Value |
|---|---|
| Accuracy / macro-F1 (all sources) | 60.5% / 0.57 |
| Accuracy by source | GTZAN 83.0%, FMA 66.6%, Jamendo 48.0% |
| Expected calibration error | 0.03 |
| Accuracy on accepted predictions / coverage | 78.6% on 55.7% of tracks (abstains below 50% confidence) |
| Non-music rejection | white/brown noise, tone, sweep, clicks and modulated noise are all rejected; about 1% of real held-out tracks are wrongly rejected |

**Why the headline number went down.** Version 1 reported 96% (a data leak). The first honest version scored
68% on GTZAN + FMA. Adding Jamendo (modern, Creative Commons, labelled by uploaders) widens the test set to
music that sounds like current releases, and that is harder: Jamendo accuracy is 48%.

### Does adding modern data help? (Jamendo held-out artists, 406 tracks)

| | Before (no Jamendo in training) | After |
|---|---|---|
| Accuracy if forced to answer | 47.5% | 48.0% |
| Accuracy when it chooses to answer | 60.1% | 69.6% |
| Confidently wrong (>= 80% sure) | 15 | 9 |
| Metal / reggae / hip-hop recall | 36% / 8% / 40% | 82% / 39% / 57% |
| Jazz / rock recall | 65% / 43% | 44% / 32% |
| Pop / soul-R&B recall | 17% / 0% | 22% / 9% |

Reading this honestly: the extra data helped small genres and made answers more trustworthy, but it did not
fix pop and soul/R&B. A head trained **only** on Jamendo and tested on Jamendo also reaches just 49%, so the
limit is the labels, not the amount of data: Jamendo's genre tags are chosen by uploaders and overlap heavily
(pop tracks are most often predicted electronic, rock tracks metal, soul/R&B tracks pop). Fixing that needs
cleaner labels for mainstream pop/R&B (for example your own labelled library) or a multi-label output, not more
of the same tags. Reproduce with `python -m genre_classifier.compare --source jamendo`.

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
   - Optional, modern music: MTG-Jamendo (<https://github.com/MTG/mtg-jamendo-dataset>). Put
     `autotagging_genre.tsv` and `raw_30s_audio-low-NN.tar` files in `Data/jamendo/` (and `tars/`), then
     `python -m genre_classifier.jamendo prepare` unpacks only the tracks that map onto our genres.
2. Embed tracks once (resumable, about 1.5 hours on a laptop for ~9k tracks):
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
      -> reject as "not music" if the encoder's AudioSet "Music" score is low
         or the embedding is far from every genre (Mahalanobis); both thresholds set on validation data
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
