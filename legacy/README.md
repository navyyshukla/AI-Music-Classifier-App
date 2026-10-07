# Legacy v1 (spectrogram CNN)

Kept for reference only. v1 split augmented spectrogram *images* randomly, so near-duplicates of the same
track landed in both train and validation. The reported ~96% accuracy was memorisation, and the model
then labelled out-of-distribution songs with extreme confidence. Replaced by `genre_classifier/`.
