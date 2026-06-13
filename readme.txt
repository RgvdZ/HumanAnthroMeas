Smooth L1 with per‑measurement weighting.
Smooth L1 behaves like squared error (L2) for tiny residuals and like absolute error (L1) for larger ones, making it robust to outliers. Per‑measurement weighting means we divide each measurement’s error by its own standard deviation (computed over the whole training set) before summing into the loss, so that a 1 mm mistake on a naturally highly variable measurement (like waist) doesn't dominate the training compared to a 1 mm mistake on a less variable one (like neck).

