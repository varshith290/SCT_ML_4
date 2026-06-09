## Why HOG + SVM

| Approach | Reason |
|----------|--------|
| HOG features | Captures shape and edge patterns — perfect for hand gestures |
| Grayscale conversion | Color is not needed for gesture shape — reduces noise |
| StandardScaler | Normalizes HOG feature magnitudes for SVM |
| PCA reduction | Reduces 1764 HOG features to 100 — faster SVM training |
| SVM RBF kernel | Best for non-linear high-dimensional feature spaces |

## Model Results

| Model | Test Accuracy | CV Accuracy |
|-------|--------------|-------------|
| SVM — RBF | ~0.95 | ~0.94 |
| SVM — Linear | ~0.90 | ~0.89 |
| Random Forest | ~0.92 | ~0.91 |
| SVM Tuned | ~0.97 | ~0.96 |

> Note: Results may vary slightly depending on images sampled and random seed.

## Hyperparameter Tuning

GridSearchCV tested the following combinations on SVM RBF:

| Parameter | Values tested |
|-----------|--------------|
| C | 1, 10, 100 |
| gamma | scale, auto, 0.001 |

Best parameters are printed at the end of the script run.

## Plots Generated

| File | Description |
|------|-------------|
| plot1_sample_gestures.png | 5 sample images per gesture class |
| plot2_hog_features.png | Original image vs HOG visualization vs class mean image |
| plot3_class_distribution.png | Number of images per gesture class |
| plot4_pca_variance.png | Cumulative and per-component PCA variance explained |
| plot5_pca_scatter.png | Gesture class separation in PCA space (PC1 vs PC2) |
| plot6_confusion_matrix.png | Default vs tuned SVM confusion matrix |
| plot7_model_comparison.png | Test accuracy and CV accuracy across all models |
| plot8_per_class_accuracy.png | Per gesture class precision scores |
| plot9_predictions.png | Correct predictions vs wrong predictions on test set |

## Key Findings

- **HOG features** are highly effective for hand gesture recognition
- **SVM with RBF kernel** outperforms Linear SVM and Random Forest
- **PCA** reduces training time significantly with minimal accuracy loss
- **Fist and Palm** gestures are the easiest to classify (distinct shapes)
- **Fist vs Fist Moved** is the hardest pair to distinguish (similar appearance)
- **Tuned SVM** achieves ~97% accuracy — suitable for real-time gesture control

## Applications

- Human-computer interaction without mouse or keyboard
- Sign language recognition for accessibility
- Gaming and virtual reality gesture controls
- Robotics and drone gesture control
- Touchless interface for medical or industrial use

## Limitations

- Works on static images only — not real-time video
- Performance depends on consistent lighting and background
- For real-time video use **MediaPipe** or **OpenCV** with this model
- For higher accuracy use a **CNN** (Convolutional Neural Network)

## Libraries Used

- `pandas` — data handling
- `numpy` — numerical operations
- `matplotlib` — base plotting
- `seaborn` — styled charts
- `scikit-learn` — SVM, PCA, StandardScaler, GridSearchCV, metrics
- `scikit-image` — HOG feature extraction
- `Pillow` — image loading and resizing
