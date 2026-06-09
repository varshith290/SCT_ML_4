# ================================================================
#  Hand Gesture Recognition — SVM + HOG Features
#  Dataset: https://www.kaggle.com/datasets/gti-upm/leapgestrecog
# ================================================================

# ── 1. IMPORTS ──────────────────────────────────────────────────
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from PIL import Image
from skimage.feature import hog
from skimage import exposure
import warnings
warnings.filterwarnings("ignore")


# ── 2. CONFIGURATION ─────────────────────────────────────────────
DATA_DIR     = "archive (1)/leapGestRecog" # folder after unzipping dataset
IMG_SIZE     = 64                # resize images to 64x64
MAX_PER_CLASS = 200              # images per gesture class
RANDOM_SEED  = 42


# ── 3. EXPLORE DATASET STRUCTURE ────────────────────────────────
print("=" * 60)
print("  HAND GESTURE RECOGNITION — SVM + HOG")
print("=" * 60)

print(f"\nDataset folder : {DATA_DIR}")
print(f"\nFolder structure:")

gesture_map   = {}
gesture_names = []

for subject in sorted(os.listdir(DATA_DIR)):
    subject_path = os.path.join(DATA_DIR, subject)
    if not os.path.isdir(subject_path):
        continue
    for gesture in sorted(os.listdir(subject_path)):
        gesture_path = os.path.join(subject_path, gesture)
        if not os.path.isdir(gesture_path):
            continue
        files = [f for f in os.listdir(gesture_path)
                 if f.endswith((".png",".jpg",".jpeg"))]
        print(f"  {subject}/{gesture} — {len(files)} images")
        key = gesture
        if key not in gesture_map:
            gesture_map[key] = []
            gesture_names.append(gesture)
        gesture_map[key].extend(
            [os.path.join(gesture_path, f) for f in files]
        )

print(f"\nTotal gesture classes : {len(gesture_map)}")
for k, v in gesture_map.items():
    print(f"  {k} : {len(v)} images")


# ── 4. LOAD & PREPROCESS IMAGES ──────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 1 — LOADING IMAGES & EXTRACTING HOG FEATURES")
print("=" * 60)

def extract_hog_features(image_path, img_size=64):
    """Load image and extract HOG (Histogram of Oriented Gradients) features."""
    img = Image.open(image_path).convert("L")   # grayscale
    img = img.resize((img_size, img_size))
    arr = np.array(img)
    features, _ = hog(
        arr,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=True,
        feature_vector=True
    )
    return features, arr

X_raw    = []    # raw pixels for visualization
X_hog    = []    # HOG features for model
y_labels = []    # gesture labels

for gesture, paths in gesture_map.items():
    selected = paths[:MAX_PER_CLASS]
    print(f"  Loading {gesture} — {len(selected)} images...")
    for path in selected:
        try:
            features, raw = extract_hog_features(path, IMG_SIZE)
            X_hog.append(features)
            X_raw.append(raw)
            y_labels.append(gesture)
        except Exception as e:
            pass

X_hog    = np.array(X_hog)
X_raw    = np.array(X_raw)
y_labels = np.array(y_labels)

# Encode labels
le = LabelEncoder()
y  = le.fit_transform(y_labels)

print(f"\n✔ Images loaded")
print(f"X_hog shape : {X_hog.shape}")
print(f"X_raw shape : {X_raw.shape}")
print(f"Classes     : {le.classes_}")
print(f"Total images: {len(y)}")


# ── 5. DATA CLEANING ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 2 — DATA CLEANING")
print("=" * 60)

# Check for NaN in HOG features
nan_count = np.isnan(X_hog).sum()
print(f"\nNaN in HOG features : {nan_count}")
if nan_count > 0:
    X_hog = np.nan_to_num(X_hog)
    print("✔ NaN replaced with 0")

# Class distribution
unique, counts = np.unique(y, return_counts=True)
print(f"\nClass distribution:")
for u, c in zip(unique, counts):
    print(f"  {le.classes_[u]:<30} : {c} images")

print(f"\nTotal samples : {len(y)}")
print(f"Features (HOG): {X_hog.shape[1]}")
print(f"✔ Data is clean")


# ── 6. FEATURE ENGINEERING ───────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 3 — FEATURE ENGINEERING")
print("=" * 60)

# Normalize HOG features
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_hog)
print(f"✔ StandardScaler applied")

# PCA for dimensionality reduction
print(f"\nApplying PCA to reduce {X_scaled.shape[1]} HOG features...")
pca   = PCA(n_components=100, random_state=RANDOM_SEED)
X_pca = pca.fit_transform(X_scaled)
explained = pca.explained_variance_ratio_.cumsum()[-1] * 100
print(f"✔ PCA: {X_scaled.shape[1]} → {X_pca.shape[1]} components")
print(f"✔ Variance explained : {explained:.1f}%")


# ── 7. EDA ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 4 — EXPLORATORY DATA ANALYSIS")
print("=" * 60)

print(f"\nDataset summary:")
print(f"  Total images      : {len(y)}")
print(f"  Gesture classes   : {len(le.classes_)}")
print(f"  HOG features      : {X_hog.shape[1]}")
print(f"  After PCA         : {X_pca.shape[1]}")
print(f"  Image size        : {IMG_SIZE}x{IMG_SIZE} grayscale")

print(f"\nGesture class names:")
for i, name in enumerate(le.classes_):
    print(f"  {i:2d} — {name}")


# ── 8. TRAIN / TEST SPLIT ────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y,
    test_size=0.2,
    random_state=RANDOM_SEED,
    stratify=y
)

print(f"\nTrain size : {X_train.shape[0]}")
print(f"Test size  : {X_test.shape[0]}")


# ── 9. TRAIN MODELS ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 5 — TRAINING MODELS")
print("=" * 60)

models = {
    "SVM — RBF"       : SVC(kernel="rbf", C=10, gamma="scale",
                            probability=True, random_state=RANDOM_SEED),
    "SVM — Linear"    : SVC(kernel="linear", C=1,
                            probability=True, random_state=RANDOM_SEED),
    "Random Forest"   : RandomForestClassifier(n_estimators=200,
                            random_state=RANDOM_SEED, n_jobs=-1),
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cv  = cross_val_score(model, X_pca, y, cv=5,
                          scoring="accuracy", n_jobs=-1)

    results[name] = {
        "model"  : model,
        "y_pred" : y_pred,
        "ACC"    : acc,
        "CV_Mean": cv.mean(),
        "CV_Std" : cv.std(),
    }

    print(f"  Accuracy    : {acc:.4f}")
    print(f"  CV Mean     : {cv.mean():.4f} ± {cv.std():.4f}")
    print(f"\n  Report:\n"
          f"{classification_report(y_test, y_pred, target_names=le.classes_)}")

best_name = max(results, key=lambda k: results[k]["ACC"])
best      = results[best_name]
print(f"\n✔ Best model : {best_name} (Accuracy = {best['ACC']:.4f})")


# ── 10. HYPERPARAMETER TUNING ────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 6 — HYPERPARAMETER TUNING (SVM RBF)")
print("=" * 60)

param_grid = {
    "C"     : [1, 10, 100],
    "gamma" : ["scale", "auto", 0.001],
}

grid = GridSearchCV(
    SVC(kernel="rbf", probability=True, random_state=RANDOM_SEED),
    param_grid, cv=3, scoring="accuracy",
    n_jobs=-1, verbose=1
)
grid.fit(X_train, y_train)

print(f"\nBest params   : {grid.best_params_}")
print(f"Best CV score : {grid.best_score_:.4f}")

tuned_model  = grid.best_estimator_
y_pred_tuned = tuned_model.predict(X_test)
acc_tuned    = accuracy_score(y_test, y_pred_tuned)

print(f"\nTuned accuracy : {acc_tuned:.4f}")
print(f"\n{classification_report(y_test, y_pred_tuned, target_names=le.classes_)}")


# ── 11. VISUALIZATIONS ───────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 7 — SAVING PLOTS")
print("=" * 60)

class_colors = [
    "#3266ad","#E24B4A","#1D9E75","#BA7517","#533AB7",
    "#63991e","#D85A30","#1a7abf","#c0392b","#8e44ad"
]


# ── Plot 1: Sample images per gesture
n_classes = len(le.classes_)
fig, axes = plt.subplots(n_classes, 5,
                          figsize=(12, n_classes * 2))
fig.suptitle("Sample Images per Gesture Class",
             fontsize=14, fontweight="bold")

for i, gesture in enumerate(le.classes_):
    indices = np.where(y_labels == gesture)[0][:5]
    for j, idx in enumerate(indices):
        ax = axes[i, j] if n_classes > 1 else axes[j]
        ax.imshow(X_raw[idx], cmap="gray")
        ax.axis("off")
        if j == 0:
            ax.set_title(gesture[:20], fontsize=7,
                         loc="left", pad=2)

plt.tight_layout()
plt.savefig("plot1_sample_gestures.png",
            dpi=150, bbox_inches="tight")
plt.show()
print("✔ Saved: plot1_sample_gestures.png")


# ── Plot 2: HOG feature visualization
fig, axes = plt.subplots(3, len(le.classes_),
                          figsize=(len(le.classes_) * 2, 7))
fig.suptitle("HOG Feature Visualization per Gesture",
             fontsize=13, fontweight="bold")

for i, gesture in enumerate(le.classes_):
    idx = np.where(y_labels == gesture)[0][0]
    img_arr = X_raw[idx]

    # Original
    axes[0, i].imshow(img_arr, cmap="gray")
    axes[0, i].axis("off")
    axes[0, i].set_title(gesture[:12], fontsize=7)

    # HOG visualization
    _, hog_img = hog(
        img_arr,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=True,
        feature_vector=True
    )
    hog_img_rescaled = exposure.rescale_intensity(
        hog_img, in_range=(0, 10)
    )
    axes[1, i].imshow(hog_img_rescaled, cmap="gray")
    axes[1, i].axis("off")

    # Mean image per class
    class_images = X_raw[y_labels == gesture]
    mean_img     = class_images.mean(axis=0)
    axes[2, i].imshow(mean_img, cmap="gray")
    axes[2, i].axis("off")

axes[0, 0].set_ylabel("Original",   fontsize=9)
axes[1, 0].set_ylabel("HOG",        fontsize=9)
axes[2, 0].set_ylabel("Class Mean", fontsize=9)

plt.tight_layout()
plt.savefig("plot2_hog_features.png",
            dpi=150, bbox_inches="tight")
plt.show()
print("✔ Saved: plot2_hog_features.png")


# ── Plot 3: Class distribution
plt.figure(figsize=(12, 5))
class_counts = pd.Series(y_labels).value_counts()
bars = plt.bar(class_counts.index,
               class_counts.values,
               color=class_colors[:len(class_counts)])
plt.title("Number of Images per Gesture Class",
          fontsize=13, fontweight="bold")
plt.xlabel("Gesture")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
for bar, val in zip(bars, class_counts.values):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 1,
             str(val), ha="center", fontsize=9)
plt.tight_layout()
plt.savefig("plot3_class_distribution.png",
            dpi=150, bbox_inches="tight")
plt.show()
print("✔ Saved: plot3_class_distribution.png")


# ── Plot 4: PCA explained variance
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("PCA Analysis", fontsize=13, fontweight="bold")

cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
axes[0].plot(cumvar, color="#3266ad", linewidth=2)
axes[0].axhline(95, color="#E24B4A", linestyle="--",
                label="95% variance")
axes[0].axhline(explained, color="#1D9E75", linestyle="--",
                label=f"{explained:.1f}% (100 components)")
axes[0].set_title("Cumulative Explained Variance")
axes[0].set_xlabel("PCA Components")
axes[0].set_ylabel("Variance Explained (%)")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].bar(range(20),
            pca.explained_variance_ratio_[:20] * 100,
            color="#3266ad")
axes[1].set_title("Top 20 Components")
axes[1].set_xlabel("PCA Component")
axes[1].set_ylabel("Variance Explained (%)")
axes[1].grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("plot4_pca_variance.png",
            dpi=150, bbox_inches="tight")
plt.show()
print("✔ Saved: plot4_pca_variance.png")


# ── Plot 5: PCA scatter (first 2 components)
plt.figure(figsize=(11, 7))
for i, gesture in enumerate(le.classes_):
    mask = y == i
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                s=20, alpha=0.5,
                color=class_colors[i % len(class_colors)],
                label=gesture)
plt.title("PCA Scatter — Gesture Classes (PC1 vs PC2)",
          fontsize=13, fontweight="bold")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(fontsize=8, markerscale=2,
           bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig("plot5_pca_scatter.png",
            dpi=150, bbox_inches="tight")
plt.show()
print("✔ Saved: plot5_pca_scatter.png")


# ── Plot 6: Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Confusion Matrices",
             fontsize=13, fontweight="bold")

ConfusionMatrixDisplay(
    confusion_matrix(y_test,
                     results["SVM — RBF"]["y_pred"]),
    display_labels=le.classes_
).plot(ax=axes[0], cmap="Blues",
       colorbar=False, xticks_rotation=45)
axes[0].set_title("SVM RBF (default)")

ConfusionMatrixDisplay(
    confusion_matrix(y_test, y_pred_tuned),
    display_labels=le.classes_
).plot(ax=axes[1], cmap="Blues",
       colorbar=False, xticks_rotation=45)
axes[1].set_title("SVM RBF (tuned)")

plt.tight_layout()
plt.savefig("plot6_confusion_matrix.png",
            dpi=150, bbox_inches="tight")
plt.show()
print("✔ Saved: plot6_confusion_matrix.png")


# ── Plot 7: Model comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Model Comparison",
             fontsize=13, fontweight="bold")

all_names = list(results.keys()) + ["SVM Tuned"]
all_accs  = [results[m]["ACC"] for m in results] + [acc_tuned]
all_cvs   = [results[m]["CV_Mean"] for m in results] + [acc_tuned]
bar_cols  = ["#3266ad","#1D9E75","#BA7517","#E24B4A"]

axes[0].bar(all_names, all_accs, color=bar_cols)
axes[0].set_title("Test Accuracy")
axes[0].set_ylabel("Accuracy")
axes[0].set_ylim(0, 1)
axes[0].tick_params(axis="x", rotation=20)
for i, v in enumerate(all_accs):
    axes[0].text(i, v + 0.005, f"{v:.4f}",
                 ha="center", fontsize=9)

axes[1].bar(list(results.keys()),
            [results[m]["CV_Mean"] for m in results],
            color=bar_cols[:3],
            yerr=[results[m]["CV_Std"] for m in results],
            capsize=5)
axes[1].set_title("Cross Validation Accuracy (5-fold)")
axes[1].set_ylabel("CV Accuracy")
axes[1].set_ylim(0, 1)
axes[1].tick_params(axis="x", rotation=20)

plt.tight_layout()
plt.savefig("plot7_model_comparison.png",
            dpi=150, bbox_inches="tight")
plt.show()
print("✔ Saved: plot7_model_comparison.png")


# ── Plot 8: Per-class accuracy
report_dict = classification_report(
    y_test, y_pred_tuned,
    target_names=le.classes_,
    output_dict=True
)
class_acc = {k: v["precision"]
             for k, v in report_dict.items()
             if k in le.classes_}

plt.figure(figsize=(12, 5))
bars = plt.bar(class_acc.keys(),
               class_acc.values(),
               color=class_colors[:len(class_acc)])
plt.axhline(acc_tuned, color="red", linestyle="--",
            linewidth=1.5,
            label=f"Overall accuracy = {acc_tuned:.3f}")
plt.title("Per-Class Precision — Tuned SVM",
          fontsize=13, fontweight="bold")
plt.xlabel("Gesture Class")
plt.ylabel("Precision")
plt.ylim(0, 1.1)
plt.xticks(rotation=45, ha="right")
plt.legend(fontsize=9)
for bar, val in zip(bars, class_acc.values()):
    plt.text(bar.get_x() + bar.get_width()/2,
             val + 0.01,
             f"{val:.2f}", ha="center", fontsize=8)
plt.tight_layout()
plt.savefig("plot8_per_class_accuracy.png",
            dpi=150, bbox_inches="tight")
plt.show()
print("✔ Saved: plot8_per_class_accuracy.png")


# ── Plot 9: Prediction samples
fig, axes = plt.subplots(2, 8, figsize=(16, 5))
fig.suptitle("Predictions  ✔ Correct (top)   ✘ Wrong (bottom)",
             fontsize=13, fontweight="bold")

test_idx = np.arange(len(X_test))
correct  = test_idx[y_pred_tuned == y_test][:8]
wrong    = test_idx[y_pred_tuned != y_test][:8]

X_test_raw = X_raw[
    np.where(np.isin(
        np.arange(len(y)),
        np.concatenate([
            np.where(y == c)[0][:int(MAX_PER_CLASS * 0.2)]
            for c in range(len(le.classes_))
        ])
    ))[0]
]

for i in range(min(8, len(correct))):
    idx = correct[i]
    if idx < len(X_test_raw):
        axes[0, i].imshow(X_test_raw[idx], cmap="gray")
    axes[0, i].axis("off")
    pred_name = le.classes_[y_pred_tuned[idx]]
    axes[0, i].set_title(f"✔ {pred_name[:10]}",
                          fontsize=7, color="green")

for i in range(min(8, len(wrong))):
    idx = wrong[i]
    if idx < len(X_test_raw):
        axes[1, i].imshow(X_test_raw[idx], cmap="gray")
    axes[1, i].axis("off")
    pred_name  = le.classes_[y_pred_tuned[idx]]
    truth_name = le.classes_[y_test[idx]]
    axes[1, i].set_title(
        f"✘ {pred_name[:8]}\n({truth_name[:8]})",
        fontsize=6, color="red"
    )

plt.tight_layout()
plt.savefig("plot9_predictions.png",
            dpi=150, bbox_inches="tight")
plt.show()
print("✔ Saved: plot9_predictions.png")


# ── 12. PREDICT A SINGLE IMAGE ───────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 8 — PREDICT A SINGLE IMAGE")
print("=" * 60)

def predict_gesture(image_path, model, scaler,
                    pca, le, img_size=64):
    img = Image.open(image_path).convert("L")
    img = img.resize((img_size, img_size))
    arr = np.array(img)
    features, _ = hog(
        arr,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=True,
        feature_vector=True
    )
    features = scaler.transform([features])
    features = pca.transform(features)
    pred     = model.predict(features)[0]
    probs    = model.predict_proba(features)[0]
    label    = le.classes_[pred]
    conf     = probs.max()

    print(f"  Image      : {image_path}")
    print(f"  Prediction : {label}")
    print(f"  Confidence : {conf:.4f}")
    print(f"\n  All class probabilities:")
    for i, (cls, prob) in enumerate(
            zip(le.classes_, probs)):
        bar = "█" * int(prob * 30)
        print(f"  {cls:<30} {bar} {prob:.4f}")
    return label

# Test with first image found
sample_img = None
for gesture, paths in gesture_map.items():
    if paths:
        sample_img = paths[0]
        break

if sample_img:
    predict_gesture(sample_img, tuned_model,
                    scaler, pca, le, IMG_SIZE)


# ── 13. FINAL SUMMARY ────────────────────────────────────────────
print("\n" + "=" * 60)
print("  FINAL SUMMARY")
print("=" * 60)
print(f"  Total images      : {len(y)}")
print(f"  Gesture classes   : {len(le.classes_)}")
print(f"  Image size        : {IMG_SIZE}x{IMG_SIZE}")
print(f"  HOG features      : {X_hog.shape[1]}")
print(f"  After PCA         : {X_pca.shape[1]}")
print(f"  Train size        : {X_train.shape[0]}")
print(f"  Test size         : {X_test.shape[0]}")

print(f"\n  Model Results:")
for name, res in results.items():
    print(f"  {name:<22} "
          f"Acc={res['ACC']:.4f}  "
          f"CV={res['CV_Mean']:.4f}±{res['CV_Std']:.4f}")
print(f"  {'SVM Tuned':<22} "
      f"Acc={acc_tuned:.4f}  "
      f"Params={grid.best_params_}")

print(f"\n  Best model : SVM Tuned "
      f"(Acc = {acc_tuned:.4f})")
print(f"\n✔ All 9 plots saved. Script complete.")
print("=" * 60)