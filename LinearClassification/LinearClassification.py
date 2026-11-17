from pathlib import Path
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = Path("/Users/juhno1023/Desktop/동국대학교/4학년 1학기/cg/cifar-10-batches-py")
OUTPUT_DIR = Path(__file__).resolve().parent / "results"
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

def load_cifar10_batch(file):
    with open(file, "rb") as fo:
        batch = pickle.load(fo, encoding="bytes")
    return batch[b"data"], np.array(batch[b"labels"])


class LinearClassifier:
    def __init__(self, input_dim, num_classes, seed=42):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, 0.001, (input_dim, num_classes))
        self.b = np.zeros((1, num_classes))

    def scores(self, X):
        # X: (N, 3072), W: (3072, 10), b: (1, 10)
        return X @ self.W + self.b

    def predict(self, X):
        return np.argmax(self.scores(X), axis=1)

    def svm_loss(self, X, y):
        scores = self.scores(X)
        N = len(y)
        correct_scores = scores[np.arange(N), y].reshape(-1, 1)
        #Subtract the true class score from each class score and compare with 0, if negative value margin is set to 0, 
        margins = np.maximum(0, scores - correct_scores + 1.0)
        margins[np.arange(N), y] = 0
        losses = margins.sum(axis=1)

        return losses, losses.mean()

    def softmax(self, X):
        # exp^score / sum of exp^scores 
        scores = self.scores(X)
        #Subtract maximum value to prevent overflow
        shifted = scores - scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(shifted)

        return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def to_image(vector):
    return vector.reshape(3, 32, 32).transpose(1, 2, 0)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    X_test, y_test = load_cifar10_batch(DATA_DIR / "test_batch")

    num_test = 20  # choose 20 images for test
    raw_images = X_test[:num_test]
    X = raw_images.astype(np.float64) / 255.0
    y = y_test[:num_test]

    #Classifitaction
    classifier = LinearClassifier(input_dim=3072, num_classes=10, seed=42)

    scores = classifier.scores(X)
    preds = classifier.predict(X)
    losses, mean_loss = classifier.svm_loss(X, y)
    probs = classifier.softmax(X)

    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-12, rtol=0)
    np.testing.assert_array_equal(probs.argmax(axis=1), preds)

    # print output
    with (OUTPUT_DIR / "results.txt").open("w", encoding="utf-8") as file:
        def show(text=""):
            print(text)
            print(text, file=file)

        np.set_printoptions(precision=6, threshold=100, edgeitems=5, linewidth=160)
        show("Seed: 42; W ~ Normal(0, 0.001^2); b = 0; no training")
        show("X @ W + b: (N, 3072) @ (3072, 10) + (1, 10) = (N, 10)")
        show("Equivalent column-vector formula: f(x) = W.T @ x + b.T")
        show("W rows: RGB pixels; W columns: " + ", ".join(CLASS_NAMES))
        show(f"\n[W] Weight matrix: {classifier.W.shape}")
        show(str(classifier.W))
        show("(... indicates omitted entries; full values are in weights.csv)")
        show(f"\n[b] Bias row vector: {classifier.b.shape}")
        show(str(classifier.b))
        for i in range(num_test):
            show(f"\nTest #{i}: true={CLASS_NAMES[y[i]]}, predicted={CLASS_NAMES[preds[i]]}")
            show(f"{'Class':12s} {'Score':>12s} {'Probability':>12s}")
            for c, name in enumerate(CLASS_NAMES):
                show(f"{name:12s} {scores[i, c]:12.6f} {probs[i, c]:12.6f}")
            show(f"Probability sum: {probs[i].sum():.12f}")
            show(f"SVM hinge loss: {losses[i]:.6f}")
        show(f"\nAverage SVM hinge loss: {mean_loss:.6f}")

    np.savetxt(OUTPUT_DIR / "weights.csv", classifier.W, delimiter=",")
    np.savetxt(OUTPUT_DIR / "bias.csv", classifier.b, delimiter=",")

    # Choosing the examples of test result 
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for row, (mask, title) in enumerate([
        (preds == y, "Correct"), (preds != y, "Incorrect")
    ]):
        indices = np.flatnonzero(mask)[:4]
        for col, ax in enumerate(axes[row]):
            ax.axis("off")
            if col < len(indices):
                i = indices[col]
                ax.imshow(to_image(raw_images[i]), interpolation="nearest")
                ax.set_title(f"{title}: test #{i}\nTrue: {CLASS_NAMES[y[i]]}"
                             f"\nPred: {CLASS_NAMES[preds[i]]}")
            elif col == 0:
                ax.text(0.5, 0.5, f"No {title.lower()} examples available",
                        ha="center", va="center")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "classification_examples.png", dpi=150)
    plt.close(fig)

    #Printing image
    W = classifier.W
    scale = np.abs(W).max()
    display_W = 0.5 + W / (2 * scale)  # 0 = gray color
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for c, ax in enumerate(axes.flat):
        ax.imshow(to_image(display_W[:, c]), interpolation="nearest")
        ax.set_title(CLASS_NAMES[c])
        ax.axis("off")
    fig.suptitle("Random class weights: shared scale, zero = gray")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "class_weights.png", dpi=150)
    plt.close(fig)
    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
