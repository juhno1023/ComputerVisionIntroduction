import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
import seaborn as sns

def load_cifar10_batch(file): # load Data
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    X = dict[b'data']
    Y = dict[b'labels']
    return X, Y

def load_cifar10(data_dir): # load DataSet
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(data_dir, 'data_batch_%d' % (b, ))
        X, Y = load_cifar10_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_cifar10_batch(os.path.join(data_dir, 'test_batch'))
    return Xtr, Ytr, Xte, np.array(Yte)

class KNearestNeighbor:
    def __init__(self):
        pass

    def train(self, X, y):
        # Keep the training data on the GPU
        self.X_train = torch.as_tensor(X, dtype=torch.float32, device=device)
        self.y_train = y

    # Distances are computed in chunks to limit GPU memory use
    def _compute_distances(self, X, p, chunk=1000):
        X = torch.as_tensor(X, dtype=torch.float32, device=device)
        return torch.cat([
            torch.cdist(X[start:start + chunk], self.X_train, p=p)
            for start in range(0, X.shape[0], chunk)
        ])

    # L1 distance
    def compute_distances_l1(self, X):
        return self._compute_distances(X, p=1)

    # L2 distance
    def compute_distances_l2(self, X):
        return self._compute_distances(X, p=2)

    # Finding K Nearest Neighbors
    def predict(self, dists, k=1):
        nearest = torch.topk(dists, k, dim=1, largest=False).indices.cpu().numpy()
        closest_y = self.y_train[nearest]
        y_pred = np.zeros(dists.shape[0])
        for i in range(dists.shape[0]):
            y_pred[i] = Counter(closest_y[i]).most_common(1)[0][0]
        return y_pred


# Plot accuracy and its rate of change over k for each distance metric
def plot_accuracy_vs_k(k_choices, metrics, k_to_accuracies, k_to_stds):
    fig, (ax_acc, ax_rate) = plt.subplots(1, 2, figsize=(14, 5))

    for metric in metrics:
        accs = np.array([k_to_accuracies[(metric, k)] for k in k_choices])
        stds = np.array([k_to_stds[(metric, k)] for k in k_choices])

        # Rate of change (%) compared to the previous k
        rates = (accs[1:] - accs[:-1]) / accs[:-1] * 100

        ax_acc.errorbar(k_choices, accs, yerr=stds, marker='o', capsize=4, label=metric)
        ax_rate.plot(k_choices[1:], rates, marker='o', label=metric)

        print(f"Accuracy change rate by k ({metric}):")
        for prev_k, k, rate in zip(k_choices[:-1], k_choices[1:], rates):
            print(f"  k={prev_k} -> k={k}: {rate:+.2f}%")

    ax_acc.set_xlabel('k')
    ax_acc.set_ylabel('Cross-validation Accuracy')
    ax_acc.set_title('Accuracy vs k (mean +/- std over folds)')
    ax_acc.set_xticks(k_choices)
    ax_acc.grid(True, alpha=0.3)
    ax_acc.legend()

    ax_rate.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax_rate.set_xlabel('k')
    ax_rate.set_ylabel('Accuracy change from previous k (%)')
    ax_rate.set_title('Accuracy Change Rate vs k')
    ax_rate.set_xticks(k_choices[1:])
    ax_rate.grid(True, alpha=0.3)
    ax_rate.legend()

    fig.tight_layout()
    fig.savefig('knn_accuracy_vs_k.png')


# Print accuracy, precision, recall and F1-score
def print_evaluation_metrics(name, y_true, y_pred, classes):
    y_pred = y_pred.astype(int)
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )

    print(f"Evaluation metrics ({name}):")
    print(f"  Accuracy : {accuracy:.3f}")
    print(f"  Precision: {precision:.3f} (macro)")
    print(f"  Recall   : {recall:.3f} (macro)")
    print(f"  F1-score : {f1:.3f} (macro)")
    print(classification_report(
        y_true, y_pred, labels=list(range(len(classes))), target_names=classes,
        digits=3, zero_division=0
    ))


# Save overall and per-class evaluation metrics as an image
def save_evaluation_metrics_image(name, y_true, y_pred, classes, filename):
    y_pred = y_pred.astype(int)
    labels = list(range(len(classes)))
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    class_scores = np.array(precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )[:3]).T

    fig, (ax_all, ax_class) = plt.subplots(
        1, 2, figsize=(13, 6), gridspec_kw={'width_ratios': [1, 1.2]}
    )

    # Overall metrics
    names = ['Accuracy', 'Precision', 'Recall', 'F1-score']
    scores = [accuracy, precision, recall, f1]
    bars = ax_all.bar(names, scores, color='steelblue')
    ax_all.bar_label(bars, fmt='%.3f', padding=3)
    ax_all.set_ylim(0, 1)
    ax_all.set_ylabel('Score')
    ax_all.set_title('Overall Metrics (macro average)')

    # Per-class metrics
    sns.heatmap(class_scores, annot=True, fmt='.3f', cmap='Blues', vmin=0, vmax=1,
                xticklabels=['Precision', 'Recall', 'F1-score'], yticklabels=classes, ax=ax_class)
    ax_class.set_title('Per-class Metrics')

    fig.suptitle(f'Evaluation Metrics ({name})')
    fig.tight_layout()
    fig.savefig(filename)


def main():
    print("Loading CIFAR-10 dataset")
    cifar10_dir = '/content/drive/MyDrive/cifar-10-batches-py'
    try:
        X_train, y_train, X_test, y_test = load_cifar10(cifar10_dir)
    except Exception as e:
        print(f"Error loading CIFAR-10 from {cifar10_dir}: {e}")
        return

    num_test = X_test.shape[0]

    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Test labels shape: {y_test.shape}")

    # Convert to float to avoid overflow
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    classifier = KNearestNeighbor()

    # Setting HyperParameters
    k_choices = [1, 3, 5, 7, 9, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39]
    metrics = ['L1', 'L2']
    num_folds = 5

    print(f"Performing {num_folds}-fold Cross-Validation")

    # K-Cross Validation : Split training data into folds
    X_train_folds = np.array_split(X_train, num_folds)
    y_train_folds = np.array_split(y_train, num_folds)

    k_to_accuracies = {}
    k_to_stds = {}

    for metric in metrics:
        print(f"Evaluating {metric} metric via cross-validation:")
        fold_accuracies = {k: [] for k in k_choices}
        for i in range(num_folds):
            print(f"  Fold {i + 1}/{num_folds}")
            # We use fold i as validation set, and others as training set
            X_val_fold = X_train_folds[i]
            y_val_fold = y_train_folds[i]

            X_train_fold = np.concatenate([X_train_folds[j] for j in range(num_folds) if j != i])
            y_train_fold = np.concatenate([y_train_folds[j] for j in range(num_folds) if j != i])

            classifier.train(X_train_fold, y_train_fold)

            # Distances do not depend on k, so compute them once per fold
            if metric == 'L1':
                dists = classifier.compute_distances_l1(X_val_fold)
            else:
                dists = classifier.compute_distances_l2(X_val_fold)

            for k in k_choices:
                y_val_pred = classifier.predict(dists, k=k)
                num_correct = np.sum(y_val_pred == y_val_fold)
                accuracy = float(num_correct) / len(y_val_fold)
                fold_accuracies[k].append(accuracy)

        for k in k_choices:
            avg_acc = np.mean(fold_accuracies[k])
            k_to_accuracies[(metric, k)] = avg_acc
            k_to_stds[(metric, k)] = np.std(fold_accuracies[k])
            print(f"Metric: {metric}, k: {k}, Cross-validation Accuracy: {avg_acc:.3f}")

    # Accuracy and rate of change over k
    plot_accuracy_vs_k(k_choices, metrics, k_to_accuracies, k_to_stds)


    # ----------------- TEST ---------------------


    best_l1_acc = -1
    best_l1_k = -1
    best_l2_acc = -1
    best_l2_k = -1

    for metric, k in k_to_accuracies:
        acc = k_to_accuracies[(metric, k)]
        if metric == 'L1' and acc > best_l1_acc:
            best_l1_acc = acc
            best_l1_k = k
        elif metric == 'L2' and acc > best_l2_acc:
            best_l2_acc = acc
            best_l2_k = k

    print(f"Best Cross-Validation L1 Accuracy: {best_l1_acc:.3f} with k={best_l1_k}")
    print(f"Best Cross-Validation L2 Accuracy: {best_l2_acc:.3f} with k={best_l2_k}")

    # Now evaluate on the actual test set with the best Ks
    print("Evaluating the best models on the TEST set")
    classifier.train(X_train, y_train)

    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # Test best L1
    print("Computing L1 distances for TEST set")
    dists_l1 = classifier.compute_distances_l1(X_test)
    best_l1_y_pred = classifier.predict(dists_l1, k=best_l1_k)
    test_l1_acc = float(np.sum(best_l1_y_pred == y_test)) / num_test
    print(f"Test L1 Accuracy (k={best_l1_k}): {test_l1_acc:.3f}")
    print_evaluation_metrics(f"L1, k={best_l1_k}", y_test, best_l1_y_pred, classes)
    save_evaluation_metrics_image(f"L1, k={best_l1_k}", y_test, best_l1_y_pred, classes,
                                  'knn_evaluation_metrics_l1.png')
    del dists_l1

    # Test best L2
    print("Computing L2 distances for TEST set")
    dists_l2 = classifier.compute_distances_l2(X_test)
    best_l2_y_pred = classifier.predict(dists_l2, k=best_l2_k)
    test_l2_acc = float(np.sum(best_l2_y_pred == y_test)) / num_test
    print(f"Test L2 Accuracy (k={best_l2_k}): {test_l2_acc:.3f}")
    print_evaluation_metrics(f"L2, k={best_l2_k}", y_test, best_l2_y_pred, classes)
    save_evaluation_metrics_image(f"L2, k={best_l2_k}", y_test, best_l2_y_pred, classes,
                                  'knn_evaluation_metrics_l2.png')

    # Confusion matrix
    # Save L1 confusion matrix
    cm_l1 = confusion_matrix(y_test, best_l1_y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_l1, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix (L1, k={best_l1_k})')
    plt.savefig('knn_confusion_matrix_l1.png')

    # Save L2 confusion matrix
    cm_l2 = confusion_matrix(y_test, best_l2_y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_l2, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix (L2, k={best_l2_k})')
    plt.savefig('knn_confusion_matrix_l2.png')

if __name__ == '__main__':
    main()