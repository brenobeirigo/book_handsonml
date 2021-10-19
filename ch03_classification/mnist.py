"""
for

abb
   a

*b

c
a:next
b:b, next


Input Actual Predicted  Classification
8     N      N          TN
3     N      N          TN
9     N      N          TN
7     N      N          TN
2     N      N          TN
6     N      Y          FP
5     Y      N          FN
5     Y      N          FN
5     Y      Y          TP
5     Y      Y          TP
5     Y      Y          TP


"""
# %%
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from pprint import pprint
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt
import numpy as np
from inspect import CORO_CLOSED
from scipy.sparse.construct import rand
from sklearn.datasets import fetch_openml

mnist = fetch_openml("mnist_784", version=1)
mnist.keys()
# %%
mnist
# %%
print(mnist.DESCR)

# %%
X, y = np.array(mnist["data"]), np.array(mnist["target"])
X.shape, y.shape
# %%
type(X), type(y), type(X[0]), type(y[0])

# %%[markdown]
# The target array `y` stores strings.
# Since most ML algorithms expect numbers, cast it to integer:
# %%
y = y.astype(np.uint8)
y
# %% [markdown]
# ### Plotting the data

# Let us print the first digit from `X`. First, we reshape the grayscale pixel array to a 28x28 grid:
# %%
digit = X[0].reshape(28, 28)
digit

# %% [markdown]
# Next, we use `matplotlib` to generate the image. It represents a `5`:
# %%
plt.imshow(digit, cmap="binary")
plt.title(f"Target = {y[0]}")
plt.axis("off")
plt.show()

# %% [markdown]
# Let us see how the first 100 pictures look like:
# %%


def plot_img_grid(X, y, predicted=None, rows=10, cols=10, figsize=(10, 10)):
    fig, ax = plt.subplots(rows, cols, figsize=figsize)
    for i in range(rows):
        for j in range(cols):
            digit = X[i + 10 * j].reshape(28, 28)
            ax[i][j].imshow(digit, cmap="binary")

            fontdict = {"fontsize": 8}
            title = y[i + 10 * j]

            # Change title color if the prediction is wrong
            if predicted is not None:
                hit_target = y[i + 10 * j] == predicted[i + 10 * j]
                font_color = "red" if not hit_target else "black"
                fontdict["color"] = font_color
                title = predicted[i + 10 * j]

            ax[i][j].set_title(title, fontdict=fontdict)
            ax[i][j].axis("off")

    st = fig.suptitle("MNIST - Hand-written numbers", fontsize=24)
    fig.subplots_adjust(top=1.2)  # Space between rows
    st.set_y(1.25)  # Title position


plot_img_grid(X, y)

# %%
# No need to use stratified sampling: sets are already shuffled
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# %% [markdown]
# Let us start by creating a simpler binary classifier.
# We aim at identifying whether an image represents a 5 (`True`) or
# anything else (`False`):
# %%
y_train_5 = y_train == 5
y_test_5 = y_test == 5
plot_img_grid(X_train, y_train_5)

# %% [markdown]
# ### Training a "is 5" classifier
# Let us fit a *stochastic gradient descent (SGD)* classifier:
# %%
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)

# %% [markdown]
# The classifier seems very accurate: only 2 out 100 predictions were wrong:
# %%
predictions = sgd_clf.predict(X_test)
plot_img_grid(X_test, y_test_5, predicted=predictions)

# %%[markdown]
# ### Using cross validation

# The `cross_val_prediction` function computes for each fold
# the predictions and concatenates them.
#
# The data is split in `K` folds, and for `i = 1, ..., K` iterations it:
# - Takes all folds (except the i'th) as the training data;
# - Fits the model with the folds;
# - Predicts labels for the i'th part (test data);
# - Merges all partially predicted labels and returns them as a whole.
#
# Hence, the predictions for each fold are based on a model trained on the other folds.

# %%

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
plot_img_grid(X_train, y_train_5, predicted=y_train_pred)

# %% [markdown]
# ### Confusion matrix
#
# The confusion matrix will present the number o correctly and wrongly classified instances.
# Each row represents an *actual class* and each column represents a *predicted class*:
#
#
# |           | *predicted* N | *predicted* P |
# |:---------:|:-------------:|:-------------:|
# |*actual* N | <span style="color:green">True negatives </span> | <span style="color:red"> False positives </span>|
# |*actual* P | <span style="color:red">  False positives </span>| <span style="color:green"> True positives  </span>|
#
# A perfect classifier would have only <span style="color:green">true positives</span> and <span style="color:green">true negatives</span> (i.e., nonzero values only on its main diagonal).
# #### Example of classifications tasks
#
# 1. Classify safe kid's content
#
# |           | *predicted* N | *predicted* P |
# |:---------:|:-------------:|:-------------:|
# |*actual* N | <span style="color:green">**TN**</span> (Correctly classified UNSAFE content) | <span style="color:red">**FP**</span>  (UNSAFE content misclassified as SAFE) |
# |*actual* P | <span style="color:red">**FN**</span> (SAFE content misclassified as UNSAFE) | <span style="color:green">**TP**</span>  (Correctly classified SAFE content) |
#
# 2. Classify shoplifters in surveillance images
#
# |           | *predicted* N | *predicted* P |
# |:---------:|:-------------:|:-------------:|
# |*actual* N | <span style="color:green">**TN**</span> (correctly classified HONEST customer) | <span style="color:red">**FP**</span> (HONEST customer misclassified as SHOPLIFTER) |
# |*actual* P | <span style="color:red">**FN**</span> (SHOPLIFTER misclassified as HONEST customer) | <span style="color:green">**TP**</span> (correclty classified SHOPLIFTER) |

# %%
# https://towardsdatascience.com/precision-recall-and-predicting-cervical-cancer-with-machine-learning-367221e70538

# %% [markdown]
# Using target and predicted classes we can generate the confusion matrix:
# %%
confusion_matrix(y_train_5, y_train_pred)
# %% [markdown]
# The perfect classifier would have target equals predicted, with confusion matrix:
# %%
confusion_matrix(y_train_5, y_train_pred)
# %% [markdown]
# The implementation of the confusion matrix is as follows:
# %%


def get_confusion_matrix(actual, predicted):
    """Count the number of:
    - True negatives
    - False positives
    - False negatives
    - True positives
    """
    tp = (actual == True) & (predicted == actual)
    tn = (actual == False) & (predicted == actual)
    fn = (actual == True) & (predicted != actual)
    fp = (actual == False) & (predicted != actual)

    tp_count = sum(tp)
    tn_count = sum(tn)
    fn_count = sum(fn)
    fp_count = sum(fp)

    confusion_matrix = np.array([[tn_count, fp_count], [fn_count, tp_count]])

    return confusion_matrix


# %% [markdown]
# ### Precision, recall, F_1 score, and the problem of accuracy

# The function below shows how accuracy, precision, recall, and the $F_1$ score (the harmonic mean) are calculated:

# %%


def get_scores(confusion_matrix):
    tn_count, fp_count, fn_count, tp_count = confusion_matrix.ravel()

    precision = tp_count / (tp_count + fp_count)
    recall = tp_count / (tp_count + fn_count)
    # f1_score = 2/(1/precision + 1/recall)
    f1_score = 2 * (precision * recall / (precision + recall))  # harmonic mean
    accuracy = (tp_count + tn_count) / np.sum(confusion_matrix)

    return {
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "f1_score": f1_score,
    }


# %%
print("# Confusion matrix:")
conf_matrix = get_confusion_matrix(y_train_5, y_train_pred)
score_dict = get_scores(get_confusion_matrix(y_train_5, y_train_pred))

print("# Scores:")
pprint(score_dict)

# %% [markdown]
# The scores are consistent with the those output from sklearn:
# %%

assert score_dict["precision"] == precision_score(y_train_5, y_train_pred)
assert score_dict["recall"] == recall_score(y_train_5, y_train_pred)
assert score_dict["f1_score"] == f1_score(y_train_5, y_train_pred)


# %% [markdown]
# The classifier seems to be very accurate: 95% of the data was correctly classified.
# But when it claims an image represents a 5, it is correct only `prediction_score` of the time.
# Moreover, it only detects (recalls) `recall_score` of the 5s.

# How many 5s are actually present in the the dataset?
# %%
sum(y_train_5) / len(y_train_5)
# %% [markdown]
# Less than 10% of the data comprises 5s.
# The dataset is *skewed*, rendering accuracy a misleading score.
#
# In this case, precision is higher than recall:
# - 16% of predicted 5s were misclassified (precision = 84%)
# - The classifier missed 35% of the 5s (recall = 65%)
#
# Let's look into further how this classifier performs with only 5's (recall):
# %%
only_5 = y_train_5 == True
plot_img_grid(
    X_train[only_5, :], y_train_5[only_5], predicted=y_train_pred[only_5]
)

# %% [markdown]
# High recall is better for:
#  - Classifying shoplifters (FPs are honest customers, FNs are uncought shoplifters)
#  - Classifying cancer (FPs are misclassified cancer cases, just do another exam)
#
# High precision is better for:
#  - Classifying safe kid content (FN is OK, content will be wrongly excluded, but FP is unacceptable)

# %% [markdown]
# Instead of returning the predictions, we can get the scores computed by the SGD classifier based on a decision function using:
# %%
y_scores = cross_val_predict(
    sgd_clf, X_train, y_train_5, cv=3, method="decision_function"
)
# %% [markdown]
# Assuming a decision threshold of 0, we can determine the rate of positive predictions (i.e., TPs and FPs):
# %%
threshold = 0
positive_pred_count = sum(y_scores > threshold)
print(positive_pred_count / len(X_train))
# %% [markdown]
# This number can be confirmed by the values presented in the confusion matrix (2nd column with TPs and FPs):
# %%
conf_matrix = confusion_matrix(y_train_5, y_train_pred)

print(conf_matrix)
positive_pred_count = sum(conf_matrix[:, 1])
print(positive_pred_count / np.sum(conf_matrix))

# %% [markdown]
# Precision recall curve
# %%
# precision : ndarray of shape (n_thresholds + 1,)
#     Precision values such that element i is the precision of predictions with score >= thresholds[i] and the last element is 1.

# recall : ndarray of shape (n_thresholds + 1,)
#     Decreasing recall values such that element i is the recall of predictions with score >= thresholds[i] and the last element is 0.

# thresholds : ndarray of shape (n_thresholds,)
#     Increasing thresholds on the decision function used to compute precision and recall. n_thresholds <= len(np.unique(probas_pred)).

# %%
len(set(y_scores)) == len(y_scores)
# %%
plt.plot()
# %%
sorted(y_scores) == y_scores
# %%
# TODO What are the scores really???
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
print(precisions, recalls, thresholds)

# %%
# precisions = []
# recalls = []
# for t in sorted(y_scores):
#     thresholds_t = y_scores >= t
#     precisions.append(precision_score(y_train_5, thresholds_t))
#     recalls.append(recall_score(y_train_5, thresholds_t))

# %%
plt.scatter(thresholds, np.arange(len(thresholds)), color="r")

# %%
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].plot(sorted(y_scores), label="Scores")
ax[1].plot(sorted(thresholds), label="Thresholds")
# TODO Why irrelevant? The difference is not empty...
print(set(y_scores).difference(set(thresholds)))
#%%
def plot_precision_recall_vs_threshold(
    precisions, recalls, thresholds, threshold_idx
):  
    
    # The last precision and recall values are 1. and 0. respectively and do not have a corresponding threshold. This ensures that the graph starts on the y axis
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend()
    plt.xlim(thresholds[0], thresholds[-1])
    plt.ylim(0, 1)
    plt.grid()
    plt.vlines(
        [thresholds[threshold_idx], thresholds[threshold_idx]],
        [0, 0],
        [precisions[threshold_idx], recalls[threshold_idx]],
        linestyle="dashed",
        colors=["r", "r"],
    )

    plt.hlines(
        [precisions[threshold_idx], recalls[threshold_idx]],
        [thresholds[0], thresholds[0]],
        [thresholds[threshold_idx], thresholds[threshold_idx]],
        linestyle="dashed",
        colors=["r", "r"],
    )

    plt.scatter(
        [thresholds[threshold_idx], thresholds[threshold_idx]],
        [precisions[threshold_idx], recalls[threshold_idx]],
        color="red",
    )


# %% [markdown]
# What is the lowest threshold that gives us at least 90% precision?
# %%
# [False, False, ..., True, True, True]
min_precision = precisions >= 0.9
# %%

# np.argmax(precisions)

# TODO continue reading https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#sphx-glr-auto-examples-model-selection-plot-precision-recall-py
plot_precision_recall_vs_threshold(
    precisions, recalls, thresholds, np.argmax(min_precision)
)

# %%
def plot_precision_recall(precisions, recalls, min_precision_threshold):
    min_precisions = precisions >= min_precision_threshold
    threshold_idx = np.argmax(min_precisions)
    plt.plot(recalls, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.scatter([recalls[threshold_idx]], [precisions[threshold_idx]], color="r")
    plt.axhline(precisions[threshold_idx], 0, recalls[threshold_idx], color="r", linestyle="dashed")
    plt.axvline(recalls[threshold_idx], 0, precisions[threshold_idx], color="r", linestyle="dashed")
    plt.text(
        recalls[threshold_idx],
        precisions[threshold_idx],
        f"(precision = {precisions[threshold_idx]:.2f}, "
        f"recall = {recalls[threshold_idx]:.2f})")
    plt.xlim(0,1)
    plt.ylim(0,1)
# %%
plot_precision_recall(precisions, recalls, 0.9)

# 52% of the 5s are not classified as so
# 10% of the instances classified as 5 are not

# %%[markdown]
### Making predictions with scores
# %%
idx_90 = np.argmax(precisions > 0.9)
y_train_pred_5_90 = y_scores > thresholds[idx_90]

# %%
# %%
from sklearn.metrics import roc_auc_score, precision_score, recall_score
def score_stats(labels, scores):
    print("ROC AUC:", roc_auc_score(labels, scores))
    print("Precision:", precision_score(labels, scores))
    print("Recall:", recall_score(labels, scores))

score_stats(y_train_5, y_train_pred_5_90)
# %% [markdown]
# ### The ROC Curve
#%%

from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
# TODO finish ROC and skip to LSTM!!!!
def plot_roc_curve(fpr, tpr, recall):
    plt.plot(fpr, tpr, "b", label="ROC")
    plt.plot([0,1],[0,1], "r--")
    idx = np.argmax(tpr > recall)
    plt.scatter([fpr[idx]], [tpr[idx]], color="r")
    plt.text(0.5,0.5, "ROC \n random classifier")
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.annotate(
        "Low FPR because the\npositive class (5s) is rare",
        xy=(fpr[idx], tpr[idx]),
        xytext=(0.1, 0.6),
        arrowprops=dict(color="red", arrowstyle="->"),
        textcoords="axes fraction"
        )
    plt.ylabel("TPR (Recall)")
    plt.xlabel("FPR")


plot_roc_curve(fpr, tpr, recall=recall_score(y_train_5, y_train_pred_5_90))
# %% [markdown]
# One way to compare classifiers is to measure the area under the curve (AUC).
# A perfect classifier will have a ROC AUC equal to 1, whereas a purely random classifier will have a ROC AUC equal to 0.5.
# %%

# %%
from sklearn.metrics import roc_auc_score
roc_auc_score(y_train_5, y_scores)

# %% [markdown]
# ## Comparing ROC AUC for 2 classifiers 
# %%
from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method="predict_proba")

# %%
import pandas as pd
pd.DataFrame(y_probas_forest, columns=["Prob. neg.", "Prob. pos."])

# %%
y_scores_forest = y_probas_forest[:, 1]
fpr_forest, tpr_forest, threshold_forest = roc_curve(
    y_train_5,
    y_scores_forest)

# %%
# TODO understand why score stats fails
plot_roc_curve(fpr, tpr, recall=recall_score(y_train_5, y_train_pred_5_90))
plt.plot(fpr_forest, tpr_forest, "b:", label="SGD")
plt.legend(loc="lower right")


score_stats(y_train_5, y_scores_forest)

"""
TODO multiclass

Whereas binary classifiers distinguish between two classes, multiclass classifiers (also called multinomial classifiers) can distinguish between more than two classes.
Some algorithms (such as SGD classifiers, Random Forest classifiers, and naive Bayes classifiers) are capable of handling multiple classes natively.

Others (such as Logistic Regression or Support Vector Machine classifiers) are strictly binary classifiers.
To perform multiclass classification with multiple binary classifiers you can:
1) One-versus-the-rest (OvR) strategy (also called one-versus-all)
 - Train 10 binary classifiers, one for each digit (a 0-detector, a 1-detector, a 2-detector, and so on).
 - Then when you want to classify an image, you get the decision score from each classifier for that image and you select the class whose classifier outputs the highest score.

2) One-versus-one (OvO) strategy
- Train a binary classifier for every pair of digits: one to distinguish 0s and 1s, another to distinguish 0s and 2s, another for 1s and 2s, and so on. If there are N classes, you need to train N × (N – 1) / 2 classifiers. For the MNIST problem, this means training 45 binary classifiers! - When you want to classify an image, you have to run the image through all 45 classifiers and see which class wins the most duels.
 - The main advantage of OvO is that each classifier only needs to be trained on the part of the training set for the two classes that it must distinguish.


Some algorithms (such as Support Vector Machine classifiers) scale poorly with the size of the training set. For these algorithms OvO is preferred because it is faster to train many classifiers on small training sets than to train few classifiers on large training sets. For most binary classification algorithms, however, OvR is preferred.
"""

# %% [markdown]
# ### Implemeting cross-validation
# %%
skfolds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

for train_index, test_index in skfolds.split(X, y):
    clone_clf = clone(sgd_clf)
    print(train_index, test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clone_clf.fit(X_train, y_train)

    # Cross validation score from the scratch
    # Counting correct predictions
    y_predictions = sgd_clf.predict(X_test)
    n_correct = sum(y_test == y_predictions)
    print(n_correct / len(y_predictions))


# %%

# %%
