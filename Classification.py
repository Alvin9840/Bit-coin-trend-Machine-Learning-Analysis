from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import (train_test_split, GridSearchCV, StratifiedKFold, cross_val_score)
from sklearn.metrics import (classification_report, accuracy_score, confusion_matrix,
                             precision_score, recall_score, f1_score, roc_curve, auc, roc_auc_score)
from sklearn.preprocessing import StandardScaler, label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import warnings
import os
import glob

warnings.filterwarnings("ignore")

RANDOM = 5805
np.random.seed(RANDOM)
FEATURES = [
    'volatility',
    'volume',
    'number_of_trades',
    'taker_buy_quote_asset_volume'
]


def plot_confusion_matrix(y_true, y_pred, classes, title):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(f"Confusion Matrix - {title}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    return cm


def calculate_specificity(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    specificity = []
    for i in range(len(classes)):
        tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
        fp = np.sum(cm[:, i]) - cm[i, i]
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificity.append(spec)
    return np.mean(specificity)


def plot_roc_curve(y_true, y_prob, classes, title):
    y_bin = label_binarize(y_true, classes=classes)
    n_classes = len(classes)

    plt.figure(figsize=(8, 6))

    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_bin, y_prob[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})')
    else:
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{classes[i]} (AUC = {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {title}')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def evaluate_classifier(model, X_train, X_test, y_train, y_test, classes, name, cv=3):
    print("Model fitting...")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Predictions completed")

    # For large datasets, use subset for cross-validation to save time
    if len(X_train) > 50000 and name in ["SVM", "Random Forest", "Bagging", "Stacking", "Boosting",
                                         "Neural Network (MLP)"]:
        sample_size = 50000
        rng = np.random.RandomState(RANDOM)
        idx = rng.choice(len(X_train), sample_size, replace=False)
        X_cv = X_train[idx]
        y_cv = y_train.iloc[idx] if hasattr(y_train, 'iloc') else y_train[idx]
    else:
        X_cv = X_train
        y_cv = y_train

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM)
    cv_scores = cross_val_score(model, X_cv, y_cv, cv=skf, scoring='accuracy')

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    specificity = calculate_specificity(y_test, y_pred, classes)

    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=[str(c) for c in classes]))
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall/Sensitivity: {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Stratified K-fold cross validation: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    cm = plot_confusion_matrix(y_test, y_pred, classes, name)
    # plt.savefig(f"CM_{name}.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Calculate and plot ROC curve if predict_proba is available
    if hasattr(model, 'predict_proba') and getattr(model, 'probability', True):
        try:
            y_prob = model.predict_proba(X_test)
            auc_score = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')
            print(f"AUC Score: {auc_score:.4f}")
            plot_roc_curve(y_test, y_prob, classes, name)
            # plt.savefig(f"ROC_{name}.png", dpi=300, bbox_inches='tight')
            plt.show()
        except Exception as e:
            auc_score = None
            print(f"AUC: probability estimation failed: {e}")
    else:
        auc_score = None
        print("AUC: probability=False or predict_proba not available")

    return {
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'Specificity': specificity,
        'F1-Score': f1,
        'AUC': auc_score,
        'CV_Accuracy': cv_scores.mean(),
        'CV_Std': cv_scores.std()
    }


def decision_tree_analysis(X_train, X_test, y_train, y_test, classes, feature_names):
    """Decision tree analysis with pre-pruning and post-pruning"""
    sample_size = min(50000, len(X_train))
    rng = np.random.RandomState(RANDOM)
    idx = rng.choice(len(X_train), sample_size, replace=False)
    X_sample = X_train[idx]
    y_sample = y_train.iloc[idx] if hasattr(y_train, 'iloc') else y_train[idx]

    print("\n--- Pre-Pruning (Grid Search for Hyperparameters) ---")

    param_grid_prepruning = {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 10, 50],
        'min_samples_leaf': [1, 10, 50],
        'max_features': ['sqrt', 'log2', None]
    }

    grid_search_prepruned = GridSearchCV(
        DecisionTreeClassifier(random_state=RANDOM),
        param_grid_prepruning,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    grid_search_prepruned.fit(X_sample, y_sample)

    print(f"Best Parameters (Pre-pruned): {grid_search_prepruned.best_params_}")
    print(f"Best CV Score (Pre-pruned): {grid_search_prepruned.best_score_:.4f}")

    clf_prepruned = DecisionTreeClassifier(**grid_search_prepruned.best_params_, random_state=RANDOM)
    clf_prepruned.fit(X_train, y_train)

    train_acc_prepruned = clf_prepruned.score(X_train, y_train)
    test_acc_prepruned = clf_prepruned.score(X_test, y_test)
    print(f"Train Accuracy (Pre-pruned): {train_acc_prepruned:.4f}")
    print(f"Test Accuracy (Pre-pruned): {test_acc_prepruned:.4f}")
    print(f"Tree Depth: {clf_prepruned.get_depth()}")
    print(f"Number of Leaves: {clf_prepruned.get_n_leaves()}")

    plt.figure(figsize=(20, 10))
    plot_tree(clf_prepruned,
              feature_names=feature_names,
              class_names=classes,
              filled=True,
              rounded=True,
              max_depth=4)
    plt.title("Pre-Pruned Decision Tree")
    plt.tight_layout()
    # plt.savefig("DT_PrePruned.png", dpi=300, bbox_inches='tight')
    plt.show()

    y_pred_prepruned = clf_prepruned.predict(X_test)
    y_prob_prepruned = clf_prepruned.predict_proba(X_test)

    print("\n--- Post-Pruning (Cost Complexity Pruning / ccp_alpha) ---")

    path = clf_prepruned.cost_complexity_pruning_path(X_sample, y_sample)
    ccp_alphas = path.ccp_alphas
    impurities = path.impurities

    plt.figure(figsize=(10, 5))
    plt.plot(ccp_alphas[:-1], impurities[:-1], marker='o', markersize=3)
    plt.xlabel("Effective Alpha (ccp_alpha)")
    plt.ylabel("Total Impurity of Leaves")
    plt.title("Cost Complexity Pruning Path")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # plt.savefig("DT_CostComplexity_Path.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Reduce number of alpha values for evaluation if too many
    if len(ccp_alphas) > 30:
        indices = np.linspace(0, len(ccp_alphas) - 1, 10, dtype=int)
        ccp_alphas_eval = ccp_alphas[indices]
    else:
        ccp_alphas_eval = ccp_alphas

    train_acc_postpruned = []
    test_acc_postpruned = []

    for alpha in ccp_alphas_eval:
        clf_temp = DecisionTreeClassifier(random_state=RANDOM, ccp_alpha=alpha)
        clf_temp.fit(X_sample, y_sample)
        train_acc_postpruned.append(clf_temp.score(X_sample, y_sample))
        test_acc_postpruned.append(clf_temp.score(X_test, y_test))

    plt.figure(figsize=(10, 6))
    plt.plot(ccp_alphas_eval, train_acc_postpruned, marker='o', markersize=3, label='Train Accuracy')
    plt.plot(ccp_alphas_eval, test_acc_postpruned, marker='s', markersize=3, label='Test Accuracy')
    plt.xlabel("ccp_alpha")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. ccp_alpha for Post-Pruning")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # plt.savefig("DT_PostPruning_Accuracy.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Select best alpha based on test accuracy
    best_alpha_idx = np.argmax(test_acc_postpruned)
    best_alpha = ccp_alphas_eval[best_alpha_idx]
    print(f"Best ccp_alpha (Post-pruned): {best_alpha:.6f}")

    clf_postpruned = DecisionTreeClassifier(random_state=RANDOM, ccp_alpha=best_alpha)
    clf_postpruned.fit(X_train, y_train)

    train_acc_postpruned_final = clf_postpruned.score(X_train, y_train)
    test_acc_postpruned_final = clf_postpruned.score(X_test, y_test)
    print(f"Train Accuracy (Post-pruned): {train_acc_postpruned_final:.4f}")
    print(f"Test Accuracy (Post-pruned): {test_acc_postpruned_final:.4f}")
    print(f"Tree Depth: {clf_postpruned.get_depth()}")
    print(f"Number of Leaves: {clf_postpruned.get_n_leaves()}")

    plt.figure(figsize=(20, 10))
    plot_tree(clf_postpruned,
              feature_names=feature_names,
              class_names=classes,
              filled=True,
              rounded=True,
              max_depth=4)
    plt.title(f"Post-Pruned Decision Tree (ccp_alpha={best_alpha:.6f})")
    plt.tight_layout()
    # plt.savefig("DT_PostPruned.png", dpi=300, bbox_inches='tight')
    plt.show()

    y_pred_postpruned = clf_postpruned.predict(X_test)
    y_prob_postpruned = clf_postpruned.predict_proba(X_test)

    print("\n--- Decision Tree Comparison ---")
    print("-" * 70)
    print(f"{'Model':<25}{'Train Acc':<12}{'Test Acc':<12}{'Depth':<10}{'Leaves':<10}")
    print("-" * 70)
    print(
        f"{'Pre-Pruned (GridSearch)':<25}{train_acc_prepruned:<12.4f}{test_acc_prepruned:<12.4f}{clf_prepruned.get_depth():<10}{clf_prepruned.get_n_leaves():<10}")
    print(
        f"{'Post-Pruned (ccp_alpha)':<25}{train_acc_postpruned_final:<12.4f}{test_acc_postpruned_final:<12.4f}{clf_postpruned.get_depth():<10}{clf_postpruned.get_n_leaves():<10}")
    print("-" * 70)

    dt_models = {
        'Pre-Pruned': (clf_prepruned, test_acc_prepruned),
        'Post-Pruned': (clf_postpruned, test_acc_postpruned_final)
    }
    best_dt_name = max(dt_models, key=lambda x: dt_models[x][1])
    best_dt_model = dt_models[best_dt_name][0]
    print(f"\nBest Decision Tree: {best_dt_name} (Test Acc: {dt_models[best_dt_name][1]:.4f})")

    return {
        'prepruned': clf_prepruned,
        'postpruned': clf_postpruned,
        'best': best_dt_model,
        'best_name': best_dt_name,
        'y_pred_prepruned': y_pred_prepruned,
        'y_prob_prepruned': y_prob_prepruned,
        'y_pred_postpruned': y_pred_postpruned,
        'y_prob_postpruned': y_prob_postpruned
    }


def classification(agg_df, classifier_name=None):
    """Run classification analysis with various classifiers"""
    print("\n===== Phase III: Classification Analysis =====\n")

    X = agg_df[FEATURES].dropna()
    y = agg_df.loc[X.index, 'price_trend']
    classes = ['down', 'flat', 'up']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, shuffle=False, random_state=RANDOM)

    print(f"Training Set Size: {len(X_train)}")
    print(f"Test Set Size: {len(X_test)}")

    results = []

    # Linear Discriminant Analysis
    if classifier_name in [None, "LDA"]:
        print("\n===== 1. Linear Discriminant Analysis (LDA) =====")
        lda_param_grid = {
            'solver': ['svd', 'lsqr', 'eigen']
        }
        lda_grid = GridSearchCV(LinearDiscriminantAnalysis(), lda_param_grid, cv=3, n_jobs=-1)
        lda_grid.fit(X_train, y_train)
        print(f"Best LDA Parameters: {lda_grid.best_params_}")
        result = evaluate_classifier(lda_grid.best_estimator_, X_train, X_test, y_train, y_test, classes, "LDA")
        results.append(result)

    # Decision Tree
    if classifier_name in [None, "DecisionTree"]:
        print("\n===== 2. Decision Tree =====")
        dt_results = decision_tree_analysis(X_train, X_test, y_train, y_test, classes, FEATURES)
        result_prepruned = evaluate_classifier(dt_results['prepruned'], X_train, X_test, y_train, y_test, classes,
                                               "DT (Pre-Pruned)")
        results.append(result_prepruned)
        result_postpruned = evaluate_classifier(dt_results['postpruned'], X_train, X_test, y_train, y_test, classes,
                                                "DT (Post-Pruned)")
        results.append(result_postpruned)

    # Logistic Regression
    if classifier_name in [None, "LogisticRegression"]:
        print("\n===== 3. Logistic Regression =====")
        lr_param_grid = {
            'C': [0.1, 1, 10],
            'penalty': ['l2'],
            'max_iter': [1000]
        }
        lr_grid = GridSearchCV(LogisticRegression(random_state=RANDOM), lr_param_grid, cv=3, n_jobs=-1)
        lr_grid.fit(X_train, y_train)
        print(f"Best LR Parameters: {lr_grid.best_params_}")
        result = evaluate_classifier(lr_grid.best_estimator_, X_train, X_test, y_train, y_test, classes,
                                     "Logistic Regression")
        results.append(result)

    # K-Nearest Neighbors
    if classifier_name in [None, "KNN"]:
        print("\n===== 4. K-Nearest Neighbors (KNN) =====")
        knn_param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance']
        }
        knn_grid = GridSearchCV(KNeighborsClassifier(), knn_param_grid, cv=3, n_jobs=-1)
        knn_grid.fit(X_train, y_train)
        print(f"Best KNN Parameters: {knn_grid.best_params_}")
        result = evaluate_classifier(knn_grid.best_estimator_, X_train, X_test, y_train, y_test, classes, "KNN")
        results.append(result)

    # Support Vector Machine
    if classifier_name in [None, "SVM"]:
        print("\n===== 5. Support Vector Machine (SVM) =====")
        svm_sample_size = 10000
        rng_svm = np.random.RandomState(RANDOM)
        idx_svm = rng_svm.choice(len(X_train), svm_sample_size, replace=False)
        X_svm_small = X_train[idx_svm]
        y_svm_small = y_train.iloc[idx_svm] if hasattr(y_train, 'iloc') else y_train[idx_svm]

        final_train_size = 20000
        idx_final = rng_svm.choice(len(X_train), final_train_size, replace=False)
        X_train_final = X_train[idx_final]
        y_train_final = y_train.iloc[idx_final] if hasattr(y_train, 'iloc') else y_train[idx_final]

        # SVM with Linear Kernel
        print("\n--- SVM with Linear Kernel ---")
        svm_linear_param = {'C': [0.1, 1, 10]}
        svm_linear_grid = GridSearchCV(SVC(kernel='linear', probability=False, random_state=RANDOM), svm_linear_param,
                                       cv=3, n_jobs=-1, verbose=1)
        svm_linear_grid.fit(X_svm_small, y_svm_small)
        print(f"Best Linear Parameters: {svm_linear_grid.best_params_}")
        final_linear_model = SVC(kernel='linear', probability=True, random_state=RANDOM, **svm_linear_grid.best_params_)
        result_linear = evaluate_classifier(final_linear_model, X_train_final, X_test, y_train_final, y_test, classes,
                                            "SVM")
        results.append(result_linear)

        # SVM with Polynomial Kernel
        print("\n--- SVM with Polynomial Kernel ---")
        svm_poly_param = {'C': [1, 10], 'degree': [2, 3]}
        svm_poly_grid = GridSearchCV(SVC(kernel='poly', probability=False, random_state=RANDOM), svm_poly_param, cv=3,
                                     n_jobs=-1, verbose=1)
        svm_poly_grid.fit(X_svm_small, y_svm_small)
        print(f"Best Poly Parameters: {svm_poly_grid.best_params_}")
        final_poly_model = SVC(kernel='poly', probability=True, random_state=RANDOM, **svm_poly_grid.best_params_)
        result_poly = evaluate_classifier(final_poly_model, X_train_final, X_test, y_train_final, y_test, classes,
                                          "SVM (Polynomial)")
        results.append(result_poly)

        # SVM with RBF Kernel
        print("\n--- SVM with RBF Kernel ---")
        svm_rbf_param = {'C': [1, 10], 'gamma': [0.01, 0.1]}
        svm_rbf_grid = GridSearchCV(SVC(kernel='rbf', probability=False, random_state=RANDOM), svm_rbf_param, cv=3,
                                    n_jobs=-1, verbose=1)
        svm_rbf_grid.fit(X_svm_small, y_svm_small)
        print(f"Best RBF Parameters: {svm_rbf_grid.best_params_}")
        final_rbf_model = SVC(kernel='rbf', probability=True, random_state=RANDOM, **svm_rbf_grid.best_params_)
        result_rbf = evaluate_classifier(final_rbf_model, X_train_final, X_test, y_train_final, y_test, classes,
                                         "SVM (RBF)")
        results.append(result_rbf)

    # Naive Bayes
    if classifier_name in [None, "NaiveBayes"]:
        print("\n===== 6. Naive Bayes =====")
        nb_param_grid = {'var_smoothing': [1e-9, 1e-8, 1e-7]}
        nb_grid = GridSearchCV(GaussianNB(), nb_param_grid, cv=3, n_jobs=-1)
        nb_grid.fit(X_train, y_train)
        print(f"Best NB Parameters: {nb_grid.best_params_}")
        result = evaluate_classifier(nb_grid.best_estimator_, X_train, X_test, y_train, y_test, classes, "Naive Bayes")
        results.append(result)

    # Random Forest
    if classifier_name in [None, "RandomForest", "Ensemble"]:
        print("\n===== 7. Random Forest =====")
        rf_sample_size = 50000
        rng_rf = np.random.RandomState(RANDOM)
        idx_rf = rng_rf.choice(len(X_train), rf_sample_size, replace=False)
        X_rf_small = X_train[idx_rf]
        y_rf_small = y_train.iloc[idx_rf] if hasattr(y_train, 'iloc') else y_train[idx_rf]

        rf_param_grid = {'n_estimators': [50, 100], 'max_depth': [10, None]}
        rf_grid = GridSearchCV(RandomForestClassifier(random_state=RANDOM, n_jobs=-1), rf_param_grid, cv=3, n_jobs=-1)
        rf_grid.fit(X_rf_small, y_rf_small)
        print(f"Best RF Parameters: {rf_grid.best_params_}")
        final_rf = RandomForestClassifier(random_state=RANDOM, n_jobs=-1, **rf_grid.best_params_)
        result_rf = evaluate_classifier(final_rf, X_train, X_test, y_train, y_test, classes, "Random Forest")
        results.append(result_rf)

    # Ensemble Methods
    if classifier_name in [None, "Ensemble"]:
        # Bagging
        print("\n--- Bagging Classifier ---")
        bag_sample_size = 50000
        rng_bag = np.random.RandomState(RANDOM)
        idx_bag = rng_bag.choice(len(X_train), bag_sample_size, replace=False)
        X_bag_small = X_train[idx_bag]
        y_bag_small = y_train.iloc[idx_bag] if hasattr(y_train, 'iloc') else y_train[idx_bag]

        bagging_param_grid = {'n_estimators': [30, 50], 'max_samples': [0.8, 1.0]}
        bagging_grid = GridSearchCV(
            BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=10, random_state=RANDOM), random_state=RANDOM,
                              n_jobs=-1), bagging_param_grid, cv=3, n_jobs=-1)
        bagging_grid.fit(X_bag_small, y_bag_small)
        print(f"Best Bagging Parameters: {bagging_grid.best_params_}")
        final_bagging = BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=10, random_state=RANDOM),
                                          random_state=RANDOM, n_jobs=-1, **bagging_grid.best_params_)
        result_bag = evaluate_classifier(final_bagging, X_train, X_test, y_train, y_test, classes, "Bagging")
        results.append(result_bag)

        # Stacking
        print("\n--- Stacking Classifier ---")
        estimators = [('lr', LogisticRegression(max_iter=1000, random_state=RANDOM)), ('nb', GaussianNB())]
        stacking = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=1000))
        result_stack = evaluate_classifier(stacking, X_train, X_test, y_train, y_test, classes, "Stacking")
        results.append(result_stack)

        # AdaBoost
        print("\n--- Boosting (AdaBoost) ---")
        boost_sample_size = 50000
        rng_boost = np.random.RandomState(RANDOM)
        idx_boost = rng_boost.choice(len(X_train), boost_sample_size, replace=False)
        X_boost_small = X_train[idx_boost]
        y_boost_small = y_train.iloc[idx_boost] if hasattr(y_train, 'iloc') else y_train[idx_boost]

        boosting_param_grid = {'n_estimators': [30, 50], 'learning_rate': [0.5, 1.0]}
        boosting_grid = GridSearchCV(
            AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=3, random_state=RANDOM), random_state=RANDOM),
            boosting_param_grid, cv=3, n_jobs=-1)
        boosting_grid.fit(X_boost_small, y_boost_small)
        print(f"Best Boosting Parameters: {boosting_grid.best_params_}")
        final_boost = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=3, random_state=RANDOM),
                                         random_state=RANDOM, **boosting_grid.best_params_)
        result_boost = evaluate_classifier(final_boost, X_train, X_test, y_train, y_test, classes, "Boosting")
        results.append(result_boost)

    # Multi-layer Perceptron
    if classifier_name in [None, "MLP"]:
        print("\n===== 8. Neural Network (MLP) =====")
        mlp_sample_size = 50000
        rng_mlp = np.random.RandomState(RANDOM)
        idx_mlp = rng_mlp.choice(len(X_train), mlp_sample_size, replace=False)
        X_mlp_small = X_train[idx_mlp]
        y_mlp_small = y_train.iloc[idx_mlp] if hasattr(y_train, 'iloc') else y_train[idx_mlp]

        mlp_param_grid = {'hidden_layer_sizes': [(50,), (100,)], 'alpha': [0.001, 0.01]}
        mlp_grid = GridSearchCV(MLPClassifier(max_iter=1000, random_state=RANDOM), mlp_param_grid, cv=3, n_jobs=-1)
        mlp_grid.fit(X_mlp_small, y_mlp_small)
        print(f"Best MLP Parameters: {mlp_grid.best_params_}")
        final_mlp = MLPClassifier(max_iter=1000, random_state=RANDOM, **mlp_grid.best_params_)
        result_mlp = evaluate_classifier(final_mlp, X_train, X_test, y_train, y_test, classes, "Neural Network (MLP)")
        results.append(result_mlp)

    # Save results to CSV
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        filename = f"classification_results_{classifier_name or 'ALL'}.csv"
        results_df.to_csv(filename, index=False)

    return results_df


def load_and_combine_results():
    """Load and combine all individual classifier CSV results"""
    csv_files = glob.glob("classification_results_*.csv")
    csv_files = [f for f in csv_files if "ALL" not in f]

    if not csv_files:
        print("No CSV files found. Please run individual classifiers first.")
        return None

    print(f"Found {len(csv_files)} result files: {csv_files}")

    all_results = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        all_results.append(df)

    if not all_results:
        return None

    # Combine and remove duplicates
    combined_df = pd.concat(all_results, ignore_index=True)
    combined_df = combined_df.sort_values('Accuracy', ascending=False).drop_duplicates(subset=['Model'])

    return combined_df


def plot_comparison(results_df):
    """Plot comprehensive comparison of all classifiers"""
    plt.figure(figsize=(14, 8))
    x = range(len(results_df))
    width = 0.15

    plt.bar([i - 2 * width for i in x], results_df['Accuracy'], width, label='Accuracy', color='blue')
    plt.bar([i - width for i in x], results_df['Precision'], width, label='Precision', color='green')
    plt.bar([i for i in x], results_df['Recall'], width, label='Recall', color='orange')
    plt.bar([i + width for i in x], results_df['F1-Score'], width, label='F1-Score', color='red')
    plt.bar([i + 2 * width for i in x], results_df['Specificity'], width, label='Specificity', color='purple')

    plt.xlabel('Classifier')
    plt.ylabel('Score')
    plt.title('Classification Performance Comparison')
    plt.xticks(x, results_df['Model'], rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    # plt.savefig("Classification_Comparison.png", dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    df = pd.read_csv("BTCUSDT.csv", low_memory=False)

    from Feature_Engineer import load_or_create_agg_df

    agg_df = load_or_create_agg_df(df)

    # ========== MODE SELECTION ==========
    # Mode 1: Run a specific classifier (e.g., "MLP", "KNN", "SVM", "LDA", "DecisionTree", "LogisticRegression", "NaiveBayes", "RandomForest", "Ensemble")
    # Mode 2: Run all classifiers at once (set to "ALL")
    # Mode 3: Compare results only from already completed classifiers (set to "COMPARE")

    mode = "COMPARE"  # Change this to switch modes

    if mode == "ALL":
        print("\n===== RUNNING ALL CLASSIFIERS =====\n")
        classifiers_to_run = ["LDA", "DecisionTree", "LogisticRegression", "KNN", "SVM", "NaiveBayes", "RandomForest",
                              "Ensemble", "MLP"]

        for clf_name in classifiers_to_run:
            print(f"\n{'=' * 70}")
            print(f"Running {clf_name}...")
            print(f"{'=' * 70}\n")
            results_df = classification(agg_df, classifier_name=clf_name)

        print("\n===== ALL CLASSIFIERS COMPLETED =====\n")
        print("Now generating comprehensive comparison...")

        all_results_df = load_and_combine_results()
        if all_results_df is not None:
            print("\n===== FINAL COMPARISON TABLE =====\n")
            print(all_results_df.to_string(index=False))

            best_model = all_results_df.iloc[0]['Model']
            best_accuracy = all_results_df.iloc[0]['Accuracy']
            print(f"\n{'=' * 70}")
            print(f"BEST CLASSIFIER: {best_model}")
            print(f"Accuracy: {best_accuracy:.4f}")
            print(f"{'=' * 70}\n")

            plot_comparison(all_results_df)
            all_results_df.to_csv("classification_results_ALL.csv", index=False)

    elif mode == "COMPARE":
        print("\n===== GENERATING COMPREHENSIVE COMPARISON =====\n")
        all_results_df = load_and_combine_results()

        if all_results_df is not None:
            print("\n===== COMPARISON TABLE =====\n")
            print(all_results_df.to_string(index=False))

            best_model = all_results_df.iloc[0]['Model']
            best_accuracy = all_results_df.iloc[0]['Accuracy']
            print(f"\n{'=' * 70}")
            print(f"BEST CLASSIFIER: {best_model}")
            print(f"Accuracy: {best_accuracy:.4f}")
            print(f"{'=' * 70}\n")

            plot_comparison(all_results_df)
            all_results_df.to_csv("classification_results_ALL.csv", index=False)

    else:
        # Single classifier mode
        results_file = f"classification_results_{mode}.csv"

        if os.path.exists(results_file):
            print(f"\nLoading cached results from {results_file}\n")
            results_df = pd.read_csv(results_file)
            print(results_df.to_string(index=False))
        else:
            print(f"\nRunning {mode}...\n")
            results_df = classification(agg_df, classifier_name=mode)

    print("\n===== Analysis Complete =====")


