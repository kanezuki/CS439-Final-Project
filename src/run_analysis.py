import os
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.inspection import permutation_importance

RANDOM_STATE = 42
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT = os.path.join(ROOT, 'outputs')
FIG = os.path.join(OUT, 'figures')
DATA = os.path.join(ROOT, 'data')
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)
os.makedirs(DATA, exist_ok=True)


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def save_fig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, name), dpi=220, bbox_inches='tight')
    plt.close()


def main():
    diabetes = load_diabetes(as_frame=True, scaled=True)
    X = diabetes.data.copy()
    y = diabetes.target.copy()
    df = X.copy()
    df['progression'] = y
    df.to_csv(os.path.join(DATA, 'diabetes_dataset.csv'), index=False)

    # EDA
    summary = df.describe().T
    summary.to_csv(os.path.join(OUT, 'dataset_summary.csv'))

    corr = df.corr(numeric_only=True)
    corr.to_csv(os.path.join(OUT, 'correlations.csv'))

    plt.figure(figsize=(7, 4.5))
    plt.hist(y, bins=25)
    plt.title('Distribution of Disease Progression Target')
    plt.xlabel('Disease progression one year after baseline')
    plt.ylabel('Number of patients')
    save_fig('target_distribution.png')

    plt.figure(figsize=(7, 5.5))
    im = plt.imshow(corr, aspect='auto')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha='right')
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title('Feature and Target Correlation Heatmap')
    save_fig('correlation_heatmap.png')

    # PCA for visualization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pcs = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame({'PC1': pcs[:,0], 'PC2': pcs[:,1], 'progression': y})
    pca_df.to_csv(os.path.join(OUT, 'pca_projection.csv'), index=False)

    plt.figure(figsize=(6.5, 5))
    sc = plt.scatter(pcs[:,0], pcs[:,1], c=y, s=22)
    plt.colorbar(sc, label='Disease progression')
    plt.title('PCA Projection of Baseline Clinical Variables')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
    save_fig('pca_projection.png')

    # Regression split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE
    )

    reg_models = {
        'Mean baseline': None,
        'Linear regression': Pipeline([('scaler', StandardScaler()), ('model', LinearRegression())]),
        'Ridge regression': Pipeline([('scaler', StandardScaler()), ('model', RidgeCV(alphas=np.logspace(-3, 3, 25)))]),
        'Lasso regression': Pipeline([('scaler', StandardScaler()), ('model', LassoCV(alphas=np.logspace(-3, 1, 50), cv=5, random_state=RANDOM_STATE, max_iter=20000))]),
        'Random forest regressor': RandomForestRegressor(n_estimators=150, max_depth=5, min_samples_leaf=8, random_state=RANDOM_STATE, n_jobs=1),
        'Gradient boosting regressor': GradientBoostingRegressor(random_state=RANDOM_STATE, n_estimators=150, learning_rate=0.05, max_depth=2)
    }

    reg_results = []
    reg_predictions = {}
    for name, model in reg_models.items():
        if model is None:
            pred = np.repeat(y_train.mean(), len(y_test))
        else:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
        reg_predictions[name] = pred
        reg_results.append({
            'model': name,
            'RMSE': rmse(y_test, pred),
            'MAE': mean_absolute_error(y_test, pred),
            'R2': r2_score(y_test, pred)
        })
    reg_df = pd.DataFrame(reg_results).sort_values('RMSE')
    reg_df.to_csv(os.path.join(OUT, 'regression_results.csv'), index=False)

    best_reg = reg_df.iloc[0]['model']
    plt.figure(figsize=(6, 5))
    plt.scatter(y_test, reg_predictions[best_reg], s=28)
    lo = min(y_test.min(), reg_predictions[best_reg].min())
    hi = max(y_test.max(), reg_predictions[best_reg].max())
    plt.plot([lo, hi], [lo, hi], linestyle='--')
    plt.title(f'Predicted vs. Actual Progression: {best_reg}')
    plt.xlabel('Actual progression')
    plt.ylabel('Predicted progression')
    save_fig('predicted_vs_actual_best_regression.png')

    # Feature importance using best tree model for interpretability
    rf = reg_models['Random forest regressor']
    rf.fit(X_train, y_train)
    perm = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE, n_jobs=1)
    imp_df = pd.DataFrame({'feature': X.columns, 'importance_mean': perm.importances_mean, 'importance_std': perm.importances_std})
    imp_df = imp_df.sort_values('importance_mean', ascending=False)
    imp_df.to_csv(os.path.join(OUT, 'permutation_importance_random_forest.csv'), index=False)
    plt.figure(figsize=(7, 4.5))
    plt.barh(imp_df['feature'], imp_df['importance_mean'], xerr=imp_df['importance_std'])
    plt.gca().invert_yaxis()
    plt.title('Permutation Importance: Random Forest Regressor')
    plt.xlabel('Mean decrease in R²')
    save_fig('feature_importance_random_forest.png')

    # Classification task: top quartile progression vs others
    threshold = y.quantile(0.75)
    y_bin = (y >= threshold).astype(int)
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X, y_bin, test_size=0.25, random_state=RANDOM_STATE, stratify=y_bin
    )
    clf_models = {
        'Majority baseline': None,
        'Gaussian Naive Bayes': Pipeline([('scaler', StandardScaler()), ('model', GaussianNB())]),
        'Logistic regression': Pipeline([('scaler', StandardScaler()), ('model', LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))]),
        'Regularized logistic regression': Pipeline([('scaler', StandardScaler()), ('model', LogisticRegression(C=0.5, penalty='l2', max_iter=2000, random_state=RANDOM_STATE))]),
        'Random forest classifier': RandomForestClassifier(n_estimators=150, max_depth=4, min_samples_leaf=8, random_state=RANDOM_STATE, n_jobs=1)
    }
    clf_results = []
    best_auc = -1
    best_clf_name = None
    best_clf_model = None
    best_clf_pred = None
    for name, model in clf_models.items():
        if model is None:
            preds = np.repeat(y_train_c.mode().iloc[0], len(y_test_c))
            probs = np.repeat(y_train_c.mean(), len(y_test_c))
        else:
            model.fit(X_train_c, y_train_c)
            preds = model.predict(X_test_c)
            probs = model.predict_proba(X_test_c)[:,1]
        auc = roc_auc_score(y_test_c, probs)
        clf_results.append({
            'model': name,
            'accuracy': accuracy_score(y_test_c, preds),
            'balanced_accuracy': balanced_accuracy_score(y_test_c, preds),
            'precision': precision_score(y_test_c, preds, zero_division=0),
            'recall': recall_score(y_test_c, preds, zero_division=0),
            'f1': f1_score(y_test_c, preds, zero_division=0),
            'roc_auc': auc
        })
        if auc > best_auc:
            best_auc = auc
            best_clf_name = name
            best_clf_model = model
            best_clf_pred = preds
    clf_df = pd.DataFrame(clf_results).sort_values('roc_auc', ascending=False)
    clf_df.to_csv(os.path.join(OUT, 'classification_results.csv'), index=False)

    cm = confusion_matrix(y_test_c, best_clf_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not top quartile', 'Top quartile'])
    disp.plot(values_format='d')
    plt.title(f'Confusion Matrix: {best_clf_name}')
    save_fig('confusion_matrix_best_classifier.png')

    # KMeans clustering (unsupervised context)
    kmeans = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=20)
    clusters = kmeans.fit_predict(X_scaled)
    cluster_df = pd.DataFrame({'cluster': clusters, 'progression': y})
    cluster_summary = cluster_df.groupby('cluster')['progression'].agg(['count', 'mean', 'median', 'std']).reset_index()
    cluster_summary.to_csv(os.path.join(OUT, 'cluster_summary.csv'), index=False)
    plt.figure(figsize=(6.5, 5))
    plt.scatter(pcs[:,0], pcs[:,1], c=clusters, s=24)
    plt.title('K-Means Clusters Projected onto First Two PCA Components')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    save_fig('kmeans_pca_projection.png')

    # Metadata JSON for report generation
    meta = {
        'n_samples': int(df.shape[0]),
        'n_features': int(X.shape[1]),
        'features': list(X.columns),
        'target_min': float(y.min()),
        'target_max': float(y.max()),
        'target_mean': float(y.mean()),
        'target_median': float(y.median()),
        'classification_threshold_top_quartile': float(threshold),
        'best_regression_model_by_rmse': best_reg,
        'best_classifier_by_auc': best_clf_name,
        'pca_variance_ratio': [float(pca.explained_variance_ratio_[0]), float(pca.explained_variance_ratio_[1])]
    }
    with open(os.path.join(OUT, 'metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print('\nRegression results:')
    print(reg_df.to_string(index=False))
    print('\nClassification results:')
    print(clf_df.to_string(index=False))
    print('\nMetadata:')
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
