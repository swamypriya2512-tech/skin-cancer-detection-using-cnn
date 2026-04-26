import os
import re

print(">>> Updating src/pso_selector.py")
pso_code = """import os
import numpy as np
import pyswarms as ps
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import PSO_MASK_PATH

def apply_pso_mask(features, mask):
    return features[:, mask]

def load_pso_mask():
    if os.path.exists(PSO_MASK_PATH):
        mask = np.load(PSO_MASK_PATH)
        return mask
    return None

class HybridPSOGAFeatureSelector:
    def __init__(self, n_particles=10, n_iterations=10, target_features=500, pg_cb=None):
        self.n_particles = n_particles
        self.n_iterations = max(2, n_iterations)
        self.target_features = target_features
        self.selected_indices = None
        self.pg_cb = pg_cb
        
    def _fitness(self, particles, X, y, estimator):
        scores = []
        for p in particles:
            sel = np.where(p > 0.5)[0]
            if len(sel) == 0:
                scores.append(0)
                continue
            X_sub = X[:, sel]
            score = cross_val_score(estimator, X_sub, y, cv=3).mean()
            scores.append(score)
        return -np.array(scores)

    def fit(self, X, y):
        dims = X.shape[1]
        base_est = SVC(kernel='linear', max_iter=50) # Very fast proxy
        
        if self.pg_cb: self.pg_cb("PSO Phase [1/2]...", 0.2)
        opt = ps.single.GlobalBestPSO(n_particles=self.n_particles, dimensions=dims, options={'c1':0.5, 'c2':0.5, 'w':0.9})
        cost, pos = opt.optimize(lambda p: self._fitness(p, X, y, base_est), iters=self.n_iterations // 2, verbose=False)
        
        if self.pg_cb: self.pg_cb("GA Phase (Crossover/Mutation) [2/2]...", 0.4)
        pop = opt.swarm.position
        for _ in range(self.n_iterations // 2):
            scores = -self._fitness(pop, X, y, base_est)
            next_p = []
            for _ in range(self.n_particles // 2):
                i1, i2 = np.random.choice(self.n_particles, 2)
                p1 = pop[i1] if scores[i1] > scores[i2] else pop[i2]
                i1, i2 = np.random.choice(self.n_particles, 2)
                p2 = pop[i1] if scores[i1] > scores[i2] else pop[i2]
                
                cpt = np.random.randint(1, dims - 1)
                c1, c2 = np.concatenate([p1[:cpt], p2[cpt:]]), np.concatenate([p2[:cpt], p1[cpt:]])
                if np.random.rand() < 0.1: c1[np.random.randint(dims)] = np.random.rand()
                if np.random.rand() < 0.1: c2[np.random.randint(dims)] = np.random.rand()
                next_p.extend([c1, c2])
            pop = np.array(next_p[:self.n_particles])
            
        fit_final = -self._fitness(pop, X, y, base_est)
        best = pop[np.argmax(fit_final)]
        self.selected_indices = np.argsort(best)[-min(self.target_features, dims):]
        return self

def pso_feature_select(train_features, y_train, progress_callback=None):
    if progress_callback: progress_callback("Initializing Hybrid PSO-GA...", 0.1)
    
    # We use a constrained 500-feature subset optimization directly to align with app requirements
    from sklearn.feature_selection import VarianceThreshold
    vt = VarianceThreshold(threshold=0.001)
    train_vt = vt.fit_transform(train_features)
    vt_mask = vt.get_support()
    
    n_features = train_vt.shape[1]
    
    # Limit to reasonable dimensions dynamically for PSO speed
    active_features = train_vt
    if n_features > 500:
        variances = np.var(train_vt, axis=0)
        top_idx = np.argsort(variances)[::-1][:500]
        active_features = train_vt[:, top_idx]
        
        subset_mask = np.zeros(n_features, dtype=bool)
        subset_mask[top_idx] = True
        
        combined_mask = np.zeros(train_features.shape[1], dtype=bool)
        combined_mask[np.where(vt_mask)[0][subset_mask]] = True
    else:
        combined_mask = vt_mask
        
    selector = HybridPSOGAFeatureSelector(n_particles=10, n_iterations=10, target_features=300, pg_cb=progress_callback)
    selector.fit(active_features, y_train)
    
    # Convert active indices into boolean mask matching app.py expectation
    mask = np.zeros(train_features.shape[1], dtype=bool)
    active_in_vt = np.where(combined_mask)[0]
    mask[active_in_vt[selector.selected_indices]] = True
    
    np.save(PSO_MASK_PATH, mask)
    if progress_callback: progress_callback(f"Hybrid PSO-GA Selected {mask.sum()} Features", 0.6)
    return mask
"""
with open("src/pso_selector.py", "w") as f:
    f.write(pso_code)

print(">>> Updating src/classifiers.py")
clf_code = """import os, json, joblib, numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import SVM_C, SVM_KERNEL, RF_N_ESTIMATORS, RANDOM_STATE, SVM_PATH, RF_PATH, KNN_PATH, METRICS_PATH

def _metrics(name, y_true, y_pred, classes):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    return {"classifier": name, "accuracy": round(float(acc), 4), "precision": round(float(prec), 4), "recall": round(float(rec), 4), "f1_score": round(float(f1), 4), "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(), "classification_report": classification_report(y_true, y_pred, target_names=classes, output_dict=True, zero_division=0)}

def train_svm(X_train, y_train, progress_callback=None):
    if progress_callback: progress_callback("Training SVM ...", 0.65)
    clf = SVC(C=SVM_C, kernel=SVM_KERNEL, probability=True, random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)
    joblib.dump(clf, SVM_PATH)
    return clf

def train_random_forest(X_train, y_train, progress_callback=None):
    if progress_callback: progress_callback("Training Random Forest ...", 0.70)
    clf = RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced")
    clf.fit(X_train, y_train)
    joblib.dump(clf, RF_PATH)
    return clf

def train_knn(X_train, y_train, progress_callback=None):
    if progress_callback: progress_callback("Training KNN ...", 0.75)
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X_train, y_train)
    joblib.dump(clf, KNN_PATH)
    return clf

def evaluate_all(clfs, X_test, y_test, class_names, progress_callback=None):
    if progress_callback: progress_callback("Evaluating classifiers ...", 0.85)
    results = {}
    for name, clf in clfs.items():
        results[name] = _metrics(name, y_test, clf.predict(X_test), class_names)
    return results

def save_metrics(metrics):
    with open(METRICS_PATH, "w") as f: json.dump(metrics, f, indent=2)

def load_all_classifiers():
    clfs = {}
    for name, path in [("SVM", SVM_PATH), ("Random Forest", RF_PATH), ("KNN", KNN_PATH)]:
        if os.path.exists(path): clfs[name] = joblib.load(path)
    return clfs
"""
with open("src/classifiers.py", "w") as f:
    f.write(clf_code)


print(">>> Updating src/ensemble.py")
ens_code = """import os, joblib
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, f1_score
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import ENSEMBLE_PATH

def build_ensemble(clfs: dict):
    estimators = [(name, clf) for name, clf in clfs.items() if name in ["SVM", "Random Forest", "KNN"]]
    ensemble = VotingClassifier(estimators=estimators, voting="soft")
    return ensemble

def evaluate_ensemble(ensemble, X_test, y_test):
    y_pred = ensemble.predict(X_test)
    return {"classifier": "Ensemble", "accuracy": float(accuracy_score(y_test, y_pred)), "f1_score": float(f1_score(y_test, y_pred, average="weighted", zero_division=0))}
"""
with open("src/ensemble.py", "w") as f:
    f.write(ens_code)


print(">>> Updating src/train.py")
with open("src/train.py", "r") as f:
    tdata = f.read()

# Replace imports
tdata = tdata.replace("train_xgboost", "train_knn").replace("train_mlp", "")
# Replace ML blocks
tdata = re.sub(
    r"if progress_callback:[\s#\w]*progress_callback\(\"Training XGBoost.*?\).*?\n.*xgb_clf.*?\n\n.*?mlp_clf.*?y_train\)",
    r"if progress_callback:\n        progress_callback('Training KNN ...', 0.70)\n    knn_clf = train_knn(train_sel, active_y_train)",
    tdata, flags=re.DOTALL
)

# Replace Dictionary Mapping
tdata = re.sub(
    r"\"XGBoost\":\s*xgb_clf,\n\s*\"MLP Neural Net\":\s*mlp_clf,",
    r"\"KNN\":                 knn_clf,",
    tdata
)

# Apply Ensemble Fitting (Sklearn VotingClassifier requires unfitted or fit() wrapper)
tdata = tdata.replace(
    "ensemble = build_ensemble(clfs)",
    "ensemble = build_ensemble(clfs)\n    # Fit soft voting on exactly the same training subspace\n    ensemble.fit(train_sel, active_y_train)\n    import joblib\n    from src.config import ENSEMBLE_PATH\n    joblib.dump(ensemble, ENSEMBLE_PATH)"
)

with open("src/train.py", "w") as f:
    f.write(tdata)


print(">>> Updating app.py")
with open("app.py", "r", encoding="utf-8") as f:
    adata = f.read()

adata = adata.replace("XGBoost", "KNN")
adata = adata.replace("MLP Neural Net", "KNN")
adata = adata.replace('<span class="hero-badge">KNN</span>\n  <span class="hero-badge">KNN</span>', '<span class="hero-badge">KNN</span>')
adata = adata.replace('("KNN", KNN_PATH), ("KNN", NB_PATH)', '("KNN", KNN_PATH)')
adata = adata.replace('["SVM", "Random Forest", "KNN", "KNN"]', '["SVM", "Random Forest", "KNN"]')

with open("app.py", "w", encoding="utf-8") as f:
    f.write(adata)

print(">>> ALL MODULES UPDATED TO STRICT WORKFLOW ALIGNMENT.")
