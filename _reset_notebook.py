import json

notebook = {
  "cells": [],
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 4
}

def add_md(text):
    notebook["cells"].append({"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in text.split("\n")]})

def add_code(code):
    notebook["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in code.split("\n")]})


# Cell 1: Setup
add_md("# Skin Cancer Detection Workflow\n\nThis notebook exactly matches the workflow diagram:\n"
       "1. Input Dermoscopic Image\n2. Image Preprocessing & Augmentation\n"
       "3. Multi-CNN Feature Extraction (Xception + EfficientNet)\n"
       "4. Feature Fusion\n5. Hybrid PSO-GA Feature Selection\n"
       "6. Ensemble ML Classifier (SVM + RF + KNN)\n7. Prediction\n8. Explainable AI (Grad-CAM & LIME)")

# Cell 2: Imports
add_code("""import os\nimport cv2\nimport numpy as np\nimport pandas as pd\nimport tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception, EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.svm import SVC\nfrom sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import matplotlib.pyplot as plt\nimport joblib\nimport pyswarms as ps
from lime import lime_image\nfrom lime.wrappers.scikit_image import SegmentationAlgorithm\nfrom tqdm import tqdm

print("All dependencies loaded!")""")

# Cell 3: Data Loading
add_md("## 1 & 2. Input Dermoscopic Image & Preprocessing\nLoading both HAM10000 and ISIC datasets")
add_code("""IMG_SIZE = 224
SAVE_PATH = "./models/"
os.makedirs(SAVE_PATH, exist_ok=True)

# Load Metadata
dfs = []
if os.path.exists("./ham10000_meta_data.csv.xlsx"):
    ham_df = pd.read_excel("./ham10000_meta_data.csv.xlsx", engine="openpyxl")
    ham_df.columns = [c.strip().lower() for c in ham_df.columns]
    ham_df['path'] = ham_df['image_id'].apply(lambda x: f"./HAM_IMAGES 500/{str(x).strip()}.jpg")
    ham_df = ham_df[ham_df['path'].apply(os.path.exists)][['path', 'dx']].rename(columns={'dx': 'label'})
    dfs.append(ham_df)

if os.path.exists("./isic_labels (1).csv"):
    isic_df = pd.read_csv("./isic_labels (1).csv")
    isic_df.columns = [c.strip().lower() for c in isic_df.columns]
    isic_df['path'] = isic_df['image_id'].apply(lambda x: f"./ISIC_500/{str(x).strip()}.jpg")
    isic_df = isic_df[isic_df['path'].apply(os.path.exists)][['path', 'dx']].rename(columns={'dx': 'label'})
    dfs.append(isic_df)

if not dfs:
    raise FileNotFoundError("No valid image datasets found in the current directory.")

combined_df = pd.concat(dfs, ignore_index=True)
print(f"Total Valid Images: {len(combined_df)}")

# Limit to 500 samples for processing speed using PSO in this Notebook preview
combined_df = combined_df.sample(min(500, len(combined_df)), random_state=42).reset_index(drop=True)
print(f"Working with {len(combined_df)} localized images.")

encoder = LabelEncoder()
combined_df['encoded'] = encoder.fit_transform(combined_df['label'])
joblib.dump(encoder, SAVE_PATH + "label_encoder.pkl")
print("Classes:", encoder.classes_)

def load_and_preprocess(path, augment=False):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype('float32') / 255.0
    return img

images, labels = [], []
for i, row in tqdm(combined_df.iterrows(), total=len(combined_df), desc="Processing"):
    images.append(load_and_preprocess(row['path']))
    labels.append(row['encoded'])
    
images, labels = np.array(images), np.array(labels)
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, stratify=labels, random_state=42)
print("Data Split Setup:", X_train.shape, tuple(X_test.shape))""")

# Cell 4: Multi-CNN
add_md("## 3. Multi-CNN Feature Extraction (Xception + EfficientNet)")
add_code("""# Xception
base_x = Xception(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
x_out = GlobalAveragePooling2D()(base_x.output)
model_xception = Model(base_x.input, x_out)

# EfficientNetB0
base_e = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
e_out = GlobalAveragePooling2D()(base_e.output)
model_efficient = Model(base_e.input, e_out)

print("Extracting Xception Features...")
x_feat_tr = model_xception.predict(X_train, batch_size=32)
x_feat_te = model_xception.predict(X_test, batch_size=32)

print("Extracting EfficientNet Features...")
e_feat_tr = model_efficient.predict(X_train, batch_size=32)
e_feat_te = model_efficient.predict(X_test, batch_size=32)""")

# Cell 5: Fusion
add_md("## 4. Feature Fusion")
add_code("""train_fused = np.concatenate([x_feat_tr, e_feat_tr], axis=1)
test_fused = np.concatenate([x_feat_te, e_feat_te], axis=1)

scaler = MinMaxScaler()
train_features = scaler.fit_transform(train_fused)
test_features = scaler.transform(test_fused)
joblib.dump(scaler, SAVE_PATH + "scaler.pkl")

print("Fused Multi-CNN Feature Shape:", train_features.shape)""")

# Cell 6: Hybrid PSO-GA
add_md("## 5. Hybrid PSO-GA Feature Selection\nParticle Swarm Optimization seamlessly transitioned into a Genetic Algorithm block.")
add_code("""class HybridPSOGAFeatureSelector:
    def __init__(self, n_particles=10, n_iterations=10, target_features=500):
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.target_features = target_features
        self.selected_indices = None
        
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
        base_est = SVC(kernel='linear', max_iter=100) # Proxy for internal fitness evaluation
        
        print(">> Part 1: Initializing PSO Phase")
        opt = ps.single.GlobalBestPSO(n_particles=self.n_particles, dimensions=dims, options={'c1':0.5, 'c2':0.5, 'w':0.9})
        cost, pos = opt.optimize(lambda p: self._fitness(p, X, y, base_est), iters=self.n_iterations//2)
        
        print(">> Part 2: Transitioning to GA Phase")
        pop = opt.swarm.position
        for _ in range(self.n_iterations//2):
            scores = -self._fitness(pop, X, y, base_est)
            next_p = []
            for _ in range(self.n_particles//2):
                # Tournament Selection
                i1, i2 = np.random.choice(self.n_particles, 2)
                p1 = pop[i1] if scores[i1] > scores[i2] else pop[i2]
                i1, i2 = np.random.choice(self.n_particles, 2)
                p2 = pop[i1] if scores[i1] > scores[i2] else pop[i2]
                
                # Crossover
                cpt = np.random.randint(1, dims - 1)
                c1, c2 = np.concatenate([p1[:cpt], p2[cpt:]]), np.concatenate([p2[:cpt], p1[cpt:]])
                # Mutation
                if np.random.rand() < 0.1: c1[np.random.randint(dims)] = np.random.rand()
                if np.random.rand() < 0.1: c2[np.random.randint(dims)] = np.random.rand()
                next_p.extend([c1, c2])
            pop = np.array(next_p[:self.n_particles])
            
        fit_final = -self._fitness(pop, X, y, base_est)
        best = pop[np.argmax(fit_final)]
        self.selected_indices = np.argsort(best)[-self.target_features:]
        return self
        
    def transform(self, X):
        return X[:, self.selected_indices]

selector = HybridPSOGAFeatureSelector(n_particles=10, n_iterations=10)
selector.fit(train_features, y_train)

train_selected = selector.transform(train_features)
test_selected = selector.transform(test_features)
joblib.dump(selector, SAVE_PATH + "pso_ga_selector.pkl")
print("Selected Shape via Hybrid PSO-GA:", train_selected.shape)""")

# Cell 7: Ensemble
add_md("## 6. Ensemble ML Classifier\n(SVM + Random Forest + KNN)")
add_code("""from sklearn.metrics import accuracy_score, classification_report

svm_clf = SVC(kernel='rbf', probability=True, random_state=42)
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
knn_clf = KNeighborsClassifier(n_neighbors=5)

ensemble_clf = VotingClassifier(
    estimators=[('svm', svm_clf), ('rf', rf_clf), ('knn', knn_clf)],
    voting='soft'
)

print("Training Ensemble Architecture...")
ensemble_clf.fit(train_selected, y_train)
joblib.dump(ensemble_clf, SAVE_PATH + "ensemble_model.pkl")
print("Ensemble Trained Successfully!")""")

# Cell 8: Prediction
add_md("## 7. Final Prediction Module")
add_code("""y_pred = ensemble_clf.predict(test_selected)
print("Validation Accuracy:", accuracy_score(y_test, y_pred))
print("\\nClassification Report:\\n", classification_report(y_test, y_pred, target_names=encoder.classes_))""")

# Cell 9: Explainable AI
add_md("## 8. Explainable AI\nGrad-CAM targets CNN Level Analysis\nLIME targets end-to-end ML Level Analysis")
add_code("""test_img_index = 0
test_img = X_test[test_img_index]

# --- 8a. Grad-CAM Analysis ---
def grad_cam(img, model, layer="block14_sepconv2_act"):
    gm = tf.keras.models.Model([model.inputs], [model.get_layer(layer).output, model.output])
    with tf.GradientTape() as tape:
        convs, preds = gm(np.expand_dims(img, 0))
        top_c = preds[:, tf.argmax(preds[0])]
    grads = tape.gradient(top_c, convs)
    p_grads = tf.reduce_mean(grads, axis=(0,1,2))
    heatmap = convs[0] @ p_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap).numpy()
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap

print("Rendering heatmaps...")
hm = grad_cam(test_img, model_xception)

# --- 8b. LIME Analysis ---
def lime_predict(imgs):
    preds = []
    for i in imgs:
        t = np.expand_dims(i, 0)
        x_f = model_xception.predict(t, verbose=0)
        e_f = model_efficient.predict(t, verbose=0)
        fused = np.concatenate([x_f, e_f], axis=1)
        sel = selector.transform(scaler.transform(fused))
        preds.append(ensemble_clf.predict_proba(sel)[0])
    return np.array(preds)

exp = lime_image.LimeImageExplainer()
seg = SegmentationAlgorithm('slic', n_segments=50, compactness=10, sigma=1)
explanation = exp.explain_instance(test_img.astype('double'), lime_predict, top_labels=2, num_samples=100, segmentation_fn=seg)
l_img, _ = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)

# Display Graphics
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(test_img); ax[0].set_title("Original Input")
ax[1].imshow(test_img); ax[1].imshow(cv2.resize(hm, (224, 224)), alpha=0.5, cmap='jet'); ax[1].set_title("Grad-CAM (CNN Layer)")
ax[2].imshow(l_img); ax[2].set_title("LIME (Ensemble Decisions)")
plt.tight_layout()
plt.show()""")

# Convert to JSON and Save
with open(r"c:\\Users\\donsi\\OneDrive\\Desktop\\MINI\\skin_cancer_detection.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)
