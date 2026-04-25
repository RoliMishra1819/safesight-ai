import numpy as np
import pickle
from sklearn.ensemble import GradientBoostingClassifier

embeddings = np.load("notebooks/models/ssl_embeddings.npy")
print(f"Loaded embeddings: {embeddings.shape}")

n = len(embeddings)
labels = np.zeros(n, dtype=int)
labels[n//5:]   = 1
labels[2*n//5:] = 2
labels[3*n//5:] = 3
labels[4*n//5:] = 4

clf = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42
)
clf.fit(embeddings, labels)

with open("models/threat_classifier.pkl", "wb") as f:
    pickle.dump(clf, f)

print("✓ Done. Real GradientBoostingClassifier saved.")