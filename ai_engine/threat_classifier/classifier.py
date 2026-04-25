import numpy as np
import pickle
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

THREAT_LEVELS = {0:"Normal",1:"Low",2:"Medium",3:"High",4:"Critical"}

class SSLThreatClassifier:
    def __init__(self):
        self.model = GradientBoostingClassifier(n_estimators=200,learning_rate=0.05,max_depth=4,subsample=0.8,random_state=42)
        self.is_trained = False
        self.metrics = {}

    def build_feature_vector(self, ssl_embedding, behaviour_features):
        KEYS = ["loitering_score","crowd_density","running_score","person_count","avg_speed","anomaly_score","time_in_zone","zone_transitions"]
        beh = np.array([behaviour_features.get(k,0.0) for k in KEYS],dtype=np.float32)
        return np.concatenate([ssl_embedding, beh])

    def train(self, X, y):
        X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
        self.model.fit(X_train,y_train)
        self.is_trained = True
        self.evaluate(X_val,y_val)

    def evaluate(self, X, y):
        y_pred = self.model.predict(X)
        labels = list(range(5))
        names  = [THREAT_LEVELS[i] for i in labels]
        p = precision_score(y,y_pred,labels=labels,average=None,zero_division=0)
        r = recall_score(y,y_pred,labels=labels,average=None,zero_division=0)
        f = f1_score(y,y_pred,labels=labels,average=None,zero_division=0)
        self.metrics = {
            "per_class":{n:{"precision":round(float(p[i]),4),"recall":round(float(r[i]),4),"f1":round(float(f[i]),4)} for i,n in enumerate(names)},
            "macro":{"precision":round(float(p.mean()),4),"recall":round(float(r.mean()),4),"f1":round(float(f.mean()),4)},
            "confusion_matrix":confusion_matrix(y,y_pred,labels=labels).tolist(),
        }
        print(classification_report(y,y_pred,target_names=names,zero_division=0))
        return self.metrics

    def predict(self, ssl_embedding, behaviour_features):
        if not self.is_trained:
            return {"threat_level":0,"threat_label":"Normal","confidence":0.0,"probabilities":{}}
        fv = self.build_feature_vector(ssl_embedding, behaviour_features)
        level = int(self.model.predict([fv])[0])
        proba = self.model.predict_proba([fv])[0]
        return {"threat_level":level,"threat_label":THREAT_LEVELS[level],"confidence":round(float(proba[level]),4),"probabilities":{THREAT_LEVELS[i]:round(float(p),4) for i,p in enumerate(proba)}}

    def save(self, path="models/threat_classifier.pkl"):
        os.makedirs(os.path.dirname(path),exist_ok=True)
        pickle.dump(self,open(path,"wb"))

    @staticmethod
    def load(path="models/threat_classifier.pkl"):
        return pickle.load(open(path,"rb"))
