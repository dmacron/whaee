Exp 1 
To collect, preprocess, integrate, and transform healthcare data related to a specific disease for further analysis.

import pandas as pd
from sklearn.preprocessing import StandardScaler
df = pd.DataFrame({
    "Name": ["Amit", "Riya", "Neha", "Raj", "Sara", "Amit"],
    "Age": [45, 30, 55, 40, 25, 45],
    "Gender": ["Male", "Female", "Female", "Male", "Female", "Male"],
    "Medical Condition": ["Diabetes", "Asthma", "Diabetes", "Diabetes", "Asthma", "Diabetes"],
    "Billing Amount": [5000, 3000, 7000, 6000, 2500, 5000]
})
print("Original Dataset:")
print(df)

df = df.drop_duplicates()
df["Age"] = df["Age"].fillna(df["Age"].median())
d = df[df["Medical Condition"] == "Diabetes"].copy()
d["Record_ID"] = range(1, len(d) + 1)
patient = d[["Record_ID", "Name", "Age", "Gender"]]
clinical = d[["Record_ID", "Medical Condition", "Billing Amount"]]
data = pd.merge(patient, clinical, on="Record_ID")

data = pd.get_dummies(data, columns=["Gender"], dtype=int)
cols = ["Age", "Billing Amount"]
data[cols] = StandardScaler().fit_transform(data[cols])

data.to_csv("diabetes_dataset.csv", index=False)
print("\nFinal Dataset:")
print(data)


Exp 2
To analyze healthcare data using Exploratory Data Analysis (EDA) techniques with Pandas and Matplotlib.

import pandas as pd
import matplotlib.pyplot as plt
df = pd.DataFrame({
    "Age": [25, 45, 60, 35, 70, 50, 40, 65],
    "Gender": ["Male", "Female", "Female", "Male", "Female", "Male", "Female", "Male"],
    "Medical Condition": ["Diabetes", "Diabetes", "Asthma", "Diabetes", "Diabetes", "Asthma”,  "Diabetes", "Diabetes"],
    "Billing Amount": [2000, 5000, 3000, 4000, 7000, 2500, 4500, 6000],
    "Test Results": ["Normal", "Abnormal", "Normal", "Abnormal", "Inconclusive", "Normal", "Abnormal", "Inconclusive"]
})
d = df[df["Medical Condition"] == "Diabetes"]

print(d.head())
print(d.describe())
print("Missing values:\n", d.isnull().sum())
print("Duplicates:", d.duplicated().sum())
print("Shape:", d.shape)

plt.figure(figsize=(10, 7))

plt.subplot(2, 2, 1)
plt.hist(d["Age"], edgecolor="black")
plt.title("Age Distribution")

plt.subplot(2, 2, 2)
d["Gender"].value_counts().plot(kind="bar")
plt.title("Gender Count")

plt.subplot(2, 2, 3)
plt.hist(d["Billing Amount"], edgecolor="black")
plt.title("Billing Distribution")

plt.subplot(2, 2, 4)
d["Test Results"].value_counts().plot(kind="bar")
plt.title("Test Results")
plt.tight_layout()
plt.show()


Exp3 
To identify patterns and relationships in healthcare data using Seaborn and perform predictive data modeling.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
df = pd.DataFrame({
    "Age": [25,40,55,30,65,45,35,50,60,28,42,70,
            32,48,58,22,38,62,27,53,44,67,36,49],
    "Billing Amount": [2000,4000,6000,3000,7000,5000,3500,4500,
                       6500,2500,4200,8000,3200,4800,6200,1800,
                       3800,7200,2300,5500,4300,7500,3600,4900],
    "Room Number": [101,102,103,104,105,106,107,108,109,110,111,112, 113,114,115,116,117,118,119,120,121,122,123,124],
    "Gender": ["Male","Female"] * 12
})

      sns.histplot(data=df, x="Age", kde=True)
plt.show()

sns.boxplot(data=df, x="Billing Amount")
plt.show()

sns.scatterplot(data=df, x="Age", y="Billing Amount")
plt.show()

sns.heatmap(df[["Age","Billing Amount","Room Number"]].corr(),
            annot=True)
plt.show()

X = df[["Age", "Billing Amount", "Room Number"]]
y = df["Gender"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = make_pipeline(StandardScaler(), LogisticRegression())
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred))



Exp 4
To analyze the features of MRI images using image segmentation techniques.

import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

uploaded = files.upload()
img = cv2.imdecode(np.frombuffer(next(iter(uploaded.values())),
                    np.uint8), cv2.IMREAD_COLOR)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

_, seg = cv2.threshold(blur, 0, 255,
                       cv2.THRESH_BINARY + cv2.THRESH_OTSU)

contours, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)

plt.figure(figsize=(10, 5))
for i, (image, title) in enumerate([
    (gray, "Grayscale"), (blur, "Blurred"),
    (seg, "Segmented"), (cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                         "Original")
]):
    plt.subplot(2, 2, i+1)
    plt.imshow(image, cmap="gray" if i < 3 else None)
    plt.title(title)
    plt.axis("off")
plt.tight_layout()
plt.show()

if contours:
    c = max(contours, key=cv2.contourArea)
    print("Area:", cv2.contourArea(c))
    print("Perimeter:", cv2.arcLength(c, True))
    print("Bounding Box:", cv2.boundingRect(c))
else:
    print("No contours detected.")

cv2.imwrite("segmented_mri.png", seg)
files.download("segmented_mri.png")





Exp 5
To Study Features of iPHR using openMRS

Exp 6 
To study and implement various evaluation metrics for assessing the performance of healthcare data models.

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

np.random.seed(42)
df = pd.DataFrame({
    "Age": np.random.randint(18, 80, 200),
    "Cholesterol": np.random.randint(130, 280, 200),
    "Gender": np.random.choice(["Male", "Female"], 200)
})

df["Outcome"] = (df["Cholesterol"] >= 200).astype(int)
X = pd.get_dummies(df[["Age", "Gender"]], dtype=int)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
print("Precision:", precision_score(y_test, pred, zero_division=0))
print("Recall:", recall_score(y_test, pred, zero_division=0))
print("F1 Score:", f1_score(y_test, pred, zero_division=0))

cm = confusion_matrix(y_test, pred)
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=["Normal", "High"],
            yticklabels=["Normal", "High"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()




Exp 7
To implement ensemble learning techniques for disease prediction using healthcare data.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

np.random.seed(42)
df = pd.DataFrame({
    "Age": np.random.randint(18, 85, 200),
    "Cholesterol": np.random.randint(130, 280, 200),
    "BP": np.random.randint(100, 180, 200)
})

df["Disease"] = np.where(df["BP"] >= 145, "Hypertension",
                  np.where(df["Cholesterol"] >= 220, "Diabetes", "Healthy"))

X = df.drop("Disease", axis=1)
y = df["Disease"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=5),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "Voting": VotingClassifier(estimators=[
        ("lr", LogisticRegression(max_iter=1000)),
        ("dt", DecisionTreeClassifier()),
    ], voting="hard")
}

scores = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    scores[name] = accuracy_score(y_test, model.predict(X_test))
    print(name, "Accuracy:", round(scores[name] * 100, 2), "%")

plt.bar(scores.keys(), scores.values())
plt.title("Model Accuracy Comparison")
plt.show()


Exp 8
To identify and extract important medical information from clinical reports using NLP

!pip -q install spacy
!python -m spacy download en_core_web_sm -q

import spacy
import pandas as pd

nlp = spacy.load("en_core_web_sm")

report = """Patient Swayam Raut was admitted to Jaslok Hospital
on 12 September 2026 with fever and chest pain.
He was diagnosed with pneumonia and prescribed Azithromycin."""

doc = nlp(report)

entities = [(e.text, e.label_) for e in doc.ents]
df = pd.DataFrame(entities, columns=["Entity", "Entity Type"])

print(df)

terms = ["pneumonia", "Azithromycin", "fever", "chest pain"]
found = [t for t in terms if t.lower() in report.lower()]

print("Medical Terms:", found)
df.to_csv("Medical_Entities.csv", index=False)
pd.DataFrame(found, columns=["Medical Term"]).to_csv(
    "Medical_Terms.csv", index=False
)

Exp9
To study and implement Explainable AI (XAI) in healthcare for interpreting machine learning model predictions.

!pip -q install shap

import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from google.colab import files

uploaded = files.upload()
df = pd.read_csv(next(iter(uploaded)))
df.columns = df.columns.str.strip()

target = "Life expectancy"
df = df.select_dtypes(include="number").dropna()
X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("R2 Score:", r2_score(y_test, pred))

explainer = shap.TreeExplainer(model)
values = explainer(X_test.sample(min(100, len(X_test)), random_state=42))

shap.summary_plot(values, X_test.sample(
    min(100, len(X_test)), random_state=42
))



