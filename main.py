import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier


# ====================================
# CARREGAMENTO DOS DADOS
# ====================================

df = pd.read_excel(
    "data/credit_default.xls",
    header=1
)

# Remover coluna ID
df = df.drop(columns=["ID"])

#Removendo colunas desnecessarias
df = df.drop(columns=[
    "SEX",
    "MARRIAGE",
    "PAY_6",
    "PAY_5",
    "EDUCATION",
    "PAY_4",
    "PAY_3"
])

# ====================================
# DEFINIR X E y
# ====================================

X = df.drop("default payment next month", axis=1)

y = df["default payment next month"]

# ====================================
# DIVIDIR TREINO E TESTE
# ====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTreino:", X_train.shape)
print("Teste:", X_test.shape)

# ====================================
# NORMALIZAÇÃO
# ====================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ====================================
# TREINAR MODELO
# ====================================

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ),

    "Decision Tree":
        DecisionTreeClassifier(
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced"
        ),

    "Gradient Boosting":
        GradientBoostingClassifier(
            random_state=42
        )
}

predictions = {}

for name, model in models.items():

    print(f"\nTreinando: {name}")

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    predictions[name] = pred

    print(
        classification_report(
            y_test,
            pred
        )
    )

print("\nModelo treinado com sucesso!")

#----------------------------
rf = models["Random Forest"]

importancias = pd.DataFrame({
    "feature": X.columns,
    "importance": rf.feature_importances_
})

importancias = importancias.sort_values(
    by="importance",
    ascending=False
)

print(importancias.head(10))
#----------------------------------

# =========================
# ENSEMBLE VOTING
# =========================
#alterando de voto igualitario para voto ponderado, onde o modelo de Random Forest tem mais peso, seguido pelo Gradient Boosting, e os outros dois modelos tem peso igual.
'''
all_predictions = np.column_stack(
    list(predictions.values())
)

# Soma dos votos
votes = np.sum(all_predictions, axis=1)

# Maioria simples
final_prediction = (votes >= 2).astype(int)

print("\nENSEMBLE VOTING:\n")

print(
    classification_report(
        y_test,
        final_prediction
    )
)

'''
ensemble = VotingClassifier(
    estimators=[
        ("lr", models["Logistic Regression"]),
        ("dt", models["Decision Tree"]),
        ("rf", models["Random Forest"]),
        ("gb", models["Gradient Boosting"])
    ],
    voting="soft",
    weights=[1, 1, 3, 4]
)

ensemble.fit(X_train, y_train)

probabilidades = ensemble.predict_proba(X_test)[:, 1]

threshold = 0.35

final_prediction = (
    probabilidades >= threshold
).astype(int)

print("\nENSEMBLE VOTING:\n")

print(
    classification_report(
        y_test,
        final_prediction
    )
)
# =========================
# PROBABILIDADES
# =========================

y_prob = model.predict_proba(X_test)[:, 1]

# =========================
# DECISÕES
# =========================

decisions = []

for prob in y_prob:

    if prob < 0.4:
        decisions.append("APROVADO")

    elif prob < 0.7:
        decisions.append("REVISÃO MANUAL")

    else:
        decisions.append("REPROVADO")

# =========================
# EXEMPLOS
# =========================

for i in range(10):

    print(
        f"Probabilidade: {y_prob[i]:.2f} "
        f"-> {decisions[i]}"
    )

# ====================================
# AVALIAÇÃO
# ====================================

#print("\nRELATÓRIO DE CLASSIFICAÇÃO:\n")

#print(classification_report(y_test, y_pred))

#print("\nMATRIZ DE CONFUSÃO:\n")

#print(confusion_matrix(y_test, y_pred))

#print(importancias.tail(10))

probabilidades = ensemble.predict_proba(X_test)

print(probabilidades[:10])