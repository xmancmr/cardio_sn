import streamlit as st
import joblib
import pandas as pd

# Chargement du modèle et des noms de features
@st.cache_resource
def load_artifacts(job: str, feature: str):
    try:
        model = joblib.load(job)
        feature_names = joblib.load(feature)
        return model, feature_names
    except FileNotFoundError as e:
        st.error(f"Erreur de chargement: {e}. Veuillez d'abord exécuter les Jupiter notebooks")
        return None, None
    
def load_data():
        data = pd.read_csv("./csv/cleaned_dataset.csv")
        return data
    
dataset = load_data()
st.session_state['dataset'] = dataset

with st.sidebar:
    st.header("**Parametrage**")
model_select = st.sidebar.selectbox("Modele",
                                    ("Decision Tree", "Naive Bayes"))
 
st.session_state['model_select'] = model_select

if model_select == "Decision Tree":
    model, feature_names = load_artifacts('./params/joblib/heart_model_pipeline_decision_tree.joblib', './params/joblib/feature_names_decision_tree.joblib')
elif model_select == "Naive Bayes":
    model, feature_names = load_artifacts('./params/joblib/heart_model_pipeline_naive_bayes.joblib', './params/joblib/feature_names_naive_bayes.joblib')  


st.title("**🔍 Prédiction de Risque Cardiaque**")
st.info("**Nous predisons le risque de maladie cardiaque en fonction des paramètres médicaux.**")

# Section de saisie
st.header("📋 Informations du Patient")

# Organisation en colonnes
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Âge", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Sexe", options=[0, 1], format_func=lambda x: "Femme" if x == 0 else "Homme")
    cp = st.selectbox("Type de douleur thoracique", options=[0, 1, 2, 3], help="0: typique, 1: atypique, 2: non-douloureux, 3: asymptomatique")
    trestbps = st.number_input("Pression artérielle (mm Hg)", min_value=90, max_value=200, value=120)
    chol = st.number_input("Cholestérol (mg/dl)", min_value=100, max_value=600, value=200)
    
with col2:
    fbs = st.selectbox("Glycémie > 120 mg/dl", options=[0, 1])
    restecg = st.selectbox("Résultat ECG", options=[0, 1, 2])
    thalach = st.number_input("Fréquence cardiaque max", min_value=60, max_value=220, value=150)
    exang = st.selectbox("Angine induite", options=[0, 1])
    oldpeak = st.number_input("Dépression ST", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

# Deuxième ligne de colonnes
col3, col4 = st.columns(2)
with col3:
    slope = st.selectbox("Pente ST", options=[0, 1, 2], index=1)

# Création du dataframe d'entrée dans le bon ordre
input_dict = {
            'age': age,
            'sex': sex,
            'chest pain type': cp,
            'resting bp s': trestbps,
            'cholesterol': chol,
            'fasting blood sugar': fbs,
            'resting ecg': restecg,
            'exercise angina': exang,
            'oldpeak': oldpeak,
            'ST slope': slope
        }
# Prédiction
input_df = pd.DataFrame([input_dict], columns=feature_names)
st.session_state['input_df'] = input_df
prediction = model.predict(input_df)
proba = model.predict_proba(input_df)[0]

# Bouton de prédiction
if st.button("🤖 Prédire le risque"):
    if model is not None:
        try:
            # Affichage des résultats
            st.subheader("📊 Résultats")
            if prediction[0] == 1:
                st.error(f"**🤖** Risque élevé de maladie cardiaque ({proba[1]:.1%} de probabilité)")
                st.markdown("""
                **Recommandations:**
                - Consultation cardiologique urgente
                - Examens complémentaires recommandés
                - Modifications du mode de vie
                """)
            else:
                st.success(f"**🤖**  Risque faible de maladie cardiaque ({proba[0]:.1%} de probabilité)")
                st.markdown("""
                **Recommandations:**
                - Maintenir un mode de vie sain
                - Contrôles réguliers recommandés
                """)
                
            # Graphique de probabilité
            proba_df = pd.DataFrame({
                'Classe': ['Sain', 'Malade'],
                'Probabilité': proba
            })
            st.bar_chart(proba_df.set_index('Classe'))
            
        except Exception as e:
            st.error(f"Erreur lors de la prédiction: {e}")
    else:
        st.warning("Le modèle n'est pas chargé. Veullez d'abord executez les Jupiter notebooks.")

# Section d'information
st.markdown("---")
st.info("**Ce modèle a été entraîné sur un jeu de données nommé 'heartdisease.csv'. Les résultats obtenus sont des estimations et ne remplacent pas un diagnostic médical professionnel.**")