# app.py - Application Streamlit complète avec les deux modèles
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from PIL import Image
from scipy import stats

# Configuration de la page
st.set_page_config(
    page_title="Prédiction de risque cardiaque - Multi-Modèles",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- 1. Chargement des modèles ----
@st.cache_resource
def load_models():
    """Charge tous les modèles et préprocesseurs"""
    try:
        # Modèle Decision Tree existant
        dt_model = joblib.load('./models/heart_model_pipeline_decision_tree.joblib')
        
        # Nouveau modèle ANN
        ann_model = tf.keras.models.load_model('./models/heart_ann_model.h5')
        
        # Normaliseur pour l'ANN
        scaler = joblib.load('./models/ann_scaler.joblib')
        
        # Features names
        feature_names = joblib.load('./models/feature_names_decision_tree.joblib')
        
        return {
            'dt': dt_model,
            'ann': ann_model,
            'scaler': scaler,
            'feature_names': feature_names
        }
    except Exception as e:
        st.error(f"Erreur de chargement des modèles: {str(e)}")
        return None
    
with st.sidebar:
    # Charger le fichier CSS
    def load_css(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
# Appliquer le CSS
load_css("./css/pre.css")

# ---- 2. Interface Utilisateur ----
def show_input_form(feature_names):
    tab1, tab2, tab3 = st.tabs(["📋 Prédiction", "📊 Analyse des Données", "ℹ️ Aide & Informations"])
    with tab1:
        
      with st.form("input_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Informations démographiques")
            age = st.slider("Âge", 1, 120, 50)
            sex = st.radio("Sexe", ["Femme", "Homme"], index=1)
            
            st.subheader("Paramètres cliniques")
            trestbps = st.slider("Pression artérielle (mm Hg)", 90, 200, 120)
            chol = st.slider("Cholestérol (mg/dl)", 100, 600, 200)
        
        with col2:
            st.subheader("Symptômes et tests")
            cp_options = {
                0: "Typique angine",
                1: "Douleur atypique", 
                2: "Douleur non-angineuse",
                3: "Asymptomatique"
            }
            cp = st.selectbox("Type de douleur thoracique", options=cp_options.keys(), 
                            format_func=lambda x: cp_options[x])
            
            fbs = st.checkbox("Glycémie à jeun > 120 mg/dl")
            exang = st.checkbox("Angine induite par l'effort")
            oldpeak = st.slider("Dépression ST (mm)", 0.0, 10.0, 1.0, 0.1)
        
        # Deuxième ligne
        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Résultats ECG")
            restecg_options = {
                0: "Normal", 
                1: "Anomalie onde ST-T",
                2: "Hypertrophie ventriculaire"
            }
            restecg = st.selectbox("ECG au repos", options=restecg_options.keys(),
                                 format_func=lambda x: restecg_options[x])
        
        with col4:
            st.subheader("Paramètres avancés")
            slope_options = {
                0: "Descendante",
                1: "Plate",
                2: "Ascendante"
            }
            slope = st.selectbox("Pente du segment ST", options=slope_options.keys(),
                               format_func=lambda x: slope_options[x])
        
        submitted = st.form_submit_button("Analyser le risque cardiaque")
            # if st.button("Analyser le risque cardiaque", use_container_width=True):
        
        if submitted:
            # Préparation des données
            input_data = {
                'age': age,
                'sex': 1 if sex == "Homme" else 0,
                'chest pain type': cp,
                'resting bp s': trestbps,
                'cholesterol': chol,
                'fasting blood sugar': 1 if fbs else 0,
                'resting ecg': restecg,
                'exercise angina': 1 if exang else 0,
                'oldpeak': oldpeak,
                'ST slope': slope
            }
            return pd.DataFrame([input_data], columns=feature_names)
        
        
        
    with tab2:
        st.subheader("Analyse exploratoire des données")
        data = pd.read_csv("./data/clean_data.csv")
        # Statistiques descriptives
        st.markdown("### 📈 Statistiques descriptives")
        st.dataframe(data.describe())
        
        # Distribution des variables clés
        st.markdown("### 📊 Distribution des variables clés")
        
        col1, col2 = st.columns(2)
        with col1:
            fig = plt.figure(figsize=(8, 4))
            sns.histplot(data['age'], bins=20, kde=True)
            plt.title('Distribution des âges')
            st.pyplot(fig)
            
        with col2:
            fig = plt.figure(figsize=(8, 4))
            sns.countplot(x='sex', data=data)
            plt.title('Répartition par sexe (0=Femme, 1=Homme)')
            st.pyplot(fig)
        
        # Corrélations
        st.markdown("### 🔗 Matrice de corrélation")
        fig = plt.figure(figsize=(10, 8))
        sns.heatmap(data.corr(), annot=True, fmt=".1f", cmap="coolwarm")
        plt.title('Corrélation entre les variables')
        st.pyplot(fig)
        
        # Analyse par groupe de risque
        st.markdown("### ⚠️ Analyse par groupe de risque")
        fig = plt.figure(figsize=(12, 6))
        
        plt.subplot(2, 2, 1)
        sns.boxplot(x='target', y='age', data=data)
        plt.title('Âge par groupe de risque')
        
        plt.subplot(2, 2, 2)
        sns.boxplot(x='target', y='resting bp s', data=data)
        plt.title('Pression artérielle par groupe de risque')
        
        plt.subplot(2, 2, 3)
        sns.boxplot(x='target', y='cholesterol', data=data)
        plt.title('Cholestérol par groupe de risque')
        
        plt.subplot(2, 2, 4)
        sns.boxplot(x='target', y='oldpeak', data=data)
        plt.title('Dépression ST par groupe de risque')
        
        plt.tight_layout()
        st.pyplot(fig)
        
    with tab3:
        st.subheader("Guide d'utilisation")
        st.markdown("""
        **Comment utiliser l'application ?**
        1. Remplissez tous les champs du formulaire de prédiction
        2. Cliquez sur le bouton "Analyser le risque cardiaque"
        3. Consultez les résultats et recommandations personnalisées
        4. Explorez les données dans l'onglet d'analyse
        """)
        
        st.markdown("---")
        st.subheader("Explications des paramètres")
        
        with st.expander("Détails des variables médicales"):
            st.markdown("""
            - **Âge**: Âge du patient en années
            - **Sexe**: 0 = Femme, 1 = Homme
            - **Type de douleur thoracique**: 
              - 0: Angine typique
              - 1: Douleur atypique
              - 2: Douleur non-angineuse
              - 3: Asymptomatique
            - **Pression artérielle**: Pression au repos en mm Hg
            - **Cholestérol**: Niveau en mg/dl
            - **Glycémie à jeun**: 0 = <120 mg/dl, 1 = >120 mg/dl
            - **Résultat ECG**: 
              - 0: Normal
              - 1: Anomalie onde ST-T
              - 2: Hypertrophie ventriculaire gauche
            - **Angine induite**: 0 = Non, 1 = Oui
            - **Dépression ST**: Mesure en mm
            - **Pente ST**: 
              - 0: Descendante
              - 1: Plate
              - 2: Ascendante
            """)
    
        st.markdown("---")
        st.subheader("À propos")
        st.markdown("""
         Notre application permet de prédire le risque cardiaque basée sur une apprentissage automatique.
        
        Notre modèle a été entraîné sur un jeu de données de patients avec différentes caractéristiques cardiovasculaires.
        
        ⚠️ **Note importante:** Les résultats sont des estimations statistiques et ne remplacent pas un diagnostic médical professionnel.
        """)           
            
    return None
    

        
        
        
# ---- 3. Affichage des résultats ----
def display_results(prediction, proba, model_type):
    """Affiche les résultats de prédiction"""
    st.markdown("---")
    st.subheader(f"📊 Résultats ({model_type})")
    
    # Jauge de risque
    risk_level = proba[1] * 100
    if risk_level < 30:
        risk_color = "#28a745"
        risk_label = "Faible"
    elif risk_level < 70:
        risk_color = "#ffc107"
        risk_label = "Modéré"
    else:
        risk_color = "#dc3545"
        risk_label = "Élevé"
    
    # Layout des résultats
    col_res1, col_res2 = st.columns([1, 2])
    
    with col_res1:
        st.markdown(f"""
        <div style="text-align: center;">
            <h2 style="color: {risk_color};">{risk_label}</h2>
            <h3>{risk_level:.1f}% de risque</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Graphique de probabilité
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.barh(["Risque"], [risk_level], color=risk_color, height=0.3)
        ax.set_xlim(0, 100)
        ax.set_title('Niveau de risque cardiaque')
        st.pyplot(fig)
    
    with col_res2:
        # Conclusion
        if prediction[0] == 1:
            st.error("**Conclusion:** Risque élevé de maladie cardiaque détecté")
        else:
            st.success("**Conclusion:** Risque faible de maladie cardiaque détecté")
        
        # Graphique détaillé
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x=['Sain', 'À risque'], y=proba, 
                    palette=['#28a745', '#dc3545'], ax=ax)
        ax.set_ylabel('Probabilité')
        ax.set_title('Distribution des probabilités')
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.1%}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', 
                        xytext=(0, 10), 
                        textcoords='offset points')
        st.pyplot(fig)
    
    # Recommandations
    show_recommendations(prediction)

def show_recommendations(prediction):
    """Affiche les recommandations personnalisées"""
    st.markdown("---")
    st.subheader("📌 Recommandations personnalisées")
    
    if prediction[0] == 1:
        cols = st.columns(2)
        with cols[0]:
            st.markdown("""
            **🏥 Consultation urgente**
            - Rendez-vous cardiologue sous 7 jours
            - Bilan complet: ECG + échocardiogramme
            - Surveillance tensionnelle quotidienne
            """)
        with cols[1]:
            st.markdown("""
            **💊 Traitement**
            - Évaluation médicamenteuse
            - Bilan lipidique approfondi
            - Test d'effort supervisé
            """)
    else:
        st.markdown("""
        **🩺 Prévention active**
        - Contrôles annuels après 40 ans
        - Activité physique 150 min/semaine
        - Régime méditerranéen recommandé
        """)

# ---- 4. Fonction principale ----
def main():
    # Chargement des modèles
    models = load_models()
    if not models:
        return
    
    # Titre de l'application
    # st.title("🔍 Analyse de risque cardiaque - Multi-Modèles")
    st.markdown('<div class="header"><h1>Prédiction cardiaque</h1></div>', unsafe_allow_html=True)
    
    # Sélecteur de modèle
    model_type = st.radio(
        "Sélectionnez le modèle:",
        ("Decision Tree", "Réseau de Neurones (ANN)"),
        horizontal=True,
        index=0
    )
    
    # Formulaire de saisie
    input_df = show_input_form(models['feature_names'])
    
    # Prédiction
    if input_df is not None:
        try:
            if model_type == "Decision Tree":
                prediction = models['dt'].predict(input_df)
                proba = models['dt'].predict_proba(input_df)[0]
            else:
                # Prétraitement pour ANN
                scaled_input = models['scaler'].transform(input_df)
                proba_ann = models['ann'].predict(scaled_input)[0][0]
                proba = [1 - proba_ann, proba_ann]
                prediction = [1] if proba_ann >= 0.5 else [0]
            
            display_results(prediction, proba, model_type)
            
            # Comparaison des modèles (optionnel)
            if st.checkbox("Afficher la comparaison des modèles"):
                compare_models(models, input_df)
                
        except Exception as e:
            st.error(f"Erreur lors de la prédiction: {str(e)}")

def compare_models(models, input_df):
    """Affiche une comparaison des deux modèles"""
    st.markdown("---")
    st.subheader("🆚 Comparaison des modèles")
    
    # Prédictions
    dt_pred = models['dt'].predict(input_df)[0]
    dt_proba = models['dt'].predict_proba(input_df)[0][1]
    
    scaled_input = models['scaler'].transform(input_df)
    ann_proba = models['ann'].predict(scaled_input)[0][0]
    ann_pred = 1 if ann_proba >= 0.5 else 0
    
    # Métriques
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Decision Tree", 
                 "À risque" if dt_pred == 1 else "Sain",
                 f"{dt_proba*100:.1f}% de confiance")
    
    with col2:
        st.metric("Réseau de Neurones", 
                 "À risque" if ann_pred == 1 else "Sain",
                 f"{ann_proba*100:.1f}% de confiance")
    
    # Graphique comparatif
    fig, ax = plt.subplots(figsize=(10, 4))
    models_data = {
        'Decision Tree': dt_proba,
        'Réseau Neuronal': ann_proba
    }
    sns.barplot(x=list(models_data.keys()), y=list(models_data.values()),
                palette="viridis", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Probabilité de risque')
    ax.set_title('Comparaison des prédictions')
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}', 
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='center', xytext=(0, 10), 
                   textcoords='offset points')
    st.pyplot(fig)

if __name__ == "__main__":
    main()