import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Configuration de la page
st.set_page_config(
    page_title="Prédiction de risque de maladie cardiaque",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar de navigation
with st.sidebar:
    # Charger le fichier CSS
    def load_css(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
# Appliquer le CSS
    load_css("./css/pre.css")

# Chargement des artefacts
@st.cache_resource
def load_artifacts(job: str, feature: str):
    try:
        model = joblib.load(job)
        feature_names = joblib.load(feature)
        return model, feature_names
    except FileNotFoundError as e:
        st.error(f"Erreur de chargement: {e}. Veuillez d'abord exécuter les notebooks Jupyter")
        return None, None

@st.cache_data
def load_data():
    data = pd.read_csv("./csv/cleaned_dataset.csv")
    return data

# Chargement des données
dataset = load_data()
model, feature_names = load_artifacts(
    './params/joblib/heart_model_pipeline_decision_tree.joblib', 
    './params/joblib/feature_names_decision_tree.joblib'
)


# En-tête de l'application
st.markdown('<div class="header"><h1>Prédiction médicale de risque de maladie cardiaque</h1></div>', unsafe_allow_html=True)

# Onglets
tab1, tab2, tab3 = st.tabs(["📋 Prédiction", "📊 Analyse des Données", "ℹ️ Aide & Informations"])

with tab1:
    # Section de saisie organisée en colonnes
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Informations démographiques")
        age = st.slider("Âge", min_value=1, max_value=120, value=50, 
                       help="Âge du patient en années")
        sex = st.selectbox("Sexe", options=[0, 1], 
                          format_func=lambda x: "Femme" if x == 0 else "Homme")
        
        st.subheader("Paramètres cliniques")
        trestbps = st.slider("Pression artérielle au repos (mm Hg)", 
                            min_value=90, max_value=200, value=120)
        chol = st.slider("Cholestérol (mg/dl)", 
                        min_value=100, max_value=600, value=200)
        
    with col2:
        st.subheader("Symptômes et tests")
        cp = st.selectbox("Type de douleur thoracique", options=[0, 1, 2, 3],
                         help="0: Typique angine\n1: Douleur atypique\n2: Douleur non-angineuse\n3: Asymptomatique")
        fbs = st.selectbox("Glycémie à jeun > 120 mg/dl", options=[0, 1],
                          help="0: Non\n1: Oui")
        exang = st.selectbox("Angine induite par l'effort", options=[0, 1],
                           help="0: Non\n1: Oui")
        oldpeak = st.slider("Dépression ST induite (mm)", 
                           min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    
    # Deuxième ligne de colonnes
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Résultats ECG")
        restecg = st.selectbox("Résultat ECG au repos", options=[0, 1, 2],
                             help="0: Normal\n1: Anomalie onde ST-T\n2: Hypertrophie ventriculaire gauche")
    with col4:
        st.subheader("Paramètres avancés")
        slope = st.selectbox("Pente du segment ST", options=[0, 1, 2],
                           help="0: Pente descendante\n1: Plat\n2: Pente ascendante")
    
    # Bouton de prédiction
    if st.button("Analyser le risque cardiaque", use_container_width=True):
        # Création du dataframe d'entrée
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
        
        input_df = pd.DataFrame([input_dict], columns=feature_names)
        
        # Prédiction
        try:
            prediction = model.predict(input_df)
            proba = model.predict_proba(input_df)[0]
            
            # Affichage des résultats
            st.markdown("---")
            st.subheader("📊 Résultats de l'analyse")
            
            # Jauge de risque visuelle
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
                ax.set_xticks([0, 25, 50, 75, 100])
                ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
                ax.set_title('Niveau de risque cardiaque')
                st.pyplot(fig)
                
            with col_res2:
                if prediction[0] == 1:
                    st.error("**Conclusion:** Risque élevé de maladie cardiaque détecté")
                else:
                    st.success("**Conclusion:** Risque faible de maladie cardiaque détecté")
                
                # Graphique détaillé des probabilités
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.barplot(x=['Sain', 'À risque'], y=proba, 
                            palette=['#28a745', '#dc3545'], ax=ax)
                ax.set_ylabel('Probabilité')
                ax.set_title('Probabilité de maladie cardiaque')
                for p in ax.patches:
                    ax.annotate(f'{p.get_height():.1%}', 
                                (p.get_x() + p.get_width() / 2., p.get_height()), 
                                ha='center', va='center', 
                                xytext=(0, 10), 
                                textcoords='offset points')
                st.pyplot(fig)
            
            # Recommandations personnalisées
            st.markdown("---")
            st.subheader("📌 Recommandations personnalisées")
            
            if prediction[0] == 1:
                rec_col1, rec_col2 = st.columns(2)
                
                with rec_col1:
                    st.markdown("""
                    **🏥 Consultation médicale urgente**
                    - Prendre rendez-vous avec un cardiologue dans les plus brefs délais
                    - Faire un bilan cardiaque complet (ECG, échocardiogramme)
                    - Surveillance régulière de la pression artérielle
                    """)
                    
                with rec_col2:
                    st.markdown("""
                    **💊 Gestion médicale**
                    - Évaluation des facteurs de risque modifiables
                    - Bilan lipidique complet
                    - Possible traitement médicamenteux selon avis médical
                    """)
                
                st.markdown("""
                **🏃 Mode de vie à adopter**
                - Arrêt immédiat du tabac si applicable
                - Réduction de la consommation d'alcool
                - Activité physique modérée et régulière (30 min/jour)
                - Alimentation méditerranéenne (riche en fruits, légumes, poissons)
                """)
            else:
                st.markdown("""
                **🩺 Conseils de prévention**
                - Maintenir un mode de vie sain (alimentation équilibrée, activité physique)
                - Contrôles médicaux réguliers (tous les 2 ans après 40 ans)
                - Surveillance des facteurs de risque (pression artérielle, cholestérol)
                """)
                
                st.markdown("""
                **🍏 Prévention active**
                - Consommation quotidienne de fruits et légumes
                - Limitation des graisses saturées et du sel
                - Activité physique régulière (150 min/semaine)
                - Gestion du stress (méditation, yoga)
                """)
            
            # Analyse des facteurs influents
            st.markdown("---")
            st.subheader("🔍 Facteurs influents dans la prédiction")
            
            # Simulation de l'importance des features (à adapter avec votre modèle)
            feature_importance = {
                'chest pain type': 0.25,
                'oldpeak': 0.20,
                'ST slope': 0.18,
                'age': 0.15,
                'resting bp s': 0.12,
                'cholesterol': 0.10
            }
            
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(x=list(feature_importance.values()), 
                        y=list(feature_importance.keys()), 
                        palette="Reds_r", ax=ax)
            ax.set_title('Importance des facteurs dans la prédiction')
            ax.set_xlabel('Importance relative')
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Erreur lors de la prédiction: {e}")

with tab2:
    st.subheader("Analyse exploratoire des données")
    
    # Statistiques descriptives
    st.markdown("### 📈 Statistiques descriptives")
    st.dataframe(dataset.describe())
    
    # Distribution des variables clés
    st.markdown("### 📊 Distribution des variables clés")
    
    col1, col2 = st.columns(2)
    with col1:
        fig = plt.figure(figsize=(8, 4))
        sns.histplot(dataset['age'], bins=20, kde=True)
        plt.title('Distribution des âges')
        st.pyplot(fig)
        
    with col2:
        fig = plt.figure(figsize=(8, 4))
        sns.countplot(x='sex', data=dataset)
        plt.title('Répartition par sexe (0=Femme, 1=Homme)')
        st.pyplot(fig)
    
    # Corrélations
    st.markdown("### 🔗 Matrice de corrélation")
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(dataset.corr(), annot=True, fmt=".1f", cmap="coolwarm")
    plt.title('Corrélation entre les variables')
    st.pyplot(fig)
    
    # Analyse par groupe de risque
    st.markdown("### ⚠️ Analyse par groupe de risque")
    fig = plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 2, 1)
    sns.boxplot(x='target', y='age', data=dataset)
    plt.title('Âge par groupe de risque')
    
    plt.subplot(2, 2, 2)
    sns.boxplot(x='target', y='resting bp s', data=dataset)
    plt.title('Pression artérielle par groupe de risque')
    
    plt.subplot(2, 2, 3)
    sns.boxplot(x='target', y='cholesterol', data=dataset)
    plt.title('Cholestérol par groupe de risque')
    
    plt.subplot(2, 2, 4)
    sns.boxplot(x='target', y='oldpeak', data=dataset)
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