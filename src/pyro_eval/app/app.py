import streamlit as st
import pandas as pd
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import base64
from io import BytesIO

class DataLoader:
    """Classe pour charger et gérer les données d'évaluation"""
    
    def __init__(self, data_path: str = "data"):
        self.data_path = Path(data_path)
        self.metrics_data = {}
        self.configs_data = {}
        self.diff_images_data = {}
    
    def load_metrics(self, metrics_file: str) -> pd.DataFrame:
        """Charge les métriques depuis un fichier JSON ou CSV"""
        file_path = self.data_path / metrics_file
        
        if file_path.suffix == '.json':
            with open(file_path, 'r') as f:
                data = json.load(f)
                return pd.DataFrame(data)
        elif file_path.suffix == '.csv':
            return pd.read_csv(file_path)
        else:
            raise ValueError(f"Format de fichier non supporté: {file_path.suffix}")
    
    def load_configs(self, configs_file: str) -> Dict:
        """Charge les configurations des modèles"""
        file_path = self.data_path / configs_file
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def load_diff_images(self, diff_images_file: str) -> Dict:
        """Charge la liste des images avec prédictions différentes"""
        file_path = self.data_path / diff_images_file
        with open(file_path, 'r') as f:
            return json.load(f)

class MetricsVisualizer:
    """Classe pour visualiser les métriques de performance"""
    
    def __init__(self):
        self.metrics_df = None
    
    def display_metrics_table(self, metrics_df: pd.DataFrame):
        """Affiche un tableau comparatif des métriques"""
        st.subheader("📊 Tableau comparatif des métriques")
        
        # Options de filtrage
        col1, col2 = st.columns(2)
        with col1:
            models_to_show = st.multiselect(
                "Sélectionner les modèles à afficher",
                options=metrics_df.index.tolist() if 'model_name' not in metrics_df.columns else metrics_df['model_name'].unique(),
                default=metrics_df.index.tolist()[:3] if len(metrics_df) > 0 else []
            )
        
        with col2:
            metrics_to_show = st.multiselect(
                "Sélectionner les métriques à afficher",
                options=[col for col in metrics_df.columns if col != 'model_name'],
                default=[col for col in metrics_df.columns if col != 'model_name'][:5]
            )
        
        # Filtrage des données
        if models_to_show and metrics_to_show:
            if 'model_name' in metrics_df.columns:
                filtered_df = metrics_df[metrics_df['model_name'].isin(models_to_show)]
                display_cols = ['model_name'] + metrics_to_show
            else:
                filtered_df = metrics_df.loc[models_to_show]
                display_cols = metrics_to_show
            
            # Affichage du tableau avec mise en forme
            st.dataframe(
                filtered_df[display_cols].round(4),
                use_container_width=True,
                height=400
            )
            
            # Graphique de comparaison
            self.plot_metrics_comparison(filtered_df, metrics_to_show)
    
    def plot_metrics_comparison(self, df: pd.DataFrame, metrics: List[str]):
        """Crée un graphique de comparaison des métriques"""
        if len(metrics) == 0:
            return

        st.subheader("📈 Comparaison visuelle des métriques")
        
        # Sélection du type de graphique
        chart_type = st.selectbox("Type de graphique", ["Barres", "Radar", "Ligne"])
        
        if chart_type == "Barres":
            # Graphique en barres
            if 'model_name' in df.columns:
                df_melted = df.melt(id_vars=['model_name'], value_vars=metrics, 
                                  var_name='metric', value_name='value')
                fig = px.bar(df_melted, x='model_name', y='value', color='metric',
                           title="Comparaison des métriques par modèle", barmode='group')
            else:
                df_melted = df[metrics].reset_index().melt(id_vars=['index'], 
                                                         var_name='metric', value_name='value')
                fig = px.bar(df_melted, x='index', y='value', color='metric',
                           title="Comparaison des métriques par modèle", barmode='group')
        
        elif chart_type == "Radar":
            # Graphique radar
            fig = go.Figure()
            
            for idx, row in df.iterrows():
                model_name = row.get('model_name', idx)
                values = [row[metric] for metric in metrics]
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=metrics,
                    fill='toself',
                    name=str(model_name)
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1])
                ),
                showlegend=True,
                title="Comparaison radar des métriques"
            )
        
        else:  # Ligne
            if 'model_name' in df.columns:
                df_melted = df.melt(id_vars=['model_name'], value_vars=metrics,
                                  var_name='metric', value_name='value')
                fig = px.line(df_melted, x='metric', y='value', color='model_name',
                            title="Évolution des métriques par modèle", markers=True)
            else:
                fig = go.Figure()
                for idx, row in df.iterrows():
                    values = [row[metric] for metric in metrics]
                    fig.add_trace(go.Scatter(x=metrics, y=values, mode='lines+markers',
                                           name=str(idx)))
                fig.update_layout(title="Évolution des métriques par modèle")
        
        st.plotly_chart(fig, use_container_width=True)

class ConfigViewer:
    """Classe pour afficher les configurations des modèles"""
    
    def display_configs(self, configs: Dict):
        """Affiche les configurations des modèles"""
        st.subheader("⚙️ Configurations des modèles")
        
        # Sélection du modèle
        model_names = list(configs.keys())
        selected_model = st.selectbox("Sélectionner un modèle", model_names)
        
        if selected_model and selected_model in configs:
            config = configs[selected_model]
            
            # Affichage en colonnes pour une meilleure lisibilité
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Paramètres principaux:**")
                main_params = {k: v for k, v in config.items() 
                             if k in ['model_type', 'learning_rate', 'batch_size', 'epochs']}
                if main_params:
                    st.json(main_params)
            
            with col2:
                st.write("**Autres paramètres:**")
                other_params = {k: v for k, v in config.items() 
                              if k not in ['model_type', 'learning_rate', 'batch_size', 'epochs']}
                if other_params:
                    st.json(other_params)
            
            # Configuration complète en format JSON
            with st.expander("Voir la configuration complète"):
                st.json(config)

class ModelEvaluator:
    """Classe principale pour l'application d'évaluation des modèles"""
    
    def __init__(self):
        self.data_loader = DataLoader()
        self.metrics_visualizer = MetricsVisualizer()
        self.config_viewer = ConfigViewer()
        self.image_analyzer = ImageAnalyzer()
        
        # Configuration de la page
        st.set_page_config(
            page_title="Évaluation de Modèles de Détection",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def run(self):
        """Lance l'application principale"""
        st.title("🤖 Évaluation de Modèles de Détection")
        st.markdown("---")
        
        # Sidebar pour la navigation
        st.sidebar.title("Navigation")
        
        # Chargement des données
        with st.sidebar.expander("📁 Chargement des données", expanded=True):
            metrics_file = st.text_input("Fichier métriques", "metrics.json")
            configs_file = st.text_input("Fichier configurations", "configs.json")
            diff_images_file = st.text_input("Fichier images différentes", "diff_images.json")
            images_path = st.text_input("Dossier images", "images")
            
            if st.button("Charger les données"):
                self.load_data(metrics_file, configs_file, diff_images_file, images_path)
        
        # Navigation principale
        page = st.sidebar.selectbox(
            "Sélectionner une page",
            ["📊 Métriques", "⚙️ Configurations", "🔍 Images différentes", "📋 Rapport complet"]
        )
        
        # Affichage conditionnel des pages
        if hasattr(self, 'metrics_df'):
            if page == "📊 Métriques":
                self.metrics_visualizer.display_metrics_table(self.metrics_df)
            
            elif page == "⚙️ Configurations":
                if hasattr(self, 'configs'):
                    self.config_viewer.display_configs(self.configs)
                else:
                    st.warning("Configurations non chargées")
            
            elif page == "🔍 Images différentes":
                if hasattr(self, 'diff_images'):
                    self.image_analyzer.display_diff_images(self.diff_images)
                else:
                    st.warning("Données d'images différentes non chargées")
            
            elif page == "📋 Rapport complet":
                self.display_full_report()
        else:
            st.info("Veuillez charger les données dans la barre latérale pour commencer.")
    
    def load_data(self, metrics_file: str, configs_file: str, diff_images_file: str, images_path: str):
        """Charge toutes les données nécessaires"""
        try:
            with st.spinner("Chargement des données..."):
                # Chargement des métriques
                if metrics_file:
                    self.metrics_df = self.data_loader.load_metrics(metrics_file)
                    st.success(f"✅ Métriques chargées: {len(self.metrics_df)} modèles")
                
                # Chargement des configurations
                if configs_file:
                    self.configs = self.data_loader.load_configs(configs_file)
                    st.success(f"✅ Configurations chargées: {len(self.configs)} modèles")
                
                # Chargement des images différentes
                if diff_images_file:
                    self.diff_images = self.data_loader.load_diff_images(diff_images_file)
                    st.success(f"✅ Images différentes chargées: {len(self.diff_images)} comparaisons")
                
                # Configuration du chemin des images
                if images_path:
                    self.image_analyzer.images_base_path = Path(images_path)
                    st.success(f"✅ Chemin des images configuré: {images_path}")
                
        except Exception as e:
            st.error(f"Erreur lors du chargement: {str(e)}")
    
    def display_full_report(self):
        """Affiche un rapport complet avec toutes les informations"""
        st.subheader("📋 Rapport complet d'évaluation")
        
        # Résumé exécutif
        st.markdown("### 📈 Résumé exécutif")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if hasattr(self, 'metrics_df'):
                st.metric("Modèles évalués", len(self.metrics_df))
        
        with col2:
            if hasattr(self, 'configs'):
                st.metric("Configurations", len(self.configs))
        
        with col3:
            if hasattr(self, 'diff_images'):
                total_diff_images = sum(len(images) for images in self.diff_images.values())
                st.metric("Images avec différences", total_diff_images)
        
        # Meilleurs modèles
        if hasattr(self, 'metrics_df'):
            st.markdown("### 🏆 Top modèles par métrique")
            numeric_cols = self.metrics_df.select_dtypes(include=['float64', 'int64']).columns
            
            if len(numeric_cols) > 0:
                selected_metric = st.selectbox("Métrique pour le classement", numeric_cols)
                top_models = self.metrics_df.nlargest(3, selected_metric)
                st.dataframe(top_models, use_container_width=True)

# Point d'entrée de l'application
if __name__ == "__main__":
    app = ModelEvaluator()
    app.run()