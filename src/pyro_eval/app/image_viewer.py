import base64
from pathlib import Path
from typing import Dict
import streamlit as st
from PIL import Image

from run_data import RunData

class ImageAnalyzer:
    """Class used to compare images on different runs"""
    
    def __init__(self, runs: List[Run]):
        self.images_base_path = Path(images_base_path)
    
    def display_diff_images(self, diff_images: Dict):
        """Affiche la liste des images avec prédictions différentes"""
        st.subheader("🔍 Images avec prédictions différentes")
        
        # Statistiques globales
        total_images = sum(len(images) for images in diff_images.values())
        st.metric("Nombre total d'images avec différences", total_images)
        
        # Sélection de la comparaison
        comparison_keys = list(diff_images.keys())
        selected_comparison = st.selectbox(
            "Sélectionner la comparaison", 
            comparison_keys,
            help="Format attendu: 'model1_vs_model2'"
        )
        
        if selected_comparison and selected_comparison in diff_images:
            images_list = diff_images[selected_comparison]
            
            st.write(f"**{len(images_list)} images** avec des prédictions différentes")
            
            # Options d'affichage
            col1, col2, col3 = st.columns(3)
            with col1:
                items_per_page = st.selectbox("Images par page", [10, 25, 50, 100], index=1)
            with col2:
                search_term = st.text_input("Rechercher une image", "")
            with col3:
                show_thumbnails = st.checkbox("Afficher les miniatures", value=False)
            
            # Filtrage par recherche
            if search_term:
                filtered_images = [img for img in images_list if search_term.lower() in img.lower()]
            else:
                filtered_images = images_list
            
            # Pagination
            total_pages = (len(filtered_images) - 1) // items_per_page + 1
            page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
            
            start_idx = (page - 1) * items_per_page
            end_idx = min(start_idx + items_per_page, len(filtered_images))
            current_images = filtered_images[start_idx:end_idx]
            
            # Affichage des images
            st.write(f"Affichage des images {start_idx + 1} à {end_idx} sur {len(filtered_images)}")
            
            if show_thumbnails:
                self.display_image_grid(current_images)
            else:
                self.display_image_list(current_images)
    
    def display_image_list(self, images_list: List[str]):
        """Affiche une liste cliquable d'images"""
        for i, image_name in enumerate(images_list):
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.write(f"**{i+1}.** {image_name}")
            
            with col2:
                if st.button(f"Voir", key=f"view_{i}_{image_name}"):
                    self.display_single_image(image_name)
            
            with col3:
                image_path = self.images_base_path / image_name
                if image_path.exists():
                    # Lien pour ouvrir dans un nouvel onglet
                    st.markdown(
                        f'<a href="data:image/png;base64,{self.get_image_base64(image_path)}" target="_blank">🔗 Ouvrir</a>',
                        unsafe_allow_html=True
                    )
                else:
                    st.write("❌ Non trouvée")
    
    def display_image_grid(self, images_list: List[str]):
        """Affiche une grille de miniatures"""
        cols = st.columns(4)
        
        for i, image_name in enumerate(images_list):
            with cols[i % 4]:
                image_path = self.images_base_path / image_name
                if image_path.exists():
                    try:
                        image = Image.open(image_path)
                        # Redimensionner pour miniature
                        image.thumbnail((150, 150))
                        st.image(image, caption=image_name, use_column_width=True)
                        
                        if st.button(f"Agrandir", key=f"enlarge_{i}_{image_name}"):
                            self.display_single_image(image_name)
                    except Exception as e:
                        st.error(f"Erreur lors du chargement de {image_name}: {str(e)}")
                else:
                    st.error(f"Image non trouvée: {image_name}")
    
    def display_single_image(self, image_name: str):
        """Affiche une image en grand format"""
        image_path = self.images_base_path / image_name
        
        if image_path.exists():
            try:
                image = Image.open(image_path)
                st.image(image, caption=image_name, use_column_width=True)
            except Exception as e:
                st.error(f"Erreur lors du chargement de {image_name}: {str(e)}")
        else:
            st.error(f"Image non trouvée: {image_name}")
    
    def get_image_base64(self, image_path: Path) -> str:
        """Convertit une image en base64 pour les liens"""
        try:
            with open(image_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode()
        except Exception:
            return ""
