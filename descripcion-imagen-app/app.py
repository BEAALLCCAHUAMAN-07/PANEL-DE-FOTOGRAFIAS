import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# Configuración de la página
st.set_page_config(page_title="Describir y Escribir", page_icon="📷", layout="wide")

# Título
st.title("📷 Descripción de Imágenes")
st.markdown("Sube una foto para que la IA la describa, o escribe tu propia descripción.")

# Cargar el modelo de IA (solo la primera vez)
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_model()

# Función para generar descripción
def generate_caption(image):
    inputs = processor(image, return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

# Layout en dos columnas
col1, col2 = st.columns(2)

with col1:
    st.subheader("📤 Subir Imagen")
    uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen cargada", use_container_width=True)
        
        # Botón para generar descripción
        if st.button("🤖 Generar Descripción con IA", type="primary"):
            with st.spinner("Analizando imagen..."):
                caption = generate_caption(image)
                st.session_state.generated_caption = caption
                st.success("¡Descripción generada!")

with col2:
    st.subheader("✍️ Descripción")
    
    # Campo para mostrar/editar la descripción generada
    if 'generated_caption' in st.session_state:
        default_text = st.session_state.generated_caption
    else:
        default_text = ""
    
    user_text = st.text_area(
        "Descripción de la imagen (puedes editar la descripción de la IA o escribir la tuya):",
        value=default_text,
        height=200,
        placeholder="La descripción aparecerá aquí después de subir una imagen y hacer clic en 'Generar Descripción con IA'..."
    )
    
    # Botón para guardar
    if st.button("💾 Guardar Descripción"):
        if user_text.strip():
            st.success("Descripción guardada correctamente!")
            st.info(f"Tu descripción: {user_text}")
        else:
            st.warning("Por favor, escribe una descripción antes de guardar.")

# Footer
st.markdown("---")
st.markdown("💡 **Tip:** Puedes editar la descripción generada por la IA o escribir tu propia descripción desde cero.")
st.markdown("Desarrollado por Cristina Beatriz Allccahuaman Lino - Ingeniería Civil UNISC")