import streamlit as st
import asyncio
from utils_combined import get_gemini_response, get_ollama_response, is_valid_image,get_surya_ocr_response,get_azure_vision_response
import logging
import time
from PIL import Image
import os
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Initialize Streamlit app
    st.set_page_config(page_title="Image Analysis")
    st.header("Image Analysis Application")

    # Model selection
    model_choice = st.selectbox(
        "Select Model:",
        ["Gemini", "Open Source", "Surya OCR","Azure OpenAI"]
    )

    # User input
    user_input = st.text_input(
        "What would you like to know about your image?",
        placeholder="Example: This is a mortality table. Please extract the age groups and their corresponding exposed to risk values."
    )

    # Image upload
    uploaded_file = st.file_uploader("Upload your image", type=["jpg", "jpeg", "png"])

    # Display image if uploaded
    if uploaded_file is not None:
        try:
            if is_valid_image(uploaded_file):
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
            else:
                st.error("Please upload a valid image file.")
        except Exception as e:
            st.error(f"Error loading image: {str(e)}")
            logger.error(f"Image loading error: {e}")

    # Process button
    if st.button("Analyze Image"):
        if uploaded_file is None:
            st.error("Please upload an image first.")
        elif not user_input:
            st.error("Please provide a question or description.")
        else:
            with st.spinner("Processing image..."):
                try:
                    start_time = time.time()
                    
                    if model_choice == "Azure Vision":
                        # Check for Azure credentials
                        if not all([os.getenv("AZURE_OPENAI_API_KEY"),
                                  os.getenv("AZURE_OPENAI_ENDPOINT"),
                                  os.getenv("AZURE_OPENAI_API_VERSION"),
                                  os.getenv("AZURE_OPENAI_GPT4V_DEPLOYMENT_NAME")]):
                            st.error("Azure OpenAI credentials not found. Please check your .env file.")
                            return
                        result = asyncio.run(get_azure_vision_response(
                            uploaded_file.getvalue(),
                            user_input
                        ))
                        response = result.get("response", "Error processing image")
                    


                    elif model_choice == "Gemini":
                        # Process with Gemini
                        if not os.getenv("GOOGLE_API_KEY"):
                            st.error("Google API Key not found. Please check your .env file.")
                            return
                        response = get_gemini_response(user_input, image)
                    elif model_choice == "Surya OCR":
                        # New Surya OCR logic
                        if not os.getenv("SURYA_API_KEY"):
                            st.error("Surya API Key not found. Please check your .env file.")
                            return
                        result = asyncio.run(get_surya_ocr_response(
                            uploaded_file.getvalue(),
                            user_input
                        ))
                        response = result.get("response", "Error processing image")
                    else:
                        # Process with Ollama
                        result = asyncio.run(get_ollama_response(
                            uploaded_file.getvalue(),
                            user_input
                        ))
                        response = result.get("response", "Error processing image")
                    
                    # Calculate processing time
                    processing_time = (time.time() - start_time) / 60
                    
                    # Display results
                    st.success(f"Analysis completed in {processing_time:.2f} minutes!")
                    st.subheader("Results:")
                    st.write(response)
                    
                    # Log completion
                    logger.info(f"Processing completed in {processing_time:.2f} minutes")
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    logger.error(f"Processing error: {e}", exc_info=True)

    # Debug information (only visible in development)
    if st.checkbox("Show Debug Information"):
        st.write("Debug Information:")
        if uploaded_file:
            st.write(f"File type: {uploaded_file.type}")
            st.write(f"Model selected: {model_choice}")
            st.write(f"User input: {user_input}")

    logger.info("App rendering completed")

if __name__ == "__main__":
    main()
