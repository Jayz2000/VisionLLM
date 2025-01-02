from dotenv import load_dotenv
import google.generativeai as genai
import os
import base64
import ollama
from typing import Dict
import asyncio
import logging
from PIL import Image
import requests
import openai
from openai import OpenAI
from openai import AzureOpenAI
# Get Azure credentials from .env file
AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
openai.api_key = AZURE_OPENAI_API_KEY

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Gemini
if os.getenv("GOOGLE_API_KEY"):
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")

def get_gemini_response(user_input: str, image: Image) -> str:
    """Process image using Gemini model."""
    try:
        logger.info("Starting Gemini processing")
        if user_input != "":
            response = gemini_model.generate_content([user_input, image])
        else:
            response = gemini_model.generate_content(image)
        logger.info("Gemini processing completed")
        return response.text
    except Exception as e:
        logger.error(f"Error in Gemini processing: {str(e)}")
        raise

async def encode_image_to_base64(image_bytes: bytes) -> str:
    """Convert image bytes to base64 string."""
    try:
        encoded = base64.b64encode(image_bytes).decode('utf-8')
        logger.debug("Successfully encoded image to base64")
        return encoded
    except Exception as e:
        logger.error(f"Error encoding image to base64: {e}")
        raise


async def get_ollama_response(image_bytes: bytes, user_input: str) -> Dict:
    """Process image using Ollama model."""
    logger.info("Starting Ollama processing")
    try:
        client = ollama.AsyncClient(host='http://localhost:11434')
        logger.info("Successfully created Ollama client")
        
        # Convert image to base64
        base64_image = await encode_image_to_base64(image_bytes)
        logger.info("Image encoded successfully")
        
        # Construct prompt using user context
        prompt = f"Please analyze this image and extract the requested information as user expressed. Do NOT make up information and keep your answer short and concise. Focus on the information of extraction. {user_input}\n\n"
        
        # Prepare message for the model
        message = {
            "role": "user",
            "content": prompt,
            "images": [base64_image]
        }
        
        logger.info("Sending request to Ollama model")
        response = await client.chat(
            # model="llama3.2-vision",
            # model="minicpm-v:latest",
            # model="llava:34b",
            # model="llama3.2-vision:90b",
            model="llama3.2-vision",
                       
            messages=[message],
            options={
                "temperature": 0,
                "top_p": 0.9,
                "num_predict": 1024
            }
        )
        logger.info("Received response from Ollama model")
        
        if response and 'message' in response and 'content' in response['message']:
            content = response['message']['content']
            logger.info("Successfully extracted content from response")
            return {"response": content}
        else:
            logger.error("Unexpected response structure")
            return {"error": "Invalid response structure from model"}
        
    except Exception as e:
        logger.error(f"Error in Ollama processing: {str(e)}")
        return {"error": str(e)}

def is_valid_image(file) -> bool:
    """Check if uploaded file is a valid image."""
    try:
        is_valid = file.type.startswith('image/')
        logger.info(f"Image validation result: {is_valid} for type {file.type}")
        return is_valid
    except Exception as e:
        logger.error(f"Error validating image: {e}")
        return False



async def get_surya_ocr_response(image_bytes: bytes, user_input: str) -> Dict:
    """Process image using Surya OCR API."""
    logger.info("Starting Surya OCR processing")
    try:
        # Convert image bytes to base64
        base64_image = await encode_image_to_base64(image_bytes)
        
        # Prepare the request payload
        payload = {
            "image": base64_image,
            "language": "eng"  # You can make this configurable if needed
        }
        
        # Make request to Surya OCR API
        response = requests.post(
            "https://api.surya-ai.com/v1/ocr",
            headers={
                "Authorization": f"Bearer {os.getenv('SURYA_API_KEY')}",
                "Content-Type": "application/json"
            },
            json=payload
        )
        
        if response.status_code == 200:
            ocr_result = response.json()
            # Format the OCR result with the user's question
            formatted_response = f"Question: {user_input}\n\nExtracted Text:\n{ocr_result['text']}"
            return {"response": formatted_response}
        else:
            logger.error(f"Surya OCR API error: {response.text}")
            return {"error": f"API Error: {response.status_code}"}
            
    except Exception as e:
        logger.error(f"Error in Surya OCR processing: {str(e)}")
        return {"error": str(e)}
    

##% get AzureOpenAI API setup
async def get_azure_vision_response(image_bytes: bytes, user_input: str) -> Dict:
    """Process image using Azure OpenAI Vision API."""
    logger.info("Starting Azure Vision processing")
    try:
        # Initialize Azure OpenAI client
        client = AzureOpenAI(
            api_key=os.getenv('AZURE_OPENAI_API_KEY'),
            azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
            azure_deployment=os.getenv('AZURE_OPENAI_CHAT_DEPLOYMENT_NAME'),  # Replace with your deployment name
            api_version=os.getenv('AZURE_OPENAI_API_VERSION'),  # Use the correct API version
        )

        # Convert bytes to base64
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        # Prepare the messages
        messages = [
            {
                "role": "system",
                "content": "You are an expert at analyzing images and extracting specific information based on user requests."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Please analyze this image and answer the following: {user_input} \n Do not make up content."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]

        # Make the API call
        response = client.chat.completions.create(
            model=os.getenv('AZURE_OPENAI_GPT4V_DEPLOYMENT_NAME'),  # Your GPT-4V deployment name
            messages=messages,
            max_tokens=500,
            temperature=0
        )

        if response.choices:
            content = response.choices[0].message.content
            logger.info("Successfully received Azure Vision response")
            return {"response": content}
        else:
            logger.error("Empty response from Azure Vision")
            return {"error": "No response from model"}

    except Exception as e:
        logger.error(f"Error in Azure Vision processing: {str(e)}")
        return {"error": str(e)}
