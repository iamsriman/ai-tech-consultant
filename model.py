import os
from dotenv import load_dotenv
from groq import Groq

# Load variables from the .env file
load_dotenv()

# This will now successfully fetch the key from your .env file
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

models_list = client.models.list()
for model in models_list.data:
    print(f"• Model ID: {model.id}")
