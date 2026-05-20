# Import the necessary packages
from openai import OpenAI
from dotenv import load_dotenv
import os
import gradio as gr

load_dotenv()

# Connect to OpenAI
client = OpenAI(
    api_key=os.environ["API_KEY"],
)

model_id = "gpt-4o-mini"  # Low cost — swap for gpt-4o if needed

# Get the query from the user input

# Generate and print the response
def generate_response(query):
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": query}],
        max_tokens=256,
        temperature=0.5,
    )
    return response.choices[0].message.content



chat_application = gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(label="Input", lines=2, placeholder="Type your question here..."),
    outputs=gr.Textbox(label="Output"),
    title="Chatbot",
    description="Ask any question and the chatbot will try to answer."
)

# Launch the app
chat_application.launch(server_name="127.0.0.1", server_port= 7860)