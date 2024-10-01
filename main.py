from openai import OpenAI
from dotenv import load_dotenv
import streamlit as st
from typing import List, Dict
import tiktoken
import base64
from io import BytesIO
from PIL import Image

load_dotenv()

client = OpenAI()

# App branding and title
st.set_page_config(page_title="ChatMaster Pro", page_icon="🤖", layout="wide")
st.title("Welcome to ChatMaster Pro")
with st.expander('Instructions'):
    st.markdown("""
### ChatMaster Pro: Your AI Chat Companion

**ChatMaster Pro** is a powerful AI-powered chat application that allows you to engage in natural and informative conversations with advanced language models. 

**Here's how to use it:**

1. **Choose a Model:** Select the desired language model from the sidebar.
2. **Start Chatting:** Type your questions or requests in the chat box.
3. **Upload Images:**  You can upload images to enhance your conversations.
4. **Clear History:** Use the "Clear Chat History" button to start a fresh conversation.

**Enjoy a seamless and engaging chat experience with ChatMaster Pro!**
""")

# Sidebar for model selection and clear chat history button
with st.sidebar:
    st.header("Settings")
    selected_model = st.selectbox(
        "Choose a Model",
        options=["gpt-4o", "gpt-3.5-turbo", ],
        index=0,
    )
    st.session_state["openai_model"] = selected_model

    # Button to clear chat history
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.uploaded_images = {}  # Clear uploaded images as well
        st.success("Chat history cleared!")

    # Image upload widget
    uploaded_files = st.file_uploader(
        "Upload Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_images" not in st.session_state:
    st.session_state.uploaded_images = {}

# Function to convert image to base64
def image_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

# Function to process messages and ensure token limit, ignoring image messages
def process_messages(messages: List[Dict[str, str]], model: str) -> List[Dict[str, str]]:
    encoding = tiktoken.encoding_for_model(model)
    total_tokens = 0

    # Calculate total tokens and remove excess messages if needed
    for message in messages:
        # Ignore messages with image content
        if isinstance(message["content"], list):
            # Check if the message contains image content; if so, skip it
            if any(item["type"] == "image_url" for item in message["content"]):
                continue

        # Only process text content for token count
        content_to_encode = message["content"] if isinstance(
            message["content"], str) else str(message["content"])
        encoded_length = len(encoding.encode(content_to_encode))
        total_tokens += encoded_length

    # Remove oldest non-system messages if token count exceeds the limit
    token_limit = 8000 if model == "gpt-4" else 4000
    while total_tokens > token_limit:
        index_to_remove = next((i for i, msg in enumerate(
            messages) if msg["role"] != "system" and not isinstance(msg["content"], list)), None)
        if index_to_remove is not None:
            removed_content = messages.pop(index_to_remove)["content"]
            total_tokens -= len(encoding.encode(removed_content if isinstance(
                removed_content, str) else str(removed_content)))
        else:
            break

    return messages

# Handle image uploads
if uploaded_files:
    st.session_state.uploaded_images = {}
    for uploaded_file in uploaded_files:
        img = Image.open(uploaded_file)
        img_base64 = image_to_base64(img)
        image_message = {
            "role": "user",
            "content": [
                # {"type": "text", "text": "Here's an image I uploaded:"},
                {
                    "type": "image_url",
                    "image_url": {"url": img_base64},
                },
            ]
        }
        with st.sidebar:
            st.image(img_base64)
        # Add image to message list but don't submit yet
        st.session_state.uploaded_images[uploaded_file.name] = image_message

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if isinstance(message["content"], list):
            for item in message["content"]:
                if item["type"] == "text":
                    st.markdown(item["text"])
                elif item["type"] == "image_url":
                    st.image(item["image_url"]["url"], width=200)
        else:
            st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("Ask anything..."):
    # Add text to messages from user
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Append uploaded images to messages list
    for image_message in st.session_state.uploaded_images.values():
        st.session_state.messages.append(image_message)

    st.session_state.uploaded_images = {}  # Clear uploaded images list

    # Process messages for token limits before sending to the model
    st.session_state.messages = process_messages(
        st.session_state.messages, st.session_state["openai_model"])

    with st.chat_message("assistant"):
        try:
            # Create a progress bar for generating responses
            with st.spinner("Generating response..."):
                stream = client.chat.completions.create(
                    model=st.session_state["openai_model"],
                    messages=[{"role": msg["role"], "content": msg["content"]}
                              for msg in st.session_state.messages],
                    stream=True,
                )
                response = st.write_stream(stream)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )

        except Exception as e:
            st.error(f"Error: {str(e)}")