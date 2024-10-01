# ChatMaster Pro: Your AI Chat Companion

ChatMaster Pro is a powerful AI-powered chat application built with Streamlit that allows you to engage in natural and informative conversations with advanced language models from OpenAI. 

## Features

* **Choose from multiple models:** Select from a variety of OpenAI models, including `gpt-4o` and `gpt-3.5-turbo`.
* **Natural language interaction:**  Engage in conversations that feel like you're talking to a real person.
* **Image uploads:** Enhance your conversations by uploading images.
* **Token limit management:**  The app automatically manages token limits to ensure smooth and efficient conversations.
* **Clear chat history:** Start fresh conversations with a single click.

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/faisal-saddique/chatmaster-pro
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your OpenAI API key:**
   * Create an account on [OpenAI](https://platform.openai.com/account/api-keys).
   * Create a new API key and copy it.
   * Create a `.env` file in the root directory of the project and add the following line:
     ```
     OPENAI_API_KEY=your_api_key
     ```

4. **Run the app:**
   ```bash
   streamlit run main.py
   ```

## Usage

1. **Choose a model:** Select the desired language model from the sidebar.
2. **Start chatting:** Type your questions or requests in the chat box.
3. **Upload images:**  You can upload images to enhance your conversations.
4. **Clear history:** Use the "Clear Chat History" button to start a fresh conversation.