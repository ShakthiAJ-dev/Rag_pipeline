# RAG Pipeline

This project is a **Retrieval-Augmented Generation (RAG)** pipeline designed to enhance large language models (LLMs) with relevant, real-time information. Built using **LangChain** and a sleek **Streamlit** UI, it enables users to upload documents, build contextual understanding, and interact with LLMs using accurate and dynamic responses.

## ✨ Features
- 📥 **Document Upload**: Seamlessly upload PDFs and text files.
- 🧩 **Context Building**: Create embeddings for text chunks using LangChain.
- 🔎 **Efficient Retrieval**: Perform quick and accurate semantic searches with FAISS.
- 🧠 **LLM Integration**: Generate context-aware answers using OpenAI's powerful models.
- 🌿 **Streamlit UI**: A user-friendly web interface for easy interaction.

## 🚀 Getting Started
Follow these steps to set up and run the application:

### Prerequisites
- Python installed (>=3.8)
- An OpenAI API Key

### Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/ShakthiAJ-dev/Rag_pipeline.git
    cd Rag_pipeline
    ```
2. Create a `.env` file and add your OpenAI API key:
    ```env
    OPENAI_API_KEY=your_openai_api_key
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Run the App
Start the Streamlit app using the following command:
```bash
streamlit run app.py
```

## 🧑‍💻 Usage
1. Upload your documents via the Streamlit interface.
2. Ask questions related to your documents.
3. The RAG pipeline will fetch relevant information and provide detailed responses.

## 📧 Contact
For any questions, feel free to reach out at ajshakthivelu@gmail.com

## 🛠️ Contributing
Contributions are welcome! Feel free to submit issues or pull requests to enhance the project.

---

**Enjoy using the RAG Pipeline! 🚀**

