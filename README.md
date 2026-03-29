# Multimodal Agentic RAG System

A state-of-the-art Multimodal Retrieval-Augmented Generation (RAG) system built with **LangChain**, **Groq (Llama-3)**, and a premium **Glassmorphic UI**. This system can process PDFs (text & OCR) and images, providing intelligent answers based on your uploaded knowledge base.



## 🚀 Features

-   **Multimodal Processing**: Handles standard PDFs, scanned documents (via OCR), and images (BLIP-base for captioning).
-   **Agentic Intelligence**: Powered by `Llama-3-70b` on Groq for ultra-fast reasoning.
-   **Vector Storage**: Local FAISS vector store for efficient similarity search.
-   **Premium UI**: A sleek, responsive frontend with a glassmorphic design and smooth animations.
-   **Docker Ready**: Fully containerized for easy deployment across environments.
-   **LangChain Integrated**: Refactored to use LangChain for better maintainability and extensibility.

## 🛠️ Tech Stack

-   **Backend**: FastAPI, LangChain, FAISS, PyMuPDF, Pytesseract, BLIP-base.
-   **LLM**: Groq (Llama-3-70b), HuggingFace (for local embeddings & image captioning).
-   **Frontend**: Vanilla HTML/CSS/JS (Glassmorphic design).
-   **DevOps**: Docker, Docker Compose, Kubernetes.

## 📋 Prerequisites

Before running the project, ensure you have the following installed:

-   **Python 3.12+**
-   **Tesseract OCR**: Required for scanned PDF text extraction.
-   **Poppler**: Required for PDF-to-image conversion.
-   **Docker** (Optional).

### Installing System Dependencies (Mac)
```bash
brew install tesseract poppler
```

### Installing System Dependencies (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr poppler-utils
```

## ⚙️ Setup Instructions

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/aayushsupe69-pixel/multimodal-agentic-rag.git
    cd multimodal-agentic-rag
    ```

2.  **Create a Virtual Environment**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Mac/Linux
    # venv\Scripts\activate  # Windows
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r rag_project/requirements.txt
    ```

4.  **Configure Environment Variables**
    Create a `.env` file in the root directory (use `.env.example` as a template):
    ```env
    GROQ_API_KEY=your_groq_api_key
    HUGGINGFACE_API_KEY=your_huggingface_api_key
    ```

## 🏃 Running the Application

### Option 1: Standard Python
```bash
python rag_project/main.py
```
The app will be available at `http://localhost:8000`.

### Option 2: Docker Compose
```bash
cd rag_project
docker-compose up --build
```

### Option 3: Kubernetes
```bash
kubectl apply -f k8s/
```

## 📁 Project Structure

```text
.
├── k8s/                # Kubernetes deployment manifests
├── rag_project/        # Main application code
│   ├── frontend/       # Glassmorphic UI files
│   ├── rag/            # Core RAG logic (LangChain)
│   ├── main.py         # FastAPI entry point
│   └── Dockerfile      # Docker configuration
├── README.md           # This file
└── .env.example        # Environment variable template
```

---
Built with ❤️ by [Aayush Supe]
