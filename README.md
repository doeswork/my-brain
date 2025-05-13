# My Brain LLM Context Q&A

> 🎥 **Demo Video:** https://x.com/hackertwinz/status/1922323515916624136

A lightweight context-driven Q&A tool using ChromaDB and Llama 3.2 for local inference.

## Features
- 🌐 Reads all `.txt` files in the `context/` folder
- 🔍 Splits documents into chunks for efficient retrieval
- 🔎 Builds a Chroma vector store for semantic search
- 🤖 Uses Llama 3.2-instruct model for generation
- 🚀 Accelerated with GPU support via Accelerate

## Requirements
- Python 3.10+
- RTX GPU (optional, for acceleration)
- Virtual environment (recommended)

## Installation
```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage
1. Place your `.txt` files into the `context/` directory.
2. Build the vector store and run a query:
   ```bash
   python my_brain.py "Your question here"
   ```
3. The script will output an answer grounded in your documents.

## Project Structure
```
.
├── context/            # Your .txt documents
├── chroma_db/          # Persisted Chroma vector database
├── my_brain.py         # Main script
├── requirements.txt    # Project dependencies
└── README.md           # This file
```

## Contributing
Feel free to open issues or submit PRs to improve functionality, add support for more LLMs, or enhance UX.

## License
This project is licensed under the MIT License.
