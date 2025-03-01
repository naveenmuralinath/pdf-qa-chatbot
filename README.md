# pdf-qa-chatbot
# PDF Question Answering Chatbot

This application allows users to upload PDF documents and ask questions about their content. The chatbot leverages Retrieval-Augmented Generation (RAG) to provide accurate answers based on the document content, using the Mistral 7B language model.

## Features

- PDF document upload and processing
- Text extraction and vectorization
- Question answering using the Mistral 7B LLM
- Conversational memory to maintain context
- Simple and intuitive Streamlit UI
- FAISS vector database for efficient semantic search
- Docker support for easy deployment

## Architecture

The application uses the following components:

- **Streamlit**: Provides the user interface for document upload and chat interaction
- **LangChain**: Orchestrates the document processing and QA pipeline
- **FAISS**: Vector database for storing document embeddings
- **Hugging Face Transformers**: For embeddings and LLM inference
- **Mistral 7B**: Open-source LLM for generating responses
- **PyPDFLoader**: For extracting text from PDF documents
- **Sentence Transformers**: For generating embeddings from text chunks

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Docker (optional, for containerized deployment)
- Hugging Face account (for API access to Mistral 7B)

### Local Setup

1. Clone the repository:
```bash
git clone https://github.com/naveenmuralinath/pdf-qa-chatbot.git
cd pdf-qa-chatbot
```

2. Install the required dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows
pip install -r requirements.txt
```

3. Set up Hugging Face credentials:
```bash
export HUGGINGFACE_API_TOKEN=your_huggingface_token
```
or create a `.env` file in the project root with:
```
HUGGINGFACE_API_TOKEN=your_huggingface_token
```

4. Run the application:
```bash
streamlit run app.py
```

5. Access the application in your browser at http://localhost:8501

### Docker Setup

1. Build the Docker image:
```bash
docker build -t pdf-qa-chatbot .
```

2. Run the container with your Hugging Face API token:
```bash
docker run -p 8501:8501 pdf-qa-chatbot
```

3. Access the application at http://localhost:8501

## Usage

### Uploading Documents

1. Start the application
2. Use the file uploader in the sidebar to upload one or more PDF documents
3. Wait for the "Processing complete" message for each document

### Asking Questions

1. Type your question in the text input field at the bottom of the main panel
2. Press Enter or click the "Submit" button
3. View the AI's response based on the content of your uploaded documents
4. Continue the conversation with follow-up questions

### Resetting the Conversation

Click the "Reset Conversation" button in the sidebar to clear the chat history and uploaded documents.

## Technical Implementation Details

### Document Processing Pipeline

1. **PDF Loading**: PDFs are loaded using PyPDFLoader from LangChain
2. **Text Chunking**: Documents are split into manageable chunks of 1000 characters with 200 character overlap using RecursiveCharacterTextSplitter
3. **Embedding Generation**: Text chunks are converted to vector embeddings using the "sentence-transformers/all-MiniLM-L6-v2" model
4. **Vector Storage**: Embeddings are stored in a FAISS vector database for efficient similarity search

### Question Answering System

1. **Query Processing**: User questions are embedded using the same model as document chunks
2. **Semantic Retrieval**: FAISS retrieves the top 3 most relevant document chunks based on similarity
3. **Context Augmentation**: Retrieved document chunks are provided as context to the Mistral 7B model
4. **Response Generation**: Mistral 7B generates a contextually relevant response based on both the question and the retrieved document content
5. **Conversation Memory**: The system maintains conversation history for contextual follow-up questions

## Parameter Customization

You can customize various parameters in the `app.py` file:

- **Chunk Size**: Adjust `chunk_size` in the text splitter (default: 1000)
- **Chunk Overlap**: Adjust `chunk_overlap` in the text splitter (default: 200)
- **Retriever Top K**: Adjust `k` in the retriever to control how many chunks are retrieved (default: 3)
- **LLM Temperature**: Adjust `temperature` in the model_kwargs to control response randomness (default: 0.5)
- **Max Length**: Adjust `max_length` in the model_kwargs to control response length (default: 512)

## System Requirements

- **CPU**: Minimum dual-core processor, quad-core recommended
- **RAM**: Minimum 4GB, 8GB or more recommended for larger documents
- **Storage**: 1GB free space for application and dependencies
- **Internet Connection**: Required for Hugging Face API access

## Known Limitations

- Large PDF files (>100 pages) may take longer to process
- Currently, new PDFs replace the previous knowledge base rather than merging with it
- PDFs with complex formatting or scanned images may not be processed correctly
- Performance depends on the available computational resources
- The quality of answers depends on the quality and relevance of the uploaded documents

## Troubleshooting

### Common Issues

1. **PDF Processing Errors**:
   - Make sure your PDF is not password-protected
   - Try converting problematic PDFs to a different format and back to PDF

2. **Out of Memory Errors**:
   - Reduce the chunk size in the text splitter
   - Process smaller documents or fewer documents at a time

3. **Hugging Face API Errors**:
   - Verify your API token is correct
   - Check your API rate limits on the Hugging Face website

4. **Slow Response Times**:
   - Consider using a more powerful machine
   - Reduce the max_length parameter for the LLM

### Getting Help

If you encounter issues not covered here, please open an issue on the GitHub repository with:
- A description of the problem
- Steps to reproduce
- Any error messages
- System information

## Future Improvements

- Implement vector store merging for multiple documents
- Add support for more document formats (DOCX, TXT, HTML, etc.)
- Improve chunk retrieval with hybrid search (keyword + semantic)
- Add user authentication and document persistence
- Implement document metadata filtering
- Add options to adjust model parameters through the UI
- Implement error handling for corrupted documents
- Add support for document summarization
- Implement caching for faster repeated queries
- Add export functionality for conversation history

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) for the document processing and RAG framework
- [FAISS](https://github.com/facebookresearch/faiss) for the vector database
- [Hugging Face](https://huggingface.co/) for the Mistral 7B model and embeddings
- [Streamlit](https://streamlit.io/) for the user interface framework
- [Mistral AI](https://mistral.ai/) for the open-source language model

## Contact

Project Maintainer - [naveenmuralinath@gmail.com](mailto:naveenmuralinath@gmail.com)

Project Link: [https://github.com/naveenmuralinath/pdf-qa-chatbot](https://github.com/naveenmuralinath/pdf-qa-chatbot)
