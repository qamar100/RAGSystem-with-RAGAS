# RAG System with Synthetic Data Generation and Evaluation using RAGAS

This project implements a **Retrieval-Augmented Generation (RAG) system** with FastAPI for real-time querying and evaluation. The system integrates multiple technologies for document indexing, context retrieval, and conversational memory. It also includes functionality for generating synthetic data and evaluating the system's performance using RAGAS.

## Features

- **RAG System**: Utilizes LlamaIndex for document indexing and retrieval, LangChain for managing conversation history, and OpenAI models for generating responses.
- **FastAPI Integration**: Provides a web API for querying the RAG system with support for real-time response streaming.
- **Synthetic Data Generation**: Generates synthetic question-answer pairs relevant to your specific use case using OpenAI's GPT models.
- **Data Formatting and Evaluation**: Formats generated data for RAGAS and evaluates the RAG system's performance.

## Components

1. **RAG System (`RAGsystem.py`)**:
   - Sets up FastAPI with endpoints for querying.
   - Uses LlamaIndex for document indexing and retrieval.
   - Manages conversational history with LangChain.

2. **Synthetic Data Generation (`GenerateData.py`)**:
   - Generates synthetic question-answer pairs using OpenAI's GPT models.
   - Formats and saves the data in JSON for RAGAS evaluation.

3. **Data Formatting for RAGAS (`formattedDS.py`)**:
   - Converts the synthetic data into a format suitable for RAGAS evaluation.

4. **Evaluation Script (`RagasEval.py`)**:
   - Loads the formatted dataset and evaluates the RAG system using RAGAS metrics.
   - Analyzes and prints the results for various evaluation metrics.

## Getting Started

### Prerequisites

- Python 3.8+
- OpenAI API key
- LlamaIndex API key
- Required Python packages (listed in `requirements.txt`)

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/qamar100/RAGSystem-with-RAGAS.git
    cd RAGSystem-with-RAGAS
    ```

2. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

3. Set up environment variables:

    Create a `.env` file in the root directory with the following content:

    ```env
    OPENAI_API_KEY=your_openai_api_key
    LLAMA_CLOUD_API_KEY=your_llama_cloud_api_key
    ```

### Usage

1. **Generate Synthetic Data:**

    Run the script to generate synthetic question-answer pairs relevant to your specific use case:

    ```bash
    python GenerateData.py
    ```

    **Note**: Modify the prompts in `GenerateData.py` to match your specific use case. Adjust the system and user messages to ensure they are relevant to your domain.

2. **Format Data for RAGAS:**

    Convert the generated data into RAGAS-compatible format:

    ```bash
    python formattedDS.py
    ```

    **Note**: After generating and formatting the data, review the `qa_dataset.json` and `ragas_dataset.json` files. Make any necessary adjustments to ensure data quality before proceeding with evaluation.

3. **Run the RAG System:**

    Start the FastAPI server:

    ```bash
    python RAGsystem.py
    ```

    **Note**: Update the prompts in `RAGsystem.py` to suit your specific use case. Modify the query handling and response generation to fit your requirements.

4. **Evaluate the RAG System:**

    Run the evaluation script to assess the system's performance:

    ```bash
    python RagasEval.py
    ```

### API Endpoints

- **POST /query**: Accepts a JSON payload with a `text` field containing the query. Returns the RAG system's response.

### Results

- Evaluation results are printed to the console, including metrics such as faithfulness, answer relevancy, context recall, and context utilization.

## Contributing

Feel free to submit issues or pull requests to improve the system.
