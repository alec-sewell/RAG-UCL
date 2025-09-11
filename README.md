# RAG-UCL

This project, "RAG-UCL", is a comprehensive platform for the development and evaluation of Retrieval-Augmented Generation (RAG) pipelines. The core of this project is "RALF," a self-contained RAG system. The project also includes extensive tooling for evaluation and analysis, leveraging frameworks like RAGAs for automated assessment and providing capabilities for statistical analysis of the results.

## Project Structure

The project is organized into the following main directories:

-   **`RAG development/`**: This directory houses the primary source code for the RAG system.
    -   **`code/RALF/`**: Contains the core "RALF" application logic.
        -   **`lib/`**: A library of Python modules supporting the RALF application, including functionalities for configuration, embedding management, logging, and data processing.
        -   **`RALF.py`**: The main executable script for running the RALF RAG pipeline.
    -   **`code/pretest/`**: Includes scripts for preliminary tests and evaluations of the RAG system.
    -   **`Requirements.txt`**: A list of Python dependencies required to run the project.
    -   **`.env`**: A file for environment variable configurations, such as API keys.

-   **`RAGAS_Automated_Eval/`**: This directory contains scripts and notebooks dedicated to the automated evaluation of the RAG pipeline, utilizing the RAGAs framework.

-   **`Statistical_Analysis/`**: Provides scripts and tools for performing statistical analysis on the data gathered from RAG evaluations.

## Core Components (RALF)

The RALF system is modular, with its key functionalities encapsulated in the `RAG development/code/RALF/lib/` directory:

-   **`RAGManager.py`**: The central controller of the RAG pipeline, responsible for orchestrating the retrieval of relevant documents and the generation of responses.
-   **`PDFProcessor.py`** and **`TextProcessor.py`**: These modules handle the ingestion, cleaning, and preprocessing of source documents in both PDF and plain text formats.
-   **`RateLimitedEmbeddings.py`** and **`DiskCacheEmbeddings.py`**: These components manage the process of generating text embeddings. They are designed to work efficiently with external embedding APIs by handling rate limits and caching embeddings on disk to avoid redundant computations.
-   **`Config.py`**: This module manages the application's configuration, loading settings from environment variables or configuration files.
-   **`Logger.py`**: A logging utility for recording session data and other important events, helping with debugging and tracking experiments.

## Getting Started

To get the project up and running, follow these steps:

### Prerequisites

-   Python 3.x

### Installation

1.  Clone the repository to your local machine.
2.  Install the required Python packages using pip:
```shell script
pip install -r "RAG development/Requirements.txt"
```


### Configuration

1.  Create a `.env` file within the `RAG development/` directory.
2.  Populate the `.env` file with the necessary environment variables. This would typically include API keys for services like OpenAI or other embedding providers.

## Usage

-   The main RAG application can be executed by running the `RALF.py` script:
```shell script
python "RAG development/code/RALF/RALF.py"
```

-   For evaluation, explore the scripts in the `RAGAS_Automated_Eval/` directory.
-   For analysis of results, use the tools provided in the `Statistical_Analysis/` directory.
