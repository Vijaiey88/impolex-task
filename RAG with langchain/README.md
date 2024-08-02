
# Retrieval Augmented Generation (RAG) with LangChain

## Overview


This project implements a Retrieval Augmented Generation (RAG) system using LangChain. The system allows users to upload a PDF file, split the PDF into chunks, embed the chunks, and then answer questions based on the retrieved chunks using a language model.

## Installation

To install the necessary libraries and dependencies, use the following command:

```bash
pip install langchain PyMuPDF google-generative-ai faiss-cpu
```


## Run locally

1. Download the rag.ipynb file to your local machine.
2. Open the rag.ipynb file in Jupyter Notebook or JupyterLab.
3. Update the path to the PDF file in the notebook according to the location on your machine.
4. Use the API key generated from Google AI Studio and update the corresponding cell in the notebook.
5. Execute each cell in the notebook sequentially to process the PDF, create embeddings, and set up the QA system.
## Acknowledgements

 - [Retrieval Augmented Generation (RAG)](https://python.langchain.com/v0.2/docs/tutorials/rag/)
 - [Google Generative AI Embeddings](https://python.langchain.com/v0.2/docs/integrations/text_embedding/google_generative_ai/)

## Conclusion

This project provides a basic implementation of a Retrieval Augmented Generation system using LangChain. It can be extended and customized based on specific needs for handling and querying PDF documents.
## Feedback

If you have any feedback, please reach out to us at vijaiey88@gmail.com

