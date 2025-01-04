# REFACTORING ACTIVITIES:

- Remove direct calls to `sentence-tranformers` bi-encoder and use `langchain_huggingface` instead. Therefore, install it

- Remove direct calls to `sentence-transformers` cross-encoder and use `langchain_community` instead. Therefore, install it

- Create prompt via `langchain` and remove raw function using bare strings

- Refactor the retrieval functions using class with a more OOP approach (e.g., using ABC)

- Refactor the `ingestion.py` module creating a client class for `OpenSerch`