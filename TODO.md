# REFACTORING ACTIVITIES:

- Remove direct calls to `sentence-tranformers` bi-encoder and use `langchain_huggingface` instead. Therefore, install it

- Remove direct calls to `sentence-transformers` cross-encoder and use `langchain_community` instead. Therefore, install it

- Create prompt via `langchain` and remove raw function using bare strings

- Refactor the retrieval functions using class with a more OOP approach

- Refactor the `ingestion.py` module creating a client class for `OpenSerch`

- Improve chunking strategy using semantic chunking and converting format to markdown to leverage logical structure of the document

- Setup an evaluation suite for the RAG components, try to use Ragas or Giskard

- Remove any `Pandas` reference with `Polars`

- Refactor constants from Python module to logical yaml files under the conf folder

- Add PR template

- Allow logs in terminal and remove unecessary logs

- Create test suite