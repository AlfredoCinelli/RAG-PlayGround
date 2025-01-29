"""Moduel for Upload Documents page."""

# Import packages, modules and literals

import os
import time

import streamlit as st

from src.constants import (
    CHUNK_OVERLAP_SIZE,
    LOGO_PATH,
    OPENSEARCH_INDEX,
    TEXT_CHUNK_SIZE,
)
from src.encoders import EmbeddingModel
from src.ingestion import DocumentLoader, DocumentSplitter
from src.logging import logger
from src.opensearch.indexing import OpenSearchIndex
from src.utils import save_uploaded_file


def render_main_page() -> None:
    """Function to render the main page."""
    # Set page config with title, icon, and layout
    st.set_page_config(page_title="Upload Documents", page_icon="📂")


def render_upload_page() -> None:
    """
    Renders the document upload page for users to upload and manage PDFs.
    Shows only the documents that are present in the OpenSearch index.
    """
    # Add a logo (replace with your own image file path or URL) # Replace with your logo file
    if os.path.exists(LOGO_PATH):  # noqa PLR0915
        st.sidebar.image(LOGO_PATH, width=220, use_container_width=True)
    else:
        st.sidebar.markdown("### Logo Placeholder")
        logger.warning("Logo not found, displaying placeholder.")

    # Sidebar header
    st.sidebar.markdown(
        "<h2 style='text-align: center;'>Your RAG Playground</h2>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        "<h4 style='text-align: center;'>© 2024 Alfredo Cinelli</h4>",
        unsafe_allow_html=True,
    )
    st.title("Upload Documents 📄")

    UPLOAD_DIR = "uploaded_files"  # noqa n806
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    # Initialize OpenSearch client
    with st.spinner("Connecting to OpenSearch..."):
        index = OpenSearchIndex()
        index_name = OPENSEARCH_INDEX  # reference to index
        index.create_index()  # ensure the index exists

    # Initialize or clear the documents list in session state
    st.session_state["documents"] = []

    # Query OpenSearch to get the list of unique document names
    query = {
        "size": 0,
        "aggs": {"unique_docs": {"terms": {"field": "document_name", "size": 10000}}},
    }
    response = index.client.search(index=index_name, body=query)
    buckets = response["aggregations"]["unique_docs"]["buckets"]
    document_names = [bucket["key"] for bucket in buckets]
    logger.info("Retrieved document names from OpenSearch.")

    # Load document information from the index
    for document_name in document_names:
        file_path = os.path.join(UPLOAD_DIR, document_name)
        if os.path.exists(file_path):
            st.session_state["documents"].append(
                {"filename": document_name, "file_path": file_path}
            )
        else:
            st.session_state["documents"].append({"filename": document_name, "file_path": None})
            logger.warning(f"File '{document_name}' does not exist locally.")

    if "deleted_file" in st.session_state:
        st.success(f"The file '{st.session_state['deleted_file']}' was successfully deleted.")
        del st.session_state["deleted_file"]

    # Allow users to upload PDF files
    uploaded_files = st.file_uploader(
        "Upload PDF documents", type="pdf", accept_multiple_files=True
    )

    if uploaded_files:
        with st.spinner("Uploading and processing documents. Please wait..."):
            for uploaded_file in uploaded_files:
                if uploaded_file.name in document_names:
                    st.warning(f"The file '{uploaded_file.name}' already exists in the index.")
                    continue

                file_path = save_uploaded_file(uploaded_file)
                loaded_file = DocumentLoader(file_path=file_path, mode="pypdf").load()
                chunks = DocumentSplitter(
                    chunk_size=TEXT_CHUNK_SIZE,
                    overlap=CHUNK_OVERLAP_SIZE,
                    tokenization=True,
                    mode="recursive",
                ).split(loaded_file)
                embeddings = EmbeddingModel().embed_documents(chunks)  # type: ignore

                documents_to_index = [
                    {
                        "doc_id": f"{uploaded_file.name}_{i}",
                        "text": chunk,
                        "embedding": embedding,
                        "document_name": uploaded_file.name,
                    }
                    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=False))
                ]
                index.bulk_index_documents(documents_to_index)
                st.session_state["documents"].append(
                    {
                        "filename": uploaded_file.name,
                        "file_path": file_path,
                    }
                )
                document_names.append(uploaded_file.name)
                logger.info(f"File '{uploaded_file.name}' uploaded and indexed.")

        st.success("Files uploaded and indexed successfully!")

    if st.session_state["documents"]:
        st.markdown("### Uploaded Documents")
        with st.expander("Manage Uploaded Documents", expanded=True):
            for idx, doc in enumerate(st.session_state["documents"], 1):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"{idx}. {doc['filename']}")
                with col2:
                    delete_button = st.button(
                        "Delete",
                        key=f"delete_{doc['filename']}_{idx}",
                        help=f"Delete {doc['filename']}",
                    )
                    if delete_button:
                        if doc["file_path"] and os.path.exists(doc["file_path"]):
                            try:
                                os.remove(doc["file_path"])
                                logger.info(f"Deleted file '{doc['filename']}' from file system.")
                            except FileNotFoundError:
                                st.error(f"File '{doc['filename']}' not found in file system.")
                                logger.error(f"File '{doc['filename']}' not found during deletion.")
                        index.delete_documents_by_name(doc["filename"])
                        st.session_state["documents"].pop(idx - 1)
                        st.session_state["deleted_file"] = doc["filename"]
                        time.sleep(0.5)
                        st.rerun()


if __name__ == "__main__":
    render_main_page()
    if "documents" not in st.session_state:
        st.session_state["documents"] = []
    render_upload_page()
