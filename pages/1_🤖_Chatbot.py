"""Module dedicated to the chatbot page."""

# Import packages, modules and literals

import os

import streamlit as st

from src.chat import generate_response_streaming, load_chat
from src.constants import LOGO_PATH, MODEL_NAMES
from src.ingestion import create_index
from src.logging import logger
from src.opensearch import get_opensearch_client


def render_main_page() -> None:
    """Function to render main page."""
    st.set_page_config(page_title="Chatbot", page_icon="🤖")


# Main chatbot page rendering function
def render_chatbot_page() -> None:
    """Function to render the Chatbot page."""
    # Set up a placeholder at the very top of the main content area
    st.title("Chatbot 🤖")
    model_loading_placeholder = st.empty()

    # Initialize session state variables for chatbot settings
    if "use_rag" not in st.session_state:
        st.session_state["use_rag"] = False
    if "num_results" not in st.session_state:
        st.session_state["num_results"] = 4
    if "temperature" not in st.session_state:
        st.session_state["temperature"] = 0.7
    if "sources" not in st.session_state:
        st.session_state["sources"] = False
    if "search_type" not in st.session_state:
        st.session_state["search_type"] = None
    if "model_name" not in st.session_state:
        st.session_state["model_name"] = None

    # Initialize OpenSearch client
    with st.spinner("Connecting to OpenSearch..."):
        client = get_opensearch_client()

    # Ensure the index exists
    create_index(client)

    # Sidebar settings for app behaviour
    st.session_state["model_name"] = st.sidebar.selectbox(
        "Model Name",
        options=tuple(MODEL_NAMES.keys()),
        index=None,
        placeholder="Select an LLM",
    )

    st.session_state["use_rag"] = st.sidebar.checkbox(
        "Enable RAG Mode",
        value=st.session_state["use_rag"],
    )
    if st.session_state["use_rag"]:
        st.session_state["search_type"] = st.sidebar.selectbox(
            "Search Type",
            options=("hybrid", "vector", "keyword", "reranking"),
            index=None,
            placeholder="Select a search type",
        )
        st.session_state["num_results"] = st.sidebar.number_input(
            "Number of Results in Context Window",
            min_value=1,
            max_value=10,
            value=st.session_state["num_results"],
            step=1,
        )
        st.session_state["sources"] = st.sidebar.checkbox(
            "Return Sources (experimental)",
            value=st.session_state["sources"],
        )
    st.session_state["temperature"] = st.sidebar.slider(
        "Response Temperature",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state["temperature"],
        step=0.1,
    )
    # Log given settings
    logger.info(f"LLM chosen is: {str(st.session_state['model_name'])}")
    logger.info(f"RAG search set to: {str(st.session_state['use_rag'])}")
    logger.info(f"Search type set to: {str(st.session_state['search_type'])}")
    logger.info(
        f"Maximum retrieved chunks set to: {str(st.session_state['num_results'])}"
    )
    logger.info(f"Response temperature set to: {str(st.session_state['temperature'])}")
    logger.info(f"Sources set to: {str(st.session_state['sources'])}")

    # Display logo or placeholder
    if os.path.exists(LOGO_PATH):
        st.sidebar.image(LOGO_PATH, width=220, use_container_width=True)
        logger.info("Logo displayed.")
    else:
        st.sidebar.markdown("### Logo Placeholder")
        logger.warning("Logo not found, displaying placeholder.")

    # Sidebar headers and footer
    st.sidebar.markdown(
        "<h2 style='text-align: center;'>Your RAG Playground</h2>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        "<h4 style='text-align: center;'>© 2024 Alfredo Cinelli</h4>",
        unsafe_allow_html=True,
    )
    logger.info("Sidebar configured with headers and footer.")

    # Display loading spinner at the top of the main content area
    with model_loading_placeholder.container():
        st.spinner("Loading models for chat...")

    # Load model if not already loaded
    if "embedding_models_loaded" not in st.session_state:
        with model_loading_placeholder:  # noqa: SIM117
            with st.spinner("Loading Embedding for RAG Search..."):
                st.session_state["embedding_models_loaded"] = True
        logger.info("Embedding model loaded.")
        model_loading_placeholder.empty()

    # Initialize chat history in session state if not already present
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Display chat history
    for message in st.session_state["chat_history"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Process user input and generate response
    if prompt := st.chat_input("Type your message here..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state["chat_history"].append({"role": "user", "content": prompt})
        logger.info("User input received.")

        # Generate response from assistant
        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                response_placeholder = st.empty()
                response_text = ""

                response_stream = generate_response_streaming(
                    model=st.session_state["model_name"],
                    query=prompt,
                    use_rag=st.session_state["use_rag"],
                    search_type=st.session_state["search_type"],
                    num_results=st.session_state["num_results"],
                    temperature=st.session_state["temperature"],
                    chat_history=st.session_state["chat_history"],
                    return_sources=st.session_state["sources"],
                )

            # Stream response content if response_stream is valid
            if response_stream is not None:
                for chunk in response_stream:
                    if (
                        isinstance(chunk, dict)
                        and "message" in chunk
                        and "content" in chunk["message"]
                    ):
                        response_text += chunk["message"]["content"]
                        response_placeholder.markdown(response_text + "▌")
                    else:
                        logger.error("Unexpected chunk format in response stream.")

            response_placeholder.markdown(response_text)
            st.session_state["chat_history"].append(
                {"role": "assistant", "content": response_text}
            )
            logger.info("Response generated and displayed.")

    if st.session_state["chat_history"]:
        # Create download button for the chat
        download_button = st.download_button(
            label="Download chat",
            data=load_chat(st.session_state["chat_history"]),
            file_name="chat.csv",
            mime="text/csv",
            help="Click to download the chat as a CSV file.",
        )
        if download_button:
            logger.info("Downloaded chat as csv file.")


# Main execution
if __name__ == "__main__":
    render_main_page()
    render_chatbot_page()
