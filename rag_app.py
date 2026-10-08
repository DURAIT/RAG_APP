
import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
# ============================================================
# CONFIGURATION
# ============================================================

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 5

EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"


# ============================================================
# STREAMLIT PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Multi-PDF RAG Assistant",
    page_icon="📚",
    layout="wide",
)


# ============================================================
# SESSION STATE
# ============================================================

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []


# ============================================================
# OPENAI API KEY
# ============================================================

def get_api_key():
    """
    Read OpenAI API key from environment variable.
    """

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable is not set."
        )

    return api_key


# ============================================================
# BUILD FAISS VECTOR STORE
# ============================================================

def build_vector_store(uploaded_files):
    """
    Load multiple PDFs, preserve filename/page metadata,
    split them into chunks and create ONE FAISS vector store.
    """

    all_documents = []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ],
    )

    # Temporary directory because PyPDFLoader works with file paths.
    with tempfile.TemporaryDirectory() as temp_dir:

        temp_dir = Path(temp_dir)

        # ----------------------------------------------------
        # Load every uploaded PDF
        # ----------------------------------------------------

        for uploaded_file in uploaded_files:

            pdf_path = temp_dir / uploaded_file.name

            # Save uploaded PDF temporarily.
            pdf_path.write_bytes(
                uploaded_file.getvalue()
            )

            # Load PDF page by page.
            loader = PyPDFLoader(
                str(pdf_path)
            )

            pages = loader.load()

            # ------------------------------------------------
            # Add our own metadata
            # ------------------------------------------------

            for document in pages:

                # PyPDFLoader uses zero-based page numbers.
                page_zero_based = document.metadata.get(
                    "page",
                    0
                )

                document.metadata["source_file"] = (
                    uploaded_file.name
                )

                # Human-readable page number.
                document.metadata["page_number"] = (
                    int(page_zero_based) + 1
                )

            # ------------------------------------------------
            # Split pages into chunks
            # ------------------------------------------------

            chunks = text_splitter.split_documents(
                pages
            )

            all_documents.extend(chunks)

    # --------------------------------------------------------
    # Make sure something was extracted
    # --------------------------------------------------------

    if not all_documents:
        raise ValueError(
            "No text could be extracted from the uploaded PDFs."
        )

    # --------------------------------------------------------
    # Create OpenAI embeddings
    # --------------------------------------------------------

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL
    )

    # --------------------------------------------------------
    # ONE FAISS DATABASE FOR ALL PDFs
    # --------------------------------------------------------

    vector_store = FAISS.from_documents(
        documents=all_documents,
        embedding=embeddings
    )

    return vector_store, len(all_documents)


# ============================================================
# QUESTION ANSWERING
# ============================================================

def answer_question(question):
    """
    Retrieve relevant chunks from ALL uploaded PDFs
    and generate an answer.
    """

    vector_store = st.session_state.vector_store

    if vector_store is None:
        raise RuntimeError(
            "FAISS vector store has not been created yet."
        )

    # --------------------------------------------------------
    # Retrieve from the single FAISS store
    # --------------------------------------------------------

    retrieved_docs = vector_store.similarity_search(
        question,
        k=TOP_K
    )

    if not retrieved_docs:

        return (
            "I could not find relevant information "
            "in the uploaded documents.",
            []
        )

    # --------------------------------------------------------
    # Build context
    # --------------------------------------------------------

    context_parts = []

    for index, document in enumerate(
        retrieved_docs,
        start=1
    ):

        source_file = document.metadata.get(
            "source_file",
            "Unknown file"
        )

        page_number = document.metadata.get(
            "page_number",
            "Unknown"
        )

        context_parts.append(
            f"""
[SOURCE {index}]
File: {source_file}
Page: {page_number}

Content:
{document.page_content}
"""
        )

    context = "\n\n".join(context_parts)

    # --------------------------------------------------------
    # Create LLM
    # --------------------------------------------------------

    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0
    )

    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    prompt = f"""
You are a document question-answering assistant.

Answer the user's question using ONLY the information
provided in the document context below.

Important rules:

1. Do not invent information.
2. Do not use outside knowledge.
3. If the documents do not contain enough information,
   clearly say that the information was not found.
4. Give a clear and concise answer.
5. You may combine information from multiple documents.
6. Do not create a bibliography or source list yourself.
   The application will display the sources separately.

User question:

{question}

Document context:

{context}
"""

    # --------------------------------------------------------
    # Ask the LLM
    # --------------------------------------------------------

    response = llm.invoke(prompt)

    answer = response.content

    # --------------------------------------------------------
    # Extract unique sources
    # --------------------------------------------------------

    sources = []

    seen_sources = set()

    for document in retrieved_docs:

        source_file = document.metadata.get(
            "source_file",
            "Unknown file"
        )

        page_number = document.metadata.get(
            "page_number",
            "Unknown"
        )

        source_key = (
            source_file,
            page_number
        )

        if source_key not in seen_sources:

            seen_sources.add(
                source_key
            )

            sources.append(
                {
                    "file": source_file,
                    "page": page_number
                }
            )

    return answer, sources


# ============================================================
# STREAMLIT UI
# ============================================================

st.title("📚 Multi-PDF RAG Assistant")

st.write(
    """
Upload multiple PDF documents and ask questions across
all of them. Every answer displays the source PDF and
page number used for retrieval.
"""
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("RAG Configuration")

    st.write(
        f"**Chunk size:** {CHUNK_SIZE}"
    )

    st.write(
        f"**Chunk overlap:** {CHUNK_OVERLAP}"
    )

    st.write(
        f"**Retrieval:** Top {TOP_K} chunks"
    )

    st.write(
        f"**Embedding:** `{EMBEDDING_MODEL}`"
    )

    st.write(
        f"**LLM:** `{LLM_MODEL}`"
    )

    st.write(
        "**Vector database:** FAISS"
    )

    st.divider()

    if st.session_state.vector_store:

        st.success(
            f"""
Index ready

Files:
{len(st.session_state.indexed_files)}

Chunks:
{st.session_state.chunk_count:,}
"""
        )


# ============================================================
# PDF UPLOAD
# ============================================================

st.header("1️⃣ Upload PDF documents")

uploaded_files = st.file_uploader(
    "Select one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True
)


# ============================================================
# DISPLAY SELECTED FILES
# ============================================================

if uploaded_files:

    st.write(
        f"**{len(uploaded_files)} PDF(s) selected:**"
    )

    for uploaded_file in uploaded_files:

        st.write(
            f"📄 {uploaded_file.name}"
        )

    # --------------------------------------------------------
    # BUILD INDEX BUTTON
    # --------------------------------------------------------

    if st.button(
        "🔨 Build FAISS Index",
        type="primary"
    ):

        try:

            # Make sure API key exists.
            get_api_key()

            with st.spinner(
                "Loading PDFs, creating chunks, "
                "generating embeddings and building FAISS..."
            ):

                vector_store, chunk_count = (
                    build_vector_store(
                        uploaded_files
                    )
                )

            # ------------------------------------------------
            # Store everything in session state
            # ------------------------------------------------

            st.session_state.vector_store = (
                vector_store
            )

            st.session_state.chunk_count = (
                chunk_count
            )

            st.session_state.indexed_files = [
                uploaded_file.name
                for uploaded_file in uploaded_files
            ]

            # ------------------------------------------------
            # Success message
            # ------------------------------------------------

            st.success(
                f"""
FAISS index built successfully!

📄 Documents indexed: {len(uploaded_files)}

🧩 Total chunks indexed: {chunk_count:,}
"""
            )

        except Exception as error:

            st.error(
                f"Error while building index: {error}"
            )


# ============================================================
# QUESTION AREA
# ============================================================

st.divider()

st.header("2️⃣ Ask a question")


if st.session_state.vector_store is None:

    st.info(
        "Upload PDFs and click 'Build FAISS Index' "
        "before asking a question."
    )

else:

    st.write(
        f"""
Searching across
**{len(st.session_state.indexed_files)} PDFs**
and **{st.session_state.chunk_count:,} chunks**.
"""
    )

    question = st.text_input(
        "Enter your question",
        placeholder=(
            "Example: What technologies are used "
            "in the backend architecture?"
        )
    )

    # --------------------------------------------------------
    # ASK BUTTON
    # --------------------------------------------------------

    if st.button(
        "🔎 Ask Question",
        type="primary",
        disabled=not question.strip()
    ):

        try:

            get_api_key()

            with st.spinner(
                "Searching all documents..."
            ):

                answer, sources = answer_question(
                    question.strip()
                )

            # ------------------------------------------------
            # Answer
            # ------------------------------------------------

            st.subheader("Answer")

            st.write(answer)

            # ------------------------------------------------
            # Sources
            # ------------------------------------------------

            st.subheader("📚 Sources")

            if sources:

                for source in sources:

                    st.markdown(
                        f"📄 **{source['file']}**  \n"
                        f"Page **{source['page']}**"
                    )

            else:

                st.write(
                    "No source metadata was returned."
                )

        except Exception as error:

            st.error(
                f"Error while answering question: {error}"
            )


# ============================================================
# ENVIRONMENT SETUP HELP
# ============================================================

st.divider()

with st.expander(
    "🔑 OpenAI API key setup"
):

    st.markdown(
        """
The application reads the API key from:

`OPENAI_API_KEY`
"""
    )

    st.markdown("### Windows PowerShell")

    st.code(
        '$env:OPENAI_API_KEY="your-api-key"',
        language="powershell"
    )

    st.markdown("### Windows CMD")

    st.code(
        "set OPENAI_API_KEY=your-api-key",
        language="cmd"
    )

    st.markdown("### macOS / Linux")

    st.code(
        'export OPENAI_API_KEY="your-api-key"',
        language="bash"
    )

    st.markdown("### Start the application")

    st.code(
        "streamlit run rag_app.py",
        language="bash"
    )

