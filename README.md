# WhatsApp Chat RAG

A retrieval-augmented generation (RAG) API built to power a WhatsApp chatbot. A question goes in, the most relevant passages are retrieved from a ChromaDB vector store, and an Azure OpenAI chat model answers using only that context.

**Status:** the RAG API backend is built and set up for deployment to Azure. The WhatsApp channel integration (receiving messages and replying through WhatsApp) is the next step and is not yet in this repository.

## How it works

```
 WhatsApp message ──(planned)──▶  POST /ask {query}  ──▶  Flask API (app.py)
                                                              │
                                                              ▼
                                            ChromaDB similarity search (db/)
                                                              │
                                                              ▼
                                       Azure OpenAI (gpt-35-turbo) answers from the retrieved text
                                                              │
 WhatsApp reply  ◀──(planned)──  {"answer": "..."}  ◀─────────┘
```

1. **Ingestion (run once).** `chromasetup.py` loads `unique_rag_context.csv`, embeds each row with Azure OpenAI embeddings and saves the vectors to a persistent ChromaDB store in `db/`.
2. **Retrieval.** For each question, `rag_pipeline.py` loads the store as a retriever and finds the closest passages.
3. **Generation.** A LangChain `RetrievalQA` chain sends the retrieved passages and the question to an Azure OpenAI `gpt-35-turbo` deployment.
4. **API.** `app.py` exposes the pipeline as a single Flask endpoint, `POST /ask`.

## Tech stack

| Layer | Technology |
|---|---|
| API | Python 3.10, Flask |
| RAG orchestration | LangChain (`RetrievalQA`, `CSVLoader`) |
| Embeddings and LLM | Azure OpenAI (embedding deployment and `gpt-35-turbo`) |
| Vector store | ChromaDB (persistent, local) |
| CI/CD and hosting | GitHub Actions, Azure Web App |

## The data

The sample knowledge base in `unique_rag_context.csv` contains five short passages about invented things (a fictional planet, a fictional language, a made-up material and so on). The facts are fictional on purpose: a model cannot know them from its training data, so a correct answer shows that the answer came from retrieval. `sample_rag_context.csv` holds five real-world facts for comparison.

To use your own knowledge base, replace the CSV (an `id` column and a `context` column) and re-run `chromasetup.py`.

## API

`POST /ask`

```json
// request
{ "query": "What is ChronoFabrics?" }

// response
{ "answer": "..." }
```

- Missing or empty `query` returns HTTP 400 with `{ "error": "Query not provided" }`.
- Any pipeline error returns HTTP 500 with `{ "error": "<message>" }`.

## Project structure

```
whatsappChat_rag/
├── app.py                      # Flask API with the /ask endpoint
├── rag_pipeline.py             # Retriever, Azure OpenAI chat model, RetrievalQA chain
├── chromasetup.py              # One-off ingestion script: CSV → embeddings → ChromaDB
├── unique_rag_context.csv      # Knowledge base (fictional facts)
├── sample_rag_context.csv      # Real-world sample facts
├── db/                         # Persistent ChromaDB vector store
├── requirements.txt
└── .github/workflows/
    └── main_rag-api-backend.yml   # Build and deploy to Azure Web App
```

## Configuration

Create a `.env` file in the project root (it is gitignored):

```
OPENAI_API_KEY=your_azure_openai_key
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_EMBEDDING_DEPLOYMENT=your_embedding_deployment_name
OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
OPENAI_DEPLOYMENT=your_chat_deployment_name
OPENAI_API_VERSION=your_api_version
```

## Running locally

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install langchain-openai langchain-community langchain-chroma

python chromasetup.py             # build the vector store (once)
python app.py                     # API on http://127.0.0.1:5000
```

Try it:

```bash
curl -X POST http://127.0.0.1:5000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is KIRO-7?"}'
```

## Deployment (CI/CD)

`.github/workflows/main_rag-api-backend.yml` builds and deploys the app to an Azure Web App named `rag-api-backend` on every push to `main` (it can also be triggered manually). The pipeline:

1. Sets up Python 3.10 and installs `requirements.txt`.
2. Packages the app as an artifact.
3. Signs in to Azure using GitHub secrets (OIDC) and deploys with `azure/webapps-deploy`.

No credentials are stored in the repository.

## Roadmap

- **WhatsApp integration.** Add a webhook endpoint that receives incoming WhatsApp messages from a messaging provider, passes the text to the RAG pipeline and sends the answer back as a reply.
- **Per-user conversation memory,** keyed by phone number, so follow-up questions keep context.
- **Efficiency.** Build the retriever and chain once at startup instead of on every request.
- **Real knowledge base.** Replace the sample facts with real documents, with chunking suited to chat-length answers.
- **Quality checks.** Add automated tests and a small question-and-answer evaluation set, and run them in the CI pipeline before deploying.
- **Production hardening.** Run behind a production server such as gunicorn and add input validation and rate limiting.
