# 🍺 Brew Oracle

**Brew Oracle** é um agente de IA especializado em cerveja artesanal, treinado a partir de PDFs técnicos.
Ele responde **usando a base de conhecimento indexada**, **cita as fontes** e mantém o tom claro, útil e bem-humorado.

---

## 🔎 Visão geral

### Stack principal

- Python 3.13
- Agno (agentes + RAG)
- Qdrant (banco vetorial)
- Sentence Transformers (embeddings)
- Google Gemini (modelo de linguagem)

**Fluxo:** Usuário → Orquestrador (Agno) → Busca no Qdrant → *(opcional: Rerank)* → Gemini → Resposta **com referências**.

---

## 📁 Estrutura do projeto

```
.
├─ knowledge/
│  └─ pdfs/                           # coloque seus PDFs aqui
├─ models/
│  └─ all-MiniLM-L6-v2/               # cache local do embedder (opcional)
├─ src/
│  └─ brew_oracle/
│     ├─ core/
│     │  └─ run.py                    # ponto de entrada do agente (CLI)
│     ├─ knowledge/
│     │  └─ pdf_kb.py                 # construção/ingestão da KB de PDFs
│     ├─ orchestrator/
│     │  └─ brewing_orchestrator.py   # agente orquestrador
│     ├─ scripts/
│     │  ├─ create_collections.py     # cria a collection no Qdrant
│     │  └─ query_with_rerank.py      # busca + rerank (opcional)
│     └─ utils/
│        └─ config.py                 # Settings (lê .env)
├─ .env
├─ pyproject.toml
└─ README.md
```

---

## ⚙️ Pré-requisitos

- Docker (para Qdrant)
- [PDM](https://pdm.fming.dev/) (gerenciador de pacotes)

---

## 🚀 Instalação & setup

1. **Clonar & instalar dependências**

    ```bash
    git clone https://github.com/<seu-usuario>/brew_oracle.git
    cd brew_oracle
    pdm install
    ```

2. **Subir Qdrant**

    ```bash
    docker run -d -p 6333:6333 \
      -v "$(pwd)/qdrant_storage:/qdrant/storage" \
      qdrant/qdrant:latest
    ```

3. **Configurar `.env`**

    Crie um arquivo `.env` na raiz:

    ```ini
    QDRANT_URL=http://localhost:6333
    QDRANT_COLLECTION=brew_books

    PDF_PATH=knowledge/pdfs

    # Use o cache local do modelo para evitar 429 do HF:
    EMBEDDER_ID=./models/all-MiniLM-L6-v2
    EMBEDDER_DIM=384

    TOP_K=25

    # Gemini (AI Studio): https://aistudio.google.com/
    GOOGLE_API_KEY=coloque_sua_chave_aqui
    ```

4. **(Opcional) Baixar o embedder localmente**

    ```bash
    pdm run python - <<'PY'
    from sentence_transformers import SentenceTransformer
    import os
    os.makedirs("models", exist_ok=True)
    m=SentenceTransformer("all-MiniLM-L6-v2")
    m.save("models/all-MiniLM-L6-v2")
    print("Modelo salvo em models/all-MiniLM-L6-v2")
    PY
    ```

5. **Criar a collection no Qdrant**

    ```bash
    pdm run python -m brew_oracle.scripts.create_collections
    ```

6. **Adicionar PDFs**

    Coloque seus PDFs em `knowledge/pdfs/`.

7. **Ingerir PDFs**

    ```bash
    pdm run python -c "from brew_oracle.knowledge.pdf_kb import ingest_pdfs; ingest_pdfs()"
    # Saída esperada: OK: 589 pontos na coleção 'brew_books'.
    ```

8. **Rodar o agente (CLI)**

    ```bash
    pdm run python -m brew_oracle.core.run
    ```

---

## 🧠 Orquestrador (com referências)

O agente líder (Gemini) consulta a base e cita as fontes quando usa dados específicos.

```python
self.agent = Agent(
  name="BrewingOrchestrator",
  model=Gemini(id="gemini-2.0-flash", api_key=...),
  knowledge=kb,                      # PDFKnowledgeBase
  search_knowledge=True,
  add_references=True,
  markdown=True,
  instructions=(
    "Responda usando SOMENTE a base; "
    "cite arquivo/seção/página em cada afirmação factualmente específica."
  ),
)
```

---

## 🔍 Busca + Rerank (opcional, recomendado)

Para melhorar a precisão: buscar `TOP_K` alto no Qdrant e reranquear com `cross-encoder/ms-marco-MiniLM-L-6-v2`.

```bash
pdm run python -m brew_oracle.scripts.query_with_rerank
```

Fluxo: embedding da query → Qdrant (`top_k=25–30`) → Rerank com CrossEncoder → mostra top-N com fonte/página/trecho → produção: passar os top-N reranqueados como contexto ao agente.

---

## 🔧 Tuning de chunking

Config padrão recomendada (boa relação custo/qualidade):

- `chunk_size`: 2000
- `overlap`: 300
- `separators`: `\n\n`, `\n`, `.`, `;`, `:`, `\t`
- `reader`: `PDFReader(chunk=False)` → quem corta é o chunker (ex.: `RecursiveChunking`)

Menos pontos → buscas mais rápidas; chunks maiores → contexto mais coeso.
Para algo mais "cirúrgico", use 1000/150 e considere reranking.

---

## 🧪 Perguntas úteis para testar

- "Perfil de água para NEIPA (sulfato/cloreto)?"
- "Como evitar *chill haze*?"
- "Mash típico de Vienna Lager (temp/tempo)?"

Aumente `TOP_K` (25–30) para bases médias e ative rerank para respostas mais precisas.

---

## 🧯 Troubleshooting

- **ModuleNotFoundError: brew_oracle**
  - Rode via módulo e garanta `PYTHONPATH=src` nos scripts do PDM.
- **Nada é ingerido (0 pontos)**
  - Geralmente é schema do Qdrant. Garanta `vectors_config` compatível com `EMBEDDER_DIM` e evite nomes divergentes de vetor (o projeto cria `content_vec` por padrão).
- **HTTP 429 na Hugging Face**
  - Use o modelo local em `./models/all-MiniLM-L6-v2` (ver passo 4).
- **Gemini sem responder**
  - Verifique `GOOGLE_API_KEY` no `.env`.
  - Teste rápido:

    ```bash
    pdm run python -c "from agno.models.google import Gemini; print(Gemini(id='gemini-2.0-flash', api_key='...').response([{'role':'user','content':'oi'}]).content)"
    ```

---

## 🗺️ Roadmap

- Consulta híbrida (denso + BM25) com fusion scoring
- Rerank integrado ao orquestrador
- Agente de Receitas (BeerXML → JSON → Qdrant)
- UI (Streamlit / Discord / Telegram)
- Limpeza de PDF (anti-rodapé, des-hifenização)

---

## 🤝 Contribuindo

Pull requests são bem-vindos! Para mudanças maiores, abra uma issue descrevendo a proposta.

---

## 📄 Licença

Apache 2.0 — veja [LICENSE](LICENSE).

