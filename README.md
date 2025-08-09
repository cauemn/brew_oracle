# 🍺 Brew Oracle

**Brew Oracle** é um agente de IA especializado em cerveja artesanal, treinado a partir de PDFs técnicos.
Ele responde **usando a base de conhecimento indexada**, **cita as fontes** e mantém o tom claro, útil e bem-humorado.

---

## 🔎 Visão Geral

### Stack Principal

- Python 3.13
- Agno (agentes + RAG)
- Qdrant (banco vetorial)
- Sentence Transformers (embeddings)
- Google Gemini (modelo de linguagem)

**Fluxo:** Usuário → Orquestrador (Agno) → Busca combinada no Qdrant (bases `brew_books` e `brew_recipes`) → *(opcional: Rerank)* → Gemini → Resposta **com referências**.

---

## 📁 Estrutura do Projeto

```
.
├─ knowledge/
│  ├─ pdfs/                           # Coloque seus PDFs aqui
│  └─ recipes/                        # Coloque suas receitas BeerXML aqui
├─ src/
│  └─ brew_oracle/
│     ├─ core/
│     │  └─ run.py                    # Ponto de entrada do agente (CLI)
│     ├─ knowledge/
│     │  ├─ pdf_kb.py                 # Construção/ingestão da base de conhecimento de PDFs
│     │  └─ beerxml_kb.py             # Construção/ingestão da base de conhecimento de receitas BeerXML
│     ├─ orchestrator/
│     │  └─ brewing_orchestrator.py   # Agente orquestrador
│     ├─ scripts/
│     │  └─ create_collections.py     # Cria as coleções no Qdrant
│     └─ utils/
│        └─ config.py                 # Configurações (lê .env)
├─ tests/                             # Testes automatizados
├─ .env
├─ pyproject.toml
└─ README.md
```

---

## ⚙️ Pré-requisitos

- Docker (para o Qdrant)
- [PDM](https://pdm.fming.dev/) (gerenciador de pacotes)

---

## 🚀 Instalação e Configuração

1. **Clone e Instale as Dependências**
   
   ```bash
   git clone https://github.com/<seu-usuario>/brew_oracle.git
   cd brew_oracle
   pdm install
   ```

2. **Inicie o Qdrant**
   
   ```bash
   docker run -d -p 6333:6333 \
     -v "$(pwd)/qdrant_storage:/qdrant/storage" \
     qdrant/qdrant:latest
   ```

3. **Configure o `.env`**
    Crie um arquivo `.env` na raiz do projeto:
   
   ```ini
   QDRANT_URL=http://localhost:6333
   QDRANT_COLLECTION=brew_books
   QDRANT_RECIPE_COLLECTION=brew_recipes
   
   PDF_PATH=knowledge/pdfs
   BEERXML_PATH=knowledge/recipes
   
   EMBEDDER_ID=./models/all-MiniLM-L6-v2
   EMBEDDER_DIM=384
   
   TOP_K=25
   
   CHUNK_SIZE=2000
   CHUNK_OVERLAP=300
   NUM_DOCUMENTS=5
   
   GOOGLE_API_KEY=sua_chave_api_do_google
   ```

4. **Crie as Coleções no Qdrant**
   
   ```bash
   pdm run create-collection --hybrid
   pdm run create-recipe-collection --hybrid
   ```

5. **Adicione os PDFs e Receitas**
    Coloque seus arquivos PDF em `knowledge/pdfs/` e seus arquivos BeerXML em `knowledge/recipes/`.

6. **Ingira os PDFs e Receitas**
   
   ```bash
   pdm run ingest-pdfs-hybrid
   pdm run ingest-recipes-hybrid
   ```

---

## 🚀 Executando o Agente

```bash
# Execute o agente
pdm run brew-oracle

# Execute com rerank
pdm run brew-oracle --rerank

# Execute com busca híbrida (denso + BM25)
pdm run brew-oracle --hybrid
```

---

## 🧪 Testes

Para executar os testes automatizados do projeto, utilize o seguinte comando:

```bash
pdm run test
```

---

## 🧠 Orquestrador (com referências)

O agente principal (Gemini) consulta a base de conhecimento e cita as fontes ao usar dados específicos.

```python
self.agent = Agent(
  name="BrewingOrchestrator",
  model=Gemini(id="gemini-1.5-flash", api_key=...), # Note: Changed from gemini-2.0-flash to gemini-1.5-flash
  knowledge=kb,                      # PDFKnowledgeBase
  search_knowledge=True,
  add_references=True,
  markdown=True,
  instructions=(
    "Responda SOMENTE usando a base de conhecimento; "
    "cite o arquivo/seção/página para cada afirmação factualmente específica."
  ),
)
```

---

## 🔍 Busca + Rerank (opcional, recomendado)

Para melhorar a precisão: busque com um `TOP_K` alto no Qdrant e reclassifique com `cross-encoder/ms-marco-MiniLM-L-6-v2`.

Ative diretamente pela CLI:

```bash
pdm run brew-oracle --rerank
```

--- 

## 🔧 Ajuste de Chunking

Configuração padrão recomendada (boa relação custo/qualidade):

- `chunk_size`: 2000
- `overlap`: 300
- `separators`: `\n\n`, `\n`, `.`, `;`, `:`, `\t`

Para resultados mais precisos, use 1000/150 e considere o reranking.

💡 Ajuste `CHUNK_SIZE`, `CHUNK_OVERLAP` e `NUM_DOCUMENTS` no arquivo `.env` para personalizar o chunking e o número de documentos retornados.

--- 

## 🧪 Perguntas Úteis para Testar

- "Perfil de água para NEIPA (sulfato/cloreto)?"
- "Como evitar a turbidez a frio (*chill haze*)?"
- "Mostura típica de uma Vienna Lager (temperatura/tempo)?"

Aumente o `TOP_K` (25–30) para bases de conhecimento médias e ative o rerank para respostas mais precisas.

--- 

## 🧯 Solução de Problemas

- **ModuleNotFoundError: brew_oracle**
  
  - Execute como um módulo e garanta que `PYTHONPATH=src` esteja nos scripts do PDM.

- **Nada é ingerido (0 pontos)**
  
  - Geralmente é um problema de esquema do Qdrant. Garanta que o `vectors_config` é compatível com o `EMBEDDER_DIM` e evite nomes de vetores divergentes.

- **HTTP 429 no Hugging Face**
  
  - Use um modelo local (veja o passo 3 da instalação).

- **Gemini não está respondendo**
  
  - Verifique a `GOOGLE_API_KEY` no `.env`.

--- 

## 🗺️ Roadmap

- Rerank integrado ao orquestrador
- UI (Streamlit / Discord / Telegram)
- Limpeza de PDF (remoção de rodapé, de-hifenização)

--- 

## 🤝 Contribuição

Pull requests são bem-vindos! Para mudanças maiores, por favor, abra uma issue primeiro para discutir o que você gostaria de mudar.
