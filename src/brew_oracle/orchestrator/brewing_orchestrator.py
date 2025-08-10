# src/brew_oracle/orchestrator/brewing_orchestrator.py
from agno.agent import Agent
from agno.models.google import Gemini

from brew_oracle.knowledge.beerxml_kb import build_recipe_kb
from brew_oracle.knowledge.pdf_kb import build_pdf_kb
from brew_oracle.utils.config import Settings


class BrewingOrchestrator:
    def __init__(
        self,
        kb=None,
        model=None,
        *,
        rerank: bool = False,
        rerank_model_id: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        rerank_model_kwargs: dict | None = None,
        hybrid: bool = False,
        pdf_weight: float = 1.0,
        recipe_weight: float = 1.0,
    ) -> None:
        self.pdf_kb = build_pdf_kb(hybrid=hybrid)
        self.recipe_kb = build_recipe_kb(hybrid=hybrid)
        s = Settings()
        self.model = model or Gemini(id=s.MODEL_ID, api_key=s.GOOGLE_API_KEY)

        self.rerank = rerank
        self.pdf_weight = pdf_weight
        self.recipe_weight = recipe_weight
        if self.rerank:
            from sentence_transformers import CrossEncoder

            self._cross_encoder = CrossEncoder(rerank_model_id, **(rerank_model_kwargs or {}))

        def _combined_search(query: str, *args, **kwargs):
            pdf_docs = self.pdf_kb.search(query, *args, **kwargs)
            recipe_docs = self.recipe_kb.search(query, *args, **kwargs)

            # Normalize metadata
            for doc in [*pdf_docs, *recipe_docs]:
                meta = getattr(doc, "meta_data", None)
                if not isinstance(meta, dict):
                    meta = getattr(doc, "metadata", None)
                if not isinstance(meta, dict):
                    meta = {}
                setattr(doc, "meta_data", meta)

            docs_with_origin = [(doc, "pdf") for doc in pdf_docs] + [
                (doc, "recipe") for doc in recipe_docs
            ]

            seen: set[tuple[str, int]] = set()
            deduped: list[tuple[object, str]] = []
            for doc, origin in docs_with_origin:
                meta = getattr(doc, "meta_data", {})
                source = meta.get("source")
                page = meta.get("page")
                key = (source, page)
                if source is not None and page is not None:
                    if key in seen:
                        continue
                    seen.add(key)
                deduped.append((doc, origin))

            if self.rerank:
                pairs = [
                    (query, getattr(doc, "content", getattr(doc, "text", "")))
                    for doc, _ in deduped
                ]
                scores = self._cross_encoder.predict(pairs)
                weights = [
                    self.pdf_weight if origin == "pdf" else self.recipe_weight
                    for _, origin in deduped
                ]
                weighted_scores = [s * w for s, w in zip(scores, weights, strict=False)]
                reranked_docs = [
                    doc
                    for (_, (doc, _)) in sorted(
                        zip(weighted_scores, deduped, strict=False),
                        key=lambda x: x[0],
                        reverse=True,
                    )
                ]
                return reranked_docs

            return [doc for doc, _ in deduped]

        self.agent = Agent(
            name="BrewingOrchestrator",
            model=self.model,
            knowledge=self.pdf_kb,  # Initial knowledge base, will be overridden by search_knowledge
            search_knowledge=_combined_search,  # type: ignore
            add_references=True,
            markdown=True,
            show_tool_calls=True,
            instructions="\n".join(
                [
                    (
                        "Você é o líder de um time de especialistas em cerveja artesanal. "
                        "TAREFA: responder objetivamente as perguntas do usuário citando"
                    ),
                    "quando julgar necessário.",
                    "- Comece com um parágrafo curto (resumo).",
                    "- Depois detalhe a resposta aprofundando sobre o assunto.",
                    "- Se precisar, formate em a resposta em tópicos, números, listas.",
                    "- Use unidades métricas (°C, L, g).",
                    "- Não invente; se não houver evidência clara, diga que falta dado.",
                    "- Adote um tom amigável, bem humorado e didático.",
                    "- Seja explicativo em tudo que fizer.",
                    "- Você pode usar emojis e resposta formatada para facilitar a leitura.",
                ]
            ),
        )

    def ask(self, question: str) -> str:
        resp = self.agent.run(question)
        print()
        self.agent.print_response(question, stream=True)
        return getattr(resp, "content", str(resp))

    def ask_with_refs(self, question: str):
        resp = self.agent.run(question)
        text = getattr(resp, "content", str(resp))
        refs = getattr(resp, "references", [])
        return text, refs
