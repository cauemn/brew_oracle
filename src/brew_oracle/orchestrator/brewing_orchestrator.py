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
    ) -> None:
        self.pdf_kb = build_pdf_kb(hybrid=hybrid)
        self.recipe_kb = build_recipe_kb(hybrid=hybrid)
        s = Settings()
        self.model = model or Gemini(id="gemini-2.0-flash", api_key=s.GOOGLE_API_KEY)

        self.rerank = rerank
        if self.rerank:
            from sentence_transformers import CrossEncoder

            self._cross_encoder = CrossEncoder(rerank_model_id, **(rerank_model_kwargs or {}))

        def _combined_search(query: str, *args, **kwargs):
            pdf_docs = self.pdf_kb.search(query, *args, **kwargs)
            recipe_docs = self.recipe_kb.search(query, *args, **kwargs)
            combined_docs = pdf_docs + recipe_docs

            if self.rerank:
                pairs = [
                    (query, getattr(doc, "content", getattr(doc, "text", "")))
                    for doc in combined_docs
                ]
                scores = self._cross_encoder.predict(pairs)
                reranked_docs = [
                    doc
                    for doc, _ in sorted(
                        zip(combined_docs, scores, strict=False), key=lambda x: x[1], reverse=True
                    )
                ]
                return reranked_docs
            return combined_docs

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
