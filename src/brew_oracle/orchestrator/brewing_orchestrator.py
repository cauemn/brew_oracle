# src/brew_oracle/orchestrator/brewing_orchestrator.py
from agno.agent import Agent
from agno.models.google import Gemini
from brew_oracle.knowledge.pdf_kb import build_pdf_kb
from brew_oracle.utils.config import Settings

class BrewingOrchestrator:
    def __init__(self, kb=None, model=None):

        self.kb = kb or build_pdf_kb()
        s = Settings()
        self.model = model or Gemini(id="gemini-2.0-flash", api_key=s.GOOGLE_API_KEY)

        self.agent = Agent(
            name="BrewingOrchestrator",
            model=self.model,
            knowledge=self.kb,
            search_knowledge=True,     
            add_references=True,       
            markdown=True,
            show_tool_calls=True,  
            instructions=(
                "Você é o líder de um time de especialistas em cerveja artesanal. "
                "TAREFA: responder objetivamente as perguntas do usuário citando \n"
                "quando julgar necessário.\n"
                "- Comece com um parágrafo curto (resumo).\n"
                "- Depois detalhe a resposta aprofundando sobre o assunto.\n"
                "- Se precisar, formate em a resposta em tópicos, números, listas.\n"
                "- Use unidades métricas (°C, L, g). Não invente; se não houver evidência clara, diga que falta dado.\n"
                "- Adote um tom amigável, bem humorado e didático, seja explicativo em tudo que fizer.\n"
                "- Você pode usar emojis e respostas formatas para facilitar a leitura do usuário."
            ),
        )

    def ask(self, question: str) -> str:
        resp = self.agent.run(question)
        self.agent.print_response(question)
        return getattr(resp, "content", str(resp))

    def ask_with_refs(self, question: str):
        resp = self.agent.run(question)
        text = getattr(resp, "content", str(resp))
        refs = getattr(resp, "references", [])
        return text, refs
