import argparse

from brew_oracle.orchestrator.brewing_orchestrator import BrewingOrchestrator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Reordena os resultados da busca com CrossEncoder",
    )
    args = parser.parse_args()

    agent = BrewingOrchestrator(rerank=args.rerank)
    print("Digite uma pergunta (ou 'exit' para sair):")
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        text, refs = agent.ask_with_refs(question)
        print(text)
        if refs:
            print("\nReferências:")
            for ref in refs:
                print(f"- {ref}")
    print("Até logo!")


if __name__ == "__main__":
    main()
