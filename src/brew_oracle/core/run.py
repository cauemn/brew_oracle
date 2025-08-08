import argparse

from brew_oracle.orchestrator.brewing_orchestrator import BrewingOrchestrator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Reordena os resultados da busca com CrossEncoder",
    )
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Combina busca densa e BM25 via fusion scoring",
    )
    args = parser.parse_args()

    agent = BrewingOrchestrator(rerank=args.rerank, hybrid=args.hybrid)
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
