from brew_oracle.orchestrator.brewing_orchestrator import BrewingOrchestrator

def main():
    agent = BrewingOrchestrator()
    agent.ask("O que você tem em sua base de conhecimento?")

if __name__ == "__main__":
    main()