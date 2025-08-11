import logging

import discord

from brew_oracle.orchestrator.brewing_orchestrator import BrewingOrchestrator
from brew_oracle.utils.config import Settings

logger = logging.getLogger(__name__)


class DiscordBot(discord.Client):
    """Discord client that routes messages to the BrewingOrchestrator."""

    def __init__(self, orchestrator: BrewingOrchestrator, **options):
        super().__init__(**options)
        self.orchestrator = orchestrator

    async def on_ready(self) -> None:  # pragma: no cover - just a log
        logger.info("Logged in as %s", self.user)

    async def on_message(self, message: discord.Message) -> None:
        """Handle incoming messages from Discord."""
        if message.author == self.user or getattr(message.author, "bot", False):
            return
        content = message.content.strip()
        if not content:
            return

        text, refs = self.orchestrator.ask_with_refs(content)
        response = text
        if refs:
            response += "\n\n" + "\n".join(str(r) for r in refs)
        await message.channel.send(response)


def main() -> None:
    """Entrypoint used by the PDM script."""
    settings = Settings()
    token = settings.DISCORD_TOKEN
    if not token:
        raise RuntimeError("DISCORD_TOKEN is not configured")
    intents = discord.Intents.default()
    intents.message_content = True
    # Use hybrid search with reranking, mirroring CLI flags `--hybrid --rerank`
    orchestrator = BrewingOrchestrator(hybrid=True, rerank=True)
    bot = DiscordBot(orchestrator, intents=intents)
    bot.run(token)


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
