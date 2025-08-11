import asyncio
import importlib
import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch


class TestDiscordBot(unittest.TestCase):
    def _import_module(self):
        fake_orchestrator_module = MagicMock()
        fake_orchestrator_module.BrewingOrchestrator = MagicMock()

        class FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            def run(self, *args, **kwargs):
                pass

        class FakeIntents:
            def __init__(self):
                self.message_content = False

            @classmethod
            def default(cls):
                return cls()

        fake_discord_module = MagicMock()
        fake_discord_module.Client = FakeClient
        fake_discord_module.Intents = FakeIntents

        fake_settings_module = MagicMock()
        fake_settings_module.Settings = MagicMock(
            return_value=MagicMock(DISCORD_TOKEN="test-token")
        )
        with patch.dict(
            sys.modules,
            {
                "brew_oracle.orchestrator.brewing_orchestrator": fake_orchestrator_module,
                "discord": fake_discord_module,
                "brew_oracle.utils.config": fake_settings_module,
            },
        ):
            module = importlib.import_module("brew_oracle.agents.discord_bot")
        return module, fake_orchestrator_module

    def test_main_initializes_and_runs(self):
        module, fake_orchestrator_module = self._import_module()
        with patch.object(module.DiscordBot, "run") as mock_run:
            with patch.dict(os.environ, {"DISCORD_TOKEN": "test-token"}):
                module.main()
        fake_orchestrator_module.BrewingOrchestrator.assert_called_once_with(
            hybrid=True, rerank=True
        )
        mock_run.assert_called_once_with("test-token")
        sys.modules.pop("brew_oracle.agents.discord_bot", None)

    def test_on_message_forwards_to_orchestrator(self):
        module, _ = self._import_module()
        orchestrator = MagicMock()
        orchestrator.ask_with_refs.return_value = ("answer", [])
        bot = module.DiscordBot.__new__(module.DiscordBot)
        bot.orchestrator = orchestrator
        bot.user = object()

        message = MagicMock()
        message.author = MagicMock(bot=False)
        message.content = "question"
        message.channel.send = AsyncMock()

        asyncio.run(bot.on_message(message))

        orchestrator.ask_with_refs.assert_called_once_with("question")
        message.channel.send.assert_awaited_once_with("answer")
        sys.modules.pop("brew_oracle.agents.discord_bot", None)
