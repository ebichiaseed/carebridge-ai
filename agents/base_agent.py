'''
All agents inherit from this class.
This takes in the agent's name, model, tools if hv and system prompt to run the model.
'''

from abc import ABC, abstractmethod
from models.base_model import BaseModel


class BaseAgent(ABC):

    def __init__(
        self,
        name: str,
        model: BaseModel,
        tools: dict | None = None,
        system_prompt: str = ""
    ):
        self.name = name
        self.model = model
        self.tools = tools or {}
        self.system_prompt = system_prompt

    def get_tool(self, tool_name: str):
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not available")
        return self.tools[tool_name]

    async def ask_model(self, prompt: str, **kwargs):
        """
        Shared function available to every agent.
        """
        full_prompt = self._build_prompt(prompt)

        return await self.model.generate(
            prompt=full_prompt,
            **kwargs
        )

    def _build_prompt(self, user_input: str) -> str:
        if not self.system_prompt:
            return user_input

        return f"""
            {self.system_prompt}

            INPUT:
            {user_input}
            """

    @abstractmethod
    async def run(self, input_data):
        """
        Each agent MUST implement its own run() logic.
        """
        pass