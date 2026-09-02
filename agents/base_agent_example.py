
from agents.base_agent import BaseAgent

class AgentExample(BaseAgent):

    def __init__(self, model, tools=None):
        super().__init__(
            name="AGENT_NAME",
            model=model,
            tools=tools, # it is an optional param, so if udn u can just leave it out
            # if more than one tool, pass in a dict of tools
            system_prompt="""
use prompt engineering + your creativity :) define roles, tone, tasks etc.
"""
        )

    # Defining tool functions
    async def run(self, text):
        tool_1 = self.get_tool("glossary_lookup")
        glossary_context = tool_1(text)

        prompt = f"""

{text}

Relevant terminology:
{glossary_context}
"""

        return await self.ask_model(prompt)