from app.orchestration.tool_contracts import BaseTool, ToolDefinition


class DuplicateToolError(ValueError):
    """Raised when a tool name is registered more than once."""


class UnknownToolError(KeyError):
    """Raised when a requested tool is not registered."""


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        name = tool.definition.name
        if name in self._tools:
            raise DuplicateToolError(f"Tool is already registered: {name}")
        self._tools[name] = tool

    def get(self, name: str) -> BaseTool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise UnknownToolError(f"Unknown tool: {name}") from exc

    def definitions(self) -> list[ToolDefinition]:
        return [tool.definition for tool in self._tools.values()]
