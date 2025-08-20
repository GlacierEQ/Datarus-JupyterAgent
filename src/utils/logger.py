"""Console logger for Datarus Pipeline"""

from rich.console import Console
from rich.syntax import Syntax
from rich.panel import Panel


class ConsoleLogger:
    """Enhanced console logger with rich formatting"""
    
    def __init__(self):
        self.console = Console()
        self.step_count = 0

    def print(self, *args, **kwargs):
        """Direct console print wrapper"""
        self.console.print(*args, **kwargs)

    def print_step(self, title: str, content: str, style: str = "blue"):
        self.step_count += 1
        self.console.print(f"[bold {style}]Step {self.step_count}: {title}[/]")
        self.console.print(Panel(content, border_style=style))

    def print_code(self, code: str, language: str = "python"):
        syntax = Syntax(code, language, theme="monokai", line_numbers=True)
        self.console.print(Panel(syntax, border_style="green"))

    def print_error(self, error: str):
        self.console.print(Panel(error, title="❌ Error", border_style="red"))

    def print_result(self, result: str):
        self.console.print(Panel(result, title="📊 Result", border_style="green"))

    def print_status(self, message: str, status: str = "info"):
        styles = {"info": "blue", "success": "green", "warning": "yellow", "error": "red"}
        style = styles.get(status, "white")
        self.console.print(f"[{style}]{message}[/]")

    def print_thought(self, thought: str):
        self.console.print(Panel(thought, title="💭 Thought", border_style="cyan"))

    def print_llm_response(self, response: str):
        self.console.print(Panel(response, title="🤖 LLM Response", border_style="cyan"))

    def print_correction(self, message: str):
        self.console.print(Panel(message, title="🔄 Correction Attempt", border_style="yellow"))
    
    def print_panel(self, content: str, style: str = "blue"):
        """Print content in a panel"""
        self.console.print(Panel(content, border_style=style))
