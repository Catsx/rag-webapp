import os
import sys
import click
from typing import List, Optional
from pathlib import Path
import json
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.markdown import Markdown
import logging

# Add src directory to path for imports
src_path = str(Path(__file__).parent)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from rag_pipeline import RAGPipeline

# Configure rich console
console = Console()

# Configure logging
logging.basicConfig(level=logging.WARNING)  # Reduce log noise in CLI


class RAGCLIInterface:
    """CLI interface for the RAG application."""
    
    def __init__(self):
        self.rag_pipeline = None
        self.config_file = Path.home() / ".rag_config.json"
        self.default_config = {
            "model_name": "gpt-3.5-turbo",
            "temperature": 0.1,
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "max_sources": 10
        }
        self.config = self.load_config()
    
    def load_config(self) -> dict:
        """Load configuration from file or use defaults."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    # Merge with defaults
                    merged_config = self.default_config.copy()
                    merged_config.update(config)
                    return merged_config
            except Exception as e:
                console.print(f"[yellow]Aviso: Não foi possível carregar o ficheiro de configuração: {e}[/yellow]")
        
        return self.default_config.copy()
    
    def save_config(self):
        """Save current configuration to file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            console.print(f"[red]Erro ao guardar configuração: {e}[/red]")
    
    def initialize_pipeline(self, api_key: Optional[str] = None) -> bool:
        """Initialize the RAG pipeline."""
        try:
            # Check for API key
            openai_key = api_key or os.getenv("OPENAI_API_KEY")
            if not openai_key:
                console.print("[red]Chave API do OpenAI não encontrada![/red]")
                console.print("Por favor, define a variável de ambiente OPENAI_API_KEY ou fornece-a via --api-key")
                return False
            
            # Initialize pipeline
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("A inicializar pipeline RAG...", total=None)
                
                self.rag_pipeline = RAGPipeline(
                    openai_api_key=openai_key,
                    model_name=self.config["model_name"],
                    temperature=self.config["temperature"],
                    chunk_size=self.config["chunk_size"],
                    chunk_overlap=self.config["chunk_overlap"],
                    embedding_model=self.config["embedding_model"]
                )
                
                progress.update(task, completed=True)
            
            console.print("[green]✓ Pipeline RAG inicializada com sucesso![/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]Erro ao inicializar pipeline: {e}[/red]")
            return False
    
    def display_welcome(self):
        """Display welcome message and instructions."""
        welcome_text = """
# 🤖 Aplicação RAG - Faz upload de documentos e insere URLs e faz as tuas perguntas!
        """
        
        console.print(Panel(
            Markdown(welcome_text),
            title="RAG - Catarina Cardoso",
            border_style="blue"
        ))
    
    def cmd_index(self, sources: List[str], save: bool = True) -> bool:
        """Index documents from sources."""
        if not self.rag_pipeline:
            console.print("[red]Pipeline não inicializada![/red]")
            return False
        
        if not sources:
            console.print("[yellow]Nenhuma fonte fornecida![/yellow]")
            return False
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("A indexar documentos...", total=None)
                
                results = self.rag_pipeline.index_documents(sources, save_index=save)
                
                progress.update(task, completed=True)
            
            # Display results
            table = Table(title="Resultados da Indexação")
            table.add_column("Métrica", style="cyan")
            table.add_column("Valor", style="green")
            
            table.add_row("Estado", results["status"])
            table.add_row("Documentos Carregados", str(results["documents_loaded"]))
            table.add_row("Chunks Criados", str(results["chunks_created"]))
            table.add_row("Fontes Indexadas", str(results["sources_indexed"]))
            
            console.print(table)
            console.print("[green]✓ Documentos indexados com sucesso![/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]Erro ao indexar documentos: {e}[/red]")
            return False
    
    def cmd_query(self, question: str, show_sources: bool = True) -> Optional[dict]:
        """Process a single query."""
        if not self.rag_pipeline or not self.rag_pipeline.is_indexed:
            console.print("[red]Nenhum documento indexado! Por favor, executa primeiro o comando 'index'.[/red]")
            return None
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("A processar pergunta...", total=None)
                
                response = self.rag_pipeline.query(question, return_sources=show_sources)
                
                progress.update(task, completed=True)
            
            # Display answer
            console.print(Panel(
                response["answer"],
                title=f"Resposta para: {question}",
                border_style="green"
            ))
            
            # Display sources if available
            if show_sources and "sources" in response:
                console.print("\n[bold]Fontes:[/bold]")
                for i, source in enumerate(response["sources"][:3], 1):  # Show top 3 sources
                    source_text = source["content"]
                    metadata = source["metadata"]
                    source_info = f"Fonte {i}: {metadata.get('source', 'Desconhecida')}"
                    console.print(f"[dim]{source_info}[/dim]")
                    console.print(f"[italic]{source_text}[/italic]\n")
            
            # Display token usage
            if "token_usage" in response:
                usage = response["token_usage"]
                console.print(f"[dim]Tokens utilizados: {usage['total_tokens']} | Custo: ${usage['total_cost']:.4f}[/dim]")
            
            return response
            
        except Exception as e:
            console.print(f"[red]Erro ao processar pergunta: {e}[/red]")
            return None
    
    def cmd_chat(self):
        """Start interactive chat mode."""
        if not self.rag_pipeline or not self.rag_pipeline.is_indexed:
            console.print("[red]Nenhum documento indexado! Por favor, executa primeiro o comando 'index'.[/red]")
            return
        
        console.print(Panel(
            "🗣️ Modo de Conversa Interativo\n\nFaz perguntas sobre os teus documentos indexados.\nEscreve 'exit' para voltar ao menu principal.",
            title="Modo Conversa",
            border_style="blue"
        ))
        
        while True:
            try:
                question = Prompt.ask("\n[bold blue]A tua pergunta[/bold blue]")
                
                if question.lower() in ['exit', 'quit', 'back']:
                    break
                
                if not question.strip():
                    continue
                
                self.cmd_query(question)
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Modo de conversa interrompido.[/yellow]")
                break
            except EOFError:
                break
    
    def cmd_config(self):
        """Configure RAG pipeline settings."""
        console.print(Panel(
            "⚙️ Definições de Configuração",
            title="Configuração",
            border_style="blue"
        ))
        
        # Display current config
        table = Table(title="Configuração Atual")
        table.add_column("Definição", style="cyan")
        table.add_column("Valor", style="green")
        
        for key, value in self.config.items():
            table.add_row(key, str(value))
        
        console.print(table)
        
        # Ask if user wants to modify
        if Confirm.ask("\nQueres modificar alguma definição?"):
            self._modify_config()
    
    def _modify_config(self):
        """Interactive configuration modification."""
        config_options = {
            "1": ("model_name", "Nome do modelo OpenAI (ex: gpt-3.5-turbo, gpt-4)"),
            "2": ("temperature", "Temperatura para geração (0.0-2.0)"),
            "3": ("chunk_size", "Tamanho dos chunks de texto em caracteres"),
            "4": ("chunk_overlap", "Sobreposição entre chunks"),
            "5": ("embedding_model", "Nome do modelo de embeddings"),
        }
        
        console.print("\n[bold]Definições disponíveis:[/bold]")
        for key, (setting, description) in config_options.items():
            console.print(f"{key}. {setting}: {description}")
        
        choice = Prompt.ask("Seleciona a definição a modificar (1-5)", choices=list(config_options.keys()))
        setting_name, description = config_options[choice]
        
        current_value = self.config[setting_name]
        console.print(f"\nValor atual: {current_value}")
        
        new_value = Prompt.ask(f"Introduz o novo valor para {setting_name}")
        
        # Type conversion
        if setting_name in ["temperature"]:
            try:
                new_value = float(new_value)
            except ValueError:
                console.print("[red]Valor decimal inválido![/red]")
                return
        elif setting_name in ["chunk_size", "chunk_overlap"]:
            try:
                new_value = int(new_value)
            except ValueError:
                console.print("[red]Valor inteiro inválido![/red]")
                return
        
        self.config[setting_name] = new_value
        self.save_config()
        console.print(f"[green]✓ {setting_name} atualizada para {new_value}[/green]")
        
        # Ask to reinitialize pipeline
        if Confirm.ask("Reinicializar pipeline com as novas definições?"):
            self.initialize_pipeline()
    
    def cmd_status(self):
        """Show pipeline status and statistics."""
        if not self.rag_pipeline:
            console.print("[red]Pipeline não inicializada![/red]")
            return
        
        stats = self.rag_pipeline.get_pipeline_stats()
        
        # Pipeline status
        status_table = Table(title="Estado da Pipeline")
        status_table.add_column("Componente", style="cyan")
        status_table.add_column("Estado", style="green")
        
        pipeline_status = stats["pipeline_status"]
        status_table.add_row("Indexado", "✓" if pipeline_status["is_indexed"] else "✗")
        status_table.add_row("LLM Disponível", "✓" if pipeline_status["llm_available"] else "✗")
        status_table.add_row("Cadeia Q&A Pronta", "✓" if pipeline_status["qa_chain_ready"] else "✗")
        
        console.print(status_table)
        
        # Usage statistics
        if pipeline_status["is_indexed"]:
            usage_stats = stats["usage_stats"]
            usage_table = Table(title="Estatísticas de Uso")
            usage_table.add_column("Métrica", style="cyan")
            usage_table.add_column("Valor", style="green")
            
            usage_table.add_row("Documentos Carregados", str(usage_stats["documents_loaded"]))
            usage_table.add_row("Chunks Criados", str(usage_stats["chunks_created"]))
            usage_table.add_row("Perguntas Processadas", str(usage_stats["queries_processed"]))
            usage_table.add_row("Última Indexação", usage_stats.get("last_indexing", "Nunca"))
            
            console.print(usage_table)
    
    def run_interactive(self, api_key: Optional[str] = None):
        """Run interactive CLI mode."""
        self.display_welcome()
        
        # Initialize pipeline
        if not self.initialize_pipeline(api_key):
            return
        
        # Main command loop
        while True:
            try:
                console.print()
                command = Prompt.ask(
                    "[bold green]RAG>[/bold green]",
                    choices=["index", "query", "chat", "config", "status", "help", "exit"],
                    show_choices=False
                )
                
                if command == "exit":
                    console.print("[blue]Adeus! 👋[/blue]")
                    break
                elif command == "help":
                    self._show_help()
                elif command == "index":
                    sources = self._get_sources_input()
                    if sources:
                        self.cmd_index(sources)
                elif command == "query":
                    question = Prompt.ask("Introduz a tua pergunta")
                    self.cmd_query(question)
                elif command == "chat":
                    self.cmd_chat()
                elif command == "config":
                    self.cmd_config()
                elif command == "status":
                    self.cmd_status()
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Usa 'exit' para sair.[/yellow]")
            except EOFError:
                break
    
    def _get_sources_input(self) -> List[str]:
        """Get sources input from user."""
        console.print("\n[bold]Introduz as fontes a indexar:[/bold]")
        console.print("- Caminhos de ficheiros (ex: /caminho/para/documento.pdf)")
        console.print("- URLs (ex: https://exemplo.com)")
        console.print("- Introduz uma por linha, linha vazia para terminar")
        
        sources = []
        while True:
            source = Prompt.ask(f"Fonte {len(sources) + 1} (ou pressiona Enter para terminar)", default="")
            if not source:
                break
            sources.append(source)
        
        return sources
    
    def _show_help(self):
        """Show help information."""
        help_text = """
## Comandos Disponíveis:

- **index**: Carrega e indexa documentos de ficheiros ou URLs
- **query**: Faz uma pergunta sobre documentos indexados
- **chat**: Inicia modo de conversa interativo para múltiplas perguntas
- **config**: Visualiza e modifica definições de configuração
- **status**: Mostra estado da pipeline e estatísticas de uso
- **help**: Mostra esta mensagem de ajuda
- **exit**: Sai da aplicação

## Dicas:
- Certifica-te de definir a tua chave API do OpenAI antes de começar
- Indexa documentos antes de fazer perguntas
- Usa o modo conversa para múltiplas perguntas relacionadas
- Verifica o estado para ver a saúde da pipeline e estatísticas de uso
        """
        
        console.print(Panel(
            Markdown(help_text),
            title="Ajuda",
            border_style="blue"
        ))


# Click CLI commands for non-interactive usage
@click.group()
@click.option('--api-key', envvar='OPENAI_API_KEY', help='Chave API do OpenAI')
@click.pass_context
def cli(ctx, api_key):
    """Aplicação RAG - Catarina Cardoso"""
    ctx.ensure_object(dict)
    ctx.obj['api_key'] = api_key


@cli.command()
@click.argument('sources', nargs=-1, required=True)
@click.option('--save/--no-save', default=True, help='Guarda índice no disco')
@click.pass_context
def index(ctx, sources, save):
    """Indexa documentos de fontes (ficheiros ou URLs)."""
    interface = RAGCLIInterface()
    if interface.initialize_pipeline(ctx.obj['api_key']):
        interface.cmd_index(list(sources), save)


@cli.command()
@click.argument('question')
@click.option('--no-sources', is_flag=True, help='Oculta documentos fonte')
@click.pass_context
def query(ctx, question, no_sources):
    """Faz uma pergunta sobre documentos indexados."""
    interface = RAGCLIInterface()
    if interface.initialize_pipeline(ctx.obj['api_key']):
        # Try to load existing index
        if interface.rag_pipeline.load_existing_index():
            interface.cmd_query(question, show_sources=not no_sources)
        else:
            console.print("[red]Nenhum documento indexado encontrado! Por favor, executa primeiro o comando 'index'.[/red]")


@cli.command()
@click.pass_context
def interactive(ctx):
    """Inicia modo interativo."""
    interface = RAGCLIInterface()
    interface.run_interactive(ctx.obj['api_key'])


@cli.command()
@click.pass_context
def status(ctx):
    """Mostra estado da pipeline."""
    interface = RAGCLIInterface()
    if interface.initialize_pipeline(ctx.obj['api_key']):
        interface.cmd_status()


if __name__ == "__main__":
    cli()
