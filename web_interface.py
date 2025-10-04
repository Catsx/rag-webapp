#!/usr/bin/env python3
"""
Web Interface for RAG - Catarina Cardoso
Creates a user-friendly web interface using Gradio
"""

import sys
from pathlib import Path
import gradio as gr
import os

# Add src directory to path
src_path = str(Path(__file__).parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from rag_pipeline import RAGPipeline

class RAGWebInterface:
    def __init__(self):
        self.rag = None
        self.indexed_sources = []
        
    def initialize_rag(self, api_key):
        """Initialize the RAG pipeline with API key."""
        try:
            if api_key and api_key.strip():
                self.rag = RAGPipeline(
                    openai_api_key=api_key.strip(),
                    temperature=0.1,
                    chunk_size=1000,
                    chunk_overlap=200
                )
                return "✅ Pipeline RAG inicializada com sucesso!"
            else:
                self.rag = RAGPipeline(openai_api_key=None)
                return "⚠️ Pipeline RAG inicializada sem chave OpenAI (apenas pesquisa por similaridade)"
        except Exception as e:
            return f"❌ Erro ao inicializar RAG: {str(e)}"
    
    def add_documents(self, files, urls_text):
        """Add documents and URLs to the RAG system."""
        if not self.rag:
            return "❌ Por favor, inicia primeiro o RAG!", ""
        
        sources = []
        
        # Handle uploaded files
        if files:
            for file in files:
                sources.append(file.name)
        
        # Handle URLs
        if urls_text and urls_text.strip():
            urls = [url.strip() for url in urls_text.split('\n') if url.strip()]
            sources.extend(urls)
        
        if not sources:
            return "❌ Nenhuma fonte fornecida!", ""
        
        try:
            results = self.rag.index_documents(sources)
            self.indexed_sources.extend(sources)
            
            success_msg = f"""✅ Documentos indexados com sucesso!
            
📊 Resultados:
• Documentos carregados: {results['documents_loaded']}
• Chunks criados: {results['chunks_created']}
• Fontes processadas: {results['sources_indexed']}

📚 Fontes atualmente indexadas:
{chr(10).join([f"• {source}" for source in self.indexed_sources])}
            """
            
            return success_msg, ""
            
        except Exception as e:
            return f"❌ Erro ao indexar documentos: {str(e)}", ""
    
    def ask_question(self, question):
        """Ask a question to the RAG system."""
        if not self.rag:
            return "❌ Por favor, inicia primeiro o RAG!"
        
        if not self.rag.is_indexed:
            return "❌ Nenhum documento indexado! Por favor, adiciona primeiro os documentos."
        
        if not question or not question.strip():
            return "❌ Por favor, faz uma pergunta!"
        
        try:
            # Always try similarity search first (works without OpenAI)
            docs = self.rag.get_similar_documents(question.strip(), k=3)
            
            if not docs:
                return "❌ Nenhum conteúdo relevante encontrado para a tua pergunta."
            
            # Try full RAG query if LLM is available and has quota
            if self.rag.llm:
                try:
                    response = self.rag.query(question.strip())
                    
                    answer = f"""🤖 **Resposta IA:**
{response['answer']}

📚 **Fontes utilizadas:**
"""
                    if 'sources' in response:
                        for i, source in enumerate(response['sources'][:3], 1):
                            content_preview = source['content'][:150] + "..." if len(source['content']) > 150 else source['content']
                            source_info = source['metadata'].get('source', 'Desconhecida')
                            answer += f"\n{i}. **{source_info}**\n   {content_preview}\n"
                    
                    if 'token_usage' in response:
                        usage = response['token_usage']
                        answer += f"\n💰 **Uso:** {usage['total_tokens']} tokens (${usage['total_cost']:.4f})"
                    
                    return answer
                    
                except Exception as llm_error:
                    # If LLM fails (quota issue), fall back to similarity search
                    if "quota" in str(llm_error).lower() or "429" in str(llm_error):
                        pass  # Continue to similarity search below
                    else:
                        return f"❌ Erro LLM: {str(llm_error)}"
            
            # Similarity search results (always works)
            answer = f"""🔍 **Conteúdo Relevante Encontrado:**

**A Tua Pergunta:** {question.strip()}

"""
            
            for i, doc in enumerate(docs, 1):
                content_preview = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                source_info = doc.metadata.get('source', 'Desconhecida')
                chunk_info = f"(Chunk {doc.metadata.get('chunk_id', 'N/A')})"
                
                answer += f"""**📄 Resultado {i}: {source_info}** {chunk_info}
{content_preview}

---

"""
            
            answer += """💡 **Como obter respostas geradas por IA:**
• Adiciona uma chave API do OpenAI com quota disponível
• Ou usa os resultados de similaridade acima para encontrar a tua resposta

✅ **O teu sistema RAG está a funcionar perfeitamente para recuperação de documentos!**"""
            
            return answer
                
        except Exception as e:
            return f"❌ Erro ao processar pergunta: {str(e)}"
    
    def get_status(self):
        """Get RAG system status."""
        if not self.rag:
            return "❌ RAG não inicializado"
        
        stats = self.rag.get_pipeline_stats()
        
        status = f"""📊 **Estado do Sistema RAG**

🔧 **Configuração:**
• Modelo: {stats['configuration']['model_name']}
• Temperatura: {stats['configuration']['temperature']}
• Tamanho do chunk: {stats['configuration']['chunk_size']}
• Modelo de embeddings: {stats['configuration']['embedding_model']}

📈 **Estatísticas de Uso:**
• Documentos carregados: {stats['usage_stats']['documents_loaded']}
• Chunks criados: {stats['usage_stats']['chunks_created']}
• Perguntas processadas: {stats['usage_stats']['queries_processed']}

🏃 **Estado da Pipeline:**
• Indexado: {'✅' if stats['pipeline_status']['is_indexed'] else '❌'}
• LLM Disponível: {'✅' if stats['pipeline_status']['llm_available'] else '❌'}
• Cadeia Q&A Pronta: {'✅' if stats['pipeline_status']['qa_chain_ready'] else '❌'}

📚 **Fontes Indexadas:**
{chr(10).join([f"• {source}" for source in self.indexed_sources]) if self.indexed_sources else "Ainda não há fontes indexadas"}
        """
        
        return status

def create_interface():
    """Create the Gradio web interface."""
    
    rag_interface = RAGWebInterface()
    
    with gr.Blocks(title="RAG - Catarina Cardoso", theme=gr.themes.Soft()) as app:
        
        gr.Markdown("""
        # 🤖 RAG - Catarina Cardoso
        ### Catarina Cardoso
        
        Faz upload de documentos, adiciona links web e faz perguntas sobre o teu conteúdo!
        """)
        
        with gr.Tab("🚀 Configuração"):
            gr.Markdown("### Passo 1: Iniciar Sistema RAG")
            
            api_key_input = gr.Textbox(
                label="Chave API OpenAI (Opcional)", 
                placeholder="sk-proj-...", 
                type="password",
                info="Deixa vazio para usar apenas pesquisa por similaridade"
            )
            
            init_btn = gr.Button("Inicializar Sistema RAG", variant="primary")
            init_output = gr.Textbox(label="Estado da Inicialização", interactive=False)
            
            init_btn.click(
                rag_interface.initialize_rag,
                inputs=[api_key_input],
                outputs=[init_output]
            )
        
        with gr.Tab("📚 Adicionar Documentos"):
            gr.Markdown("### Passo 2: Adiciona os Teus Documentos e Fontes")
            
            with gr.Row():
                with gr.Column():
                    file_upload = gr.Files(
                        label="Upload de Ficheiros (PDF, DOCX, TXT)", 
                        file_types=[".pdf", ".docx", ".txt", ".md"]
                    )
                    
                with gr.Column():
                    urls_input = gr.Textbox(
                        label="URLs Web (uma por linha)",
                        placeholder="https://exemplo.com/artigo1\nhttps://exemplo.com/artigo2",
                        lines=5
                    )
            
            add_btn = gr.Button("Adicionar Documentos ao RAG", variant="primary")
            add_output = gr.Textbox(label="Resultados da Indexação", lines=10, interactive=False)
            
            add_btn.click(
                rag_interface.add_documents,
                inputs=[file_upload, urls_input],
                outputs=[add_output, urls_input]
            )
        
        with gr.Tab("💬 Fazer Perguntas"):
            gr.Markdown("### Passo 3: Faz Perguntas Sobre os Teus Documentos")
            
            question_input = gr.Textbox(
                label="A Tua Pergunta",
                placeholder="Qual é o tópico principal dos documentos?",
                lines=2
            )
            
            ask_btn = gr.Button("Fazer Pergunta", variant="primary")
            answer_output = gr.Textbox(label="Resposta", lines=15, interactive=False)
            
            ask_btn.click(
                rag_interface.ask_question,
                inputs=[question_input],
                outputs=[answer_output]
            )
            
            # Quick question examples
            gr.Markdown("### 💡 Perguntas de Exemplo:")
            example_questions = [
                "Qual é o tópico principal?",
                "Resume os pontos-chave",
                "Quais são os benefícios mencionados?",
                "Quem são as principais pessoas ou organizações mencionadas?"
            ]
            
            for question in example_questions:
                example_btn = gr.Button(f"📝 {question}", size="sm")
                example_btn.click(
                    lambda q=question: q,
                    outputs=[question_input]
                )
        
        with gr.Tab("📊 Estado"):
            gr.Markdown("### Estado do Sistema e Estatísticas")
            
            status_btn = gr.Button("Atualizar Estado", variant="secondary")
            status_output = gr.Textbox(label="Estado do Sistema RAG", lines=20, interactive=False)
            
            status_btn.click(
                rag_interface.get_status,
                outputs=[status_output]
            )
        
        gr.Markdown("""
        ---
        ### 📖 Como Usar:
        1. **Configuração**: Inicia o sistema RAG (com ou sem chave API OpenAI)
        2. **Adicionar Documentos**: Faz upload de ficheiros ou cola URLs web
        3. **Fazer Perguntas**: Faz perguntas sobre os teus documentos
        4. **Verificar Estado**: Monitoriza a saúde do sistema e estatísticas de uso
        
        ### 💡 Dicas:
        - Sem chave API OpenAI: Vais obter resultados de pesquisa por similaridade
        - Com chave API OpenAI: Vais obter respostas completas geradas por IA
        - Formatos suportados: PDF, DOCX, TXT, Markdown, URLs Web
        """)
    
    return app

if __name__ == "__main__":
    # Check if gradio is installed
    try:
        import gradio
    except ImportError:
        print("A instalar Gradio para interface web...")
        os.system("pip3 install gradio")
        import gradio as gr
    
    print("🚀 A iniciar Interface Web RAG...")
    print("📱 A interface vai abrir no teu navegador web")
    print("🔗 Também podes aceder em: http://localhost:7864")
    
    app = create_interface()
    app.launch(
        server_name="0.0.0.0",
        server_port=7864,
        share=False,
        show_error=True,
        quiet=False
    )
