"""
Chainlit Curator Interface
===========================

Integration module for Natural Language Curator Interface with Chainlit UI.
Enables chatbot-style curator commands in the Chainlit web interface.

Features:
- Parse natural language curator commands
- Execute library searches and metadata enrichment
- Display results in Chainlit UI
- Support for author searches, topic research, recommendations
- Real-time processing with streaming results

Usage:
    Add to your Chainlit app.py:
    
    from chainlit_curator_interface import setup_curator_interface
    setup_curator_interface(app)

Author: Xoe-NovAi Team
Last Updated: 2026-01-03
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

try:
    import chainlit as cl
    CHAINLIT_AVAILABLE = True
except ImportError:
    CHAINLIT_AVAILABLE = False
    cl = None

from app.XNAi_rag_app.library_api_integrations import (
    NLCuratorInterface,
    LibraryEnrichmentEngine,
    LibraryAPIConfig
)

logger = logging.getLogger(__name__)


class ChainlitCuratorInterface:
    """Chainlit integration for Natural Language Curator Interface."""
    
    def __init__(self):
        """Initialize curator interface for Chainlit."""
        self.engine = LibraryEnrichmentEngine(config=LibraryAPIConfig(enable_cache=True))
        self.curator = NLCuratorInterface(self.engine)
        self.chat_history = []
    
    async def process_curator_message(self, message: str) -> Dict[str, Any]:
        """
        Process curator command from user message.
        
        Args:
            message: User input in natural language
            
        Returns:
            Dict with command results and metadata
        """
        logger.info(f"Processing curator command: {message}")
        
        try:
            # Parse and execute command
            result = self.curator.process_user_input(message)
            
            # Store in chat history
            self.chat_history.append({
                "timestamp": datetime.now().isoformat(),
                "user_input": message,
                "command_type": result.get("command_type"),
                "success": result.get("success"),
                "results_count": result.get("results_count", 0)
            })
            
            return result
        except Exception as e:
            logger.error(f"Curator processing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to process curator command"
            }
    
    def format_results_for_chainlit(self, result: Dict[str, Any]) -> str:
        """Format curator results for Chainlit UI display."""
        if not result.get("success"):
            return f"❌ **Error**: {result.get('error', 'Unknown error')}\n\n{result.get('message', '')}"
        
        message = f"### {result.get('message', 'Results')}\n\n"
        
        # Add results table
        command_type = result.get("command_type")
        results = result.get("results", result.get("recommendations", []))
        
        if results:
            message += f"**Found {len(results)} results:**\n\n"
            for i, item in enumerate(results[:10], 1):  # Show top 10
                title = item.get("title", "Unknown")
                authors = ", ".join(item.get("authors", [])) if item.get("authors") else "Unknown author"
                confidence = item.get("enrichment_confidence", 0)
                
                if command_type == "get_recommendations":
                    rank = item.get("recommendation_rank", i)
                    message += f"**#{rank}. {title}**\n"
                    message += f"   📝 by {authors}\n"
                    message += f"   💡 {item.get('recommendation_reason', 'Relevant resource')}\n"
                    message += f"   ⭐ Confidence: {confidence:.2%}\n\n"
                else:
                    message += f"**{i}. {title}**\n"
                    message += f"   👤 {authors}\n"
                    
                    if item.get("publication_date"):
                        message += f"   📅 {item.get('publication_date')}\n"
                    if item.get("publisher"):
                        message += f"   🏢 {item.get('publisher')}\n"
                    if item.get("subjects"):
                        subjects = ", ".join(item.get("subjects", [])[:3])
                        message += f"   🏷️ {subjects}\n"
                    if item.get("dewey_decimal"):
                        message += f"   📚 Dewey: {item.get('dewey_decimal')}\n"
                    
                    message += f"   ⭐ Confidence: {confidence:.2%}\n\n"
        
        # Add parsing information
        if result.get("parsing_confidence"):
            message += f"\n---\n"
            message += f"*Detected Intent: {result.get('detected_intent')}*\n"
            message += f"*Parsing Confidence: {result.get('parsing_confidence'):.2%}*\n"
        
        return message


# Global curator instance
_curator_instance: Optional[ChainlitCuratorInterface] = None


def get_curator_interface() -> ChainlitCuratorInterface:
    """Get or create global curator interface instance."""
    global _curator_instance
    if _curator_instance is None:
        _curator_instance = ChainlitCuratorInterface()
    return _curator_instance


def setup_curator_interface(app):
    """
    Setup curator interface in Chainlit app.
    
    Add this to your Chainlit app.py:
    
    import chainlit as cl
    from chainlit_curator_interface import setup_curator_interface
    
    @cl.on_chat_start
    async def start():
        setup_curator_interface(cl)
    
    @cl.on_message
    async def main(message: cl.Message):
        ...
    """
    if not CHAINLIT_AVAILABLE:
        logger.error("Chainlit not available - curator interface cannot be initialized")
        return
    
    # Initialize curator instance
    curator = get_curator_interface()
    logger.info("✓ Curator interface initialized for Chainlit")


async def process_curator_command(message_text: str) -> str:
    """
    Process curator command and return formatted response.
    
    Use in Chainlit message handler:
    
    @cl.on_message
    async def handle_message(message: cl.Message):
        if "find" in message.content.lower() or "research" in message.content.lower():
            response = await process_curator_command(message.content)
            await cl.Message(response).send()
    """
    curator = get_curator_interface()
    result = await curator.process_curator_message(message_text)
    formatted = curator.format_results_for_chainlit(result)
    return formatted


# ============================================================================
# CHAINLIT MESSAGE HANDLERS (If Chainlit available)
# ============================================================================

if CHAINLIT_AVAILABLE:
    
    @cl.on_chat_start
    async def start():
        """Initialize chat session."""
        curator = get_curator_interface()
        
        welcome_message = """
# 📚 Xoe-NovAi Curator Assistant

Welcome! I'm your library curator assistant. You can ask me to:

**Examples:**
- "Find all works by Plato"
- "Research quantum mechanics and give me top 10 recommendations"
- "Locate books on philosophy"
- "Show me all science fiction novels"
- "What are the best resources on machine learning?"

Just type your request in natural language and I'll search across 7 major library APIs!
        """
        await cl.Message(welcome_message).send()
        logger.info("Chat session started")
    
    @cl.on_message
    async def handle_curator_message(message: cl.Message):
        """Handle user messages - detect curator commands."""
        user_input = message.content.strip()
        
        # Check if this is a curator command
        curator_keywords = [
            "find", "locate", "search", "research", "recommend", "suggest",
            "book", "author", "works", "by", "on", "about", "top",
            "curate", "collection", "show", "list", "discover"
        ]
        
        is_curator_command = any(kw in user_input.lower() for kw in curator_keywords)
        
        if is_curator_command:
            # Show thinking indicator
            msg = cl.Message("")
            msg.status = "⏳ Searching libraries..."
            await msg.send()
            
            try:
                # Process curator command
                curator = get_curator_interface()
                result = await curator.process_curator_message(user_input)
                formatted_response = curator.format_results_for_chainlit(result)
                
                # Update message with results
                msg.content = formatted_response
                msg.status = "✓ Complete"
                await msg.update()
                
            except Exception as e:
                logger.error(f"Error processing curator command: {e}")
                msg.content = f"❌ Error processing your request: {str(e)}"
                msg.status = "✗ Error"
                await msg.update()
        else:
            # Regular chat message - could be handled by RAG system
            response = f"I'm specialized in library and book curation. Try asking about books, authors, topics, or recommendations!"
            await cl.Message(response).send()
    
    @cl.on_session_end
    async def end():
        """End chat session."""
        curator = get_curator_instance()
        if curator:
            logger.info(f"Chat session ended - Processed {len(curator.chat_history)} curator commands")


# ============================================================================
# EXAMPLE USAGE WITHOUT CHAINLIT
# ============================================================================

def demo_curator_interface():
    """Demo the curator interface without Chainlit."""
    print("\n" + "="*80)
    print("NATURAL LANGUAGE CURATOR INTERFACE DEMO")
    print("="*80 + "\n")
    
    # Initialize curator
    curator = ChainlitCuratorInterface()
    
    # Test commands
    test_commands = [
        "Find all works by Plato",
        "Research books on quantum mechanics and give me your top 10 recommendations to add to my library",
        "Locate and download scientific papers on AI",
        "Show me all philosophy books",
        "What are the best resources on machine learning?"
    ]
    
    for command in test_commands:
        print(f"\n👤 User: {command}")
        print("-" * 80)
        
        import asyncio
        result = asyncio.run(curator.process_curator_message(command))
        
        if result.get("success"):
            formatted = curator.format_results_for_chainlit(result)
            print(formatted)
        else:
            print(f"❌ Error: {result.get('error')}")
        
        print("-" * 80)
    
    print("\n" + "="*80)
    print("DEMO COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    # Run demo if script executed directly
    demo_curator_interface()
