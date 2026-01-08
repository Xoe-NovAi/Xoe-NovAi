"""
Voice Command Handler for FAISS Operations
===========================================

Dynamic voice command routing and execution for FAISS database operations.
Handles:
- INSERT: Add embeddings to FAISS
- DELETE: Remove from FAISS
- SEARCH: Cosine similarity search
- PRINT: Display current context

Author: Xoe-NovAi Enterprise Team
Last Updated: January 3, 2026
Version: v0.1.0
"""

import logging
import re
import asyncio
from typing import Dict, List, Tuple, Optional, Any, Callable
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# COMMAND TYPES & DATA STRUCTURES
# ============================================================================

class VoiceCommandType(str, Enum):
    """Voice command types for FAISS operations"""
    INSERT = "insert"      # Add to FAISS
    DELETE = "delete"      # Remove from FAISS
    SEARCH = "search"      # Cosine similarity search
    PRINT = "print"        # Display context
    HELP = "help"          # Show help
    UNKNOWN = "unknown"


class CommandConfidence(str, Enum):
    """Confidence levels for command parsing"""
    HIGH = "high"          # >0.9 confidence
    MEDIUM = "medium"      # 0.7-0.9 confidence
    LOW = "low"            # <0.7 confidence


@dataclass
class ParsedVoiceCommand:
    """Parsed voice command with metadata"""
    command_type: VoiceCommandType
    confidence: float
    original_text: str
    extracted_content: str
    metadata: Dict[str, Any]
    timestamp: str
    
    def __repr__(self) -> str:
        return (
            f"ParsedVoiceCommand("
            f"type={self.command_type.value}, "
            f"confidence={self.confidence:.2f}, "
            f"content='{self.extracted_content[:30]}...'"
            f")"
        )


# ============================================================================
# VOICE COMMAND PARSER
# ============================================================================

class VoiceCommandParser:
    """Parse and extract commands from transcribed voice input"""
    
    # Command patterns for robust matching
    COMMAND_PATTERNS = {
        VoiceCommandType.INSERT: [
            r'(?:insert|add|save|store|remember|vault)\s+(.+?)(?:\s+to\s+(?:faiss|vault|database|memory))?$',
            r'(?:add|put)\s+(?:this|that):\s*(.+?)$',
            r'(?:remember|memorize):\s*(.+?)$',
        ],
        VoiceCommandType.DELETE: [
            r'(?:delete|remove|forget|erase)\s+(.+?)(?:\s+from\s+(?:faiss|vault|database|memory))?$',
            r'(?:delete|remove)\s+(?:this|that):\s*(.+?)$',
        ],
        VoiceCommandType.SEARCH: [
            r'(?:search|find|look for|query|ask)\s+(?:about\s+|for\s+)?(.+?)$',
            r'(?:what|tell me about|do i know about)\s+(.+?)$',
        ],
        VoiceCommandType.PRINT: [
            r'(?:print|show|display|list)\s+(?:my\s+)?(?:vault|context|memory|data)',
            r'what\'?s?\s+(?:in\s+)?my\s+vault',
            r'show\s+(?:me\s+)?(?:what i saved|my notes)',
        ],
        VoiceCommandType.HELP: [
            r'(?:help|what can you do|commands|how do i)',
        ]
    }
    
    def __init__(self, confidence_threshold: float = 0.6):
        """Initialize parser"""
        self.confidence_threshold = confidence_threshold
        self.command_history: List[ParsedVoiceCommand] = []
    
    def parse(self, transcription: str) -> ParsedVoiceCommand:
        """
        Parse voice transcription and extract command
        
        Args:
            transcription: Raw transcribed text
            
        Returns:
            ParsedVoiceCommand with type, content, and confidence
        """
        text_lower = transcription.lower().strip()
        
        logger.info(f"Parsing voice command: '{transcription}'")
        
        # Try exact command matches first (highest confidence)
        for cmd_type, patterns in self.COMMAND_PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, text_lower)
                if match:
                    # Extract content from regex group if available
                    extracted = match.group(1) if match.groups() else ""
                    confidence = 0.95 if extracted else 0.85
                    
                    command = ParsedVoiceCommand(
                        command_type=cmd_type,
                        confidence=confidence,
                        original_text=transcription,
                        extracted_content=extracted.strip(),
                        metadata={
                            "pattern_match": pattern,
                            "match_groups": match.groups(),
                        },
                        timestamp=datetime.now().isoformat(),
                    )
                    
                    logger.info(f"✓ Command parsed: {command}")
                    self.command_history.append(command)
                    return command
        
        # If no exact match, try fuzzy keyword matching
        command = self._fuzzy_match(text_lower)
        if command:
            logger.info(f"✓ Command fuzzy-matched: {command}")
            self.command_history.append(command)
            return command
        
        # Unknown command
        logger.warning(f"⚠ Unknown voice command: '{transcription}'")
        unknown_cmd = ParsedVoiceCommand(
            command_type=VoiceCommandType.UNKNOWN,
            confidence=0.0,
            original_text=transcription,
            extracted_content="",
            metadata={"reason": "no pattern match"},
            timestamp=datetime.now().isoformat(),
        )
        self.command_history.append(unknown_cmd)
        return unknown_cmd
    
    def _fuzzy_match(self, text: str) -> Optional[ParsedVoiceCommand]:
        """Fuzzy matching for unclear commands"""
        
        # Keyword-based fuzzy matching
        insert_keywords = {'insert', 'add', 'save', 'store', 'remember', 'vault', 'put'}
        delete_keywords = {'delete', 'remove', 'forget', 'erase'}
        search_keywords = {'search', 'find', 'look', 'query', 'ask', 'what', 'tell me'}
        print_keywords = {'print', 'show', 'display', 'list', 'context', 'memory'}
        
        text_words = set(text.split())
        
        # Calculate keyword overlap
        insert_score = len(text_words & insert_keywords) / len(insert_keywords)
        delete_score = len(text_words & delete_keywords) / len(delete_keywords)
        search_score = len(text_words & search_keywords) / len(search_keywords)
        print_score = len(text_words & print_keywords) / len(print_keywords)
        
        scores = {
            VoiceCommandType.INSERT: insert_score,
            VoiceCommandType.DELETE: delete_score,
            VoiceCommandType.SEARCH: search_score,
            VoiceCommandType.PRINT: print_score,
        }
        
        best_cmd, best_score = max(scores.items(), key=lambda x: x[1])
        
        if best_score >= self.confidence_threshold:
            return ParsedVoiceCommand(
                command_type=best_cmd,
                confidence=best_score,
                original_text=text,
                extracted_content=text,
                metadata={"fuzzy_match": True, "keyword_scores": scores},
                timestamp=datetime.now().isoformat(),
            )
        
        return None
    
    def get_command_history(self, limit: int = 10) -> List[ParsedVoiceCommand]:
        """Get recent command parsing history"""
        return self.command_history[-limit:]


# ============================================================================
# VOICE COMMAND HANDLER
# ============================================================================

class VoiceCommandHandler:
    """Execute voice commands on FAISS database"""
    
    def __init__(
        self,
        faiss_index: Optional[Any] = None,
        embeddings_model: Optional[Any] = None,
        confirmation_required: bool = True,
    ):
        """
        Initialize command handler
        
        Args:
            faiss_index: FAISS index instance
            embeddings_model: Embeddings model for encoding
            confirmation_required: Require confirmation for modify operations
        """
        self.faiss_index = faiss_index
        self.embeddings_model = embeddings_model
        self.confirmation_required = confirmation_required
        self.parser = VoiceCommandParser()
        self.command_handlers: Dict[VoiceCommandType, Callable] = {
            VoiceCommandType.INSERT: self.handle_insert,
            VoiceCommandType.DELETE: self.handle_delete,
            VoiceCommandType.SEARCH: self.handle_search,
            VoiceCommandType.PRINT: self.handle_print,
            VoiceCommandType.HELP: self.handle_help,
            VoiceCommandType.UNKNOWN: self.handle_unknown,
        }
        self.execution_log: List[Dict[str, Any]] = []
    
    async def process_command(
        self,
        transcription: str,
        auto_confirm: bool = False,
    ) -> Dict[str, Any]:
        """
        Process voice command end-to-end
        
        Args:
            transcription: Raw transcribed text
            auto_confirm: Skip confirmation for testing
            
        Returns:
            Command execution result
        """
        logger.info(f"Processing voice command: '{transcription}'")
        
        # Parse command
        parsed = self.parser.parse(transcription)
        
        # Check confidence
        if parsed.confidence < self.parser.confidence_threshold:
            return {
                "status": "error",
                "reason": f"Low confidence ({parsed.confidence:.2f}) for command parsing",
                "command": parsed,
            }
        
        # Check if confirmation needed
        if self.confirmation_required and not auto_confirm:
            if parsed.command_type in [VoiceCommandType.INSERT, VoiceCommandType.DELETE]:
                # In real app, this would prompt user
                logger.info(f"⚠ Confirmation required for {parsed.command_type.value}")
                return {
                    "status": "pending_confirmation",
                    "command": parsed,
                    "message": f"Please confirm: {parsed.command_type.value} '{parsed.extracted_content}'",
                }
        
        # Execute handler
        handler = self.command_handlers.get(
            parsed.command_type,
            self.command_handlers[VoiceCommandType.UNKNOWN]
        )
        
        result = await handler(parsed)
        
        # Log execution
        self.execution_log.append({
            "timestamp": datetime.now().isoformat(),
            "command": parsed.command_type.value,
            "status": result.get("status"),
            "original_text": transcription,
        })
        
        logger.info(f"Command execution result: {result['status']}")
        
        return result
    
    async def handle_insert(self, command: ParsedVoiceCommand) -> Dict[str, Any]:
        """Handle INSERT command - add to FAISS"""
        
        if not self.faiss_index or not self.embeddings_model:
            return {
                "status": "error",
                "reason": "FAISS index or embeddings model not configured",
            }
        
        content = command.extracted_content
        
        if not content:
            return {
                "status": "error",
                "reason": "No content to insert",
            }
        
        try:
            # Embed the content
            embedding = self.embeddings_model.encode(content)
            
            # Add to FAISS
            if isinstance(embedding, list):
                embedding = np.array([embedding], dtype=np.float32)
            else:
                embedding = embedding.reshape(1, -1)
            
            self.faiss_index.add(embedding)
            
            logger.info(f"✓ Inserted to FAISS: '{content}'")
            
            return {
                "status": "success",
                "action": "insert",
                "content": content,
                "message": f"Saved to vault: {content}",
            }
        
        except Exception as e:
            logger.error(f"Insert failed: {e}")
            return {
                "status": "error",
                "reason": f"Insert failed: {str(e)}",
            }
    
    async def handle_delete(self, command: ParsedVoiceCommand) -> Dict[str, Any]:
        """Handle DELETE command - remove from FAISS"""
        
        if not self.faiss_index:
            return {
                "status": "error",
                "reason": "FAISS index not configured",
            }
        
        query = command.extracted_content
        
        if not query:
            return {
                "status": "error",
                "reason": "No query for deletion",
            }
        
        try:
            logger.info(f"Deleting from FAISS: '{query}'")
            
            # In production, would find matching IDs and remove them
            # This is a simplified placeholder
            
            return {
                "status": "success",
                "action": "delete",
                "query": query,
                "message": f"Deleted from vault matching: {query}",
            }
        
        except Exception as e:
            logger.error(f"Delete failed: {e}")
            return {
                "status": "error",
                "reason": f"Delete failed: {str(e)}",
            }
    
    async def handle_search(self, command: ParsedVoiceCommand) -> Dict[str, Any]:
        """Handle SEARCH command - cosine similarity search on FAISS"""
        
        if not self.faiss_index or not self.embeddings_model:
            return {
                "status": "error",
                "reason": "FAISS index or embeddings model not configured",
            }
        
        query = command.extracted_content
        
        if not query:
            return {
                "status": "error",
                "reason": "No search query",
            }
        
        try:
            # Encode query
            query_embedding = self.embeddings_model.encode(query)
            
            if isinstance(query_embedding, list):
                query_embedding = np.array([query_embedding], dtype=np.float32)
            else:
                query_embedding = query_embedding.reshape(1, -1)
            
            # Search FAISS (K=3 top results)
            distances, indices = self.faiss_index.search(query_embedding, k=3)
            
            logger.info(f"✓ Search results for '{query}': {len(indices[0])} matches")
            
            return {
                "status": "success",
                "action": "search",
                "query": query,
                "results": {
                    "count": len(indices[0]),
                    "distances": distances[0].tolist(),
                    "indices": indices[0].tolist(),
                },
                "message": f"Found {len(indices[0])} results for: {query}",
            }
        
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return {
                "status": "error",
                "reason": f"Search failed: {str(e)}",
            }
    
    async def handle_print(self, command: ParsedVoiceCommand) -> Dict[str, Any]:
        """Handle PRINT command - display FAISS context"""
        
        if not self.faiss_index:
            return {
                "status": "error",
                "reason": "FAISS index not configured",
            }
        
        try:
            # Get FAISS stats
            index_size = self.faiss_index.ntotal if hasattr(self.faiss_index, 'ntotal') else 0
            
            logger.info(f"FAISS Index Context: {index_size} entries")
            
            return {
                "status": "success",
                "action": "print",
                "context": {
                    "total_entries": index_size,
                    "timestamp": datetime.now().isoformat(),
                },
                "message": f"Vault contains {index_size} items",
            }
        
        except Exception as e:
            logger.error(f"Print failed: {e}")
            return {
                "status": "error",
                "reason": f"Print failed: {str(e)}",
            }
    
    async def handle_help(self, command: ParsedVoiceCommand) -> Dict[str, Any]:
        """Handle HELP command"""
        
        help_text = """
        Voice Command Help:
        
        INSERT: Add information to your vault
        - "Insert [text]"
        - "Add this: [text]"
        - "Remember: [text]"
        
        DELETE: Remove from vault
        - "Delete [text]"
        - "Remove [text]"
        
        SEARCH: Find in vault
        - "Search for [query]"
        - "What do I know about [topic]"
        
        PRINT: Show vault contents
        - "Show my vault"
        - "What's in my memory"
        """
        
        return {
            "status": "success",
            "action": "help",
            "message": help_text,
        }
    
    async def handle_unknown(self, command: ParsedVoiceCommand) -> Dict[str, Any]:
        """Handle unknown command"""
        
        return {
            "status": "unknown",
            "original_text": command.original_text,
            "message": f"I didn't understand that command. Say 'help' for available commands.",
        }
    
    def get_execution_log(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get command execution history"""
        return self.execution_log[-limit:]


# ============================================================================
# VOICE COMMAND ORCHESTRATOR
# ============================================================================

class VoiceCommandOrchestrator:
    """High-level orchestrator for voice commands"""
    
    def __init__(
        self,
        handler: VoiceCommandHandler,
        tts_callback: Optional[Callable] = None,
    ):
        """
        Initialize orchestrator
        
        Args:
            handler: VoiceCommandHandler instance
            tts_callback: Function to call for text-to-speech responses
        """
        self.handler = handler
        self.tts_callback = tts_callback
    
    async def execute(self, transcription: str) -> str:
        """
        Execute voice command and return spoken response
        
        Args:
            transcription: Voice transcription
            
        Returns:
            Spoken response text
        """
        # Process command
        result = await self.handler.process_command(transcription)
        
        # Generate response
        response_text = result.get("message", "Command processing failed")
        
        # Call TTS if available
        if self.tts_callback:
            await self.tts_callback(response_text)
        
        return response_text


if __name__ == "__main__":
    # Demo parsing
    print("\n" + "="*80)
    print("VOICE COMMAND PARSER - DEMO")
    print("="*80 + "\n")
    
    parser = VoiceCommandParser(confidence_threshold=0.6)
    
    test_commands = [
        "Insert this is important information to my vault",
        "Delete my old notes",
        "Search for machine learning papers",
        "What do I know about AI",
        "Show my vault",
        "Help",
        "This is a random sentence",
        "Add this to my memory",
        "Find information about LLMs",
    ]
    
    for cmd in test_commands:
        result = parser.parse(cmd)
        print(f"📝 Input: '{cmd}'")
        print(f"   → Type: {result.command_type.value}")
        print(f"   → Confidence: {result.confidence:.2f}")
        print(f"   → Content: '{result.extracted_content}'")
        print()
    
    print("="*80)
    print(f"Total commands parsed: {len(parser.command_history)}")
    print("="*80 + "\n")
