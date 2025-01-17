from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

@dataclass
class Message:
    role: str  # 'user', 'assistant', or 'system'
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    pipeline_results: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'role': self.role,
            'content': self.content,
            'timestamp': self.timestamp.isoformat(),
            'pipeline_results': self.pipeline_results,
            'metadata': self.metadata
        }

class ChatThread:
    def __init__(self, thread_id: str):
        self.thread_id = thread_id
        self.messages: List[Message] = []
        self.context: Dict[str, Any] = {}
    
    def add_message(self, role: str, content: str, pipeline_results: Optional[Dict[str, Any]] = None, metadata: Dict[str, Any] = None) -> Message:
        """Add a new message to the thread."""
        message = Message(
            role=role,
            content=content,
            pipeline_results=pipeline_results,
            metadata=metadata or {}
        )
        self.messages.append(message)
        return message
    
    def get_context_window(self, max_messages: int = 10) -> List[Dict[str, Any]]:
        """Get the recent message history with their associated pipeline results."""
        return [msg.to_dict() for msg in self.messages[-max_messages:]]
    
    def get_pipeline_context(self) -> List[Dict[str, Any]]:
        """Get all pipeline results from the conversation history."""
        return [
            msg.pipeline_results 
            for msg in self.messages 
            if msg.pipeline_results is not None
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the thread to a dictionary format."""
        return {
            'thread_id': self.thread_id,
            'messages': [msg.to_dict() for msg in self.messages],
            'context': self.context
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatThread':
        """Create a ChatThread instance from a dictionary."""
        thread = cls(data['thread_id'])
        thread.context = data.get('context', {})
        
        for msg_data in data['messages']:
            timestamp = datetime.fromisoformat(msg_data['timestamp'])
            message = Message(
                role=msg_data['role'],
                content=msg_data['content'],
                timestamp=timestamp,
                pipeline_results=msg_data.get('pipeline_results'),
                metadata=msg_data.get('metadata', {})
            )
            thread.messages.append(message)
        
        return thread

class ChatThreadManager:
    def __init__(self):
        self.threads: Dict[str, ChatThread] = {}
    
    def create_thread(self, thread_id: str) -> ChatThread:
        """Create a new chat thread."""
        if thread_id in self.threads:
            raise ValueError(f"Thread {thread_id} already exists")
        
        thread = ChatThread(thread_id)
        self.threads[thread_id] = thread
        return thread
    
    def get_thread(self, thread_id: str) -> ChatThread:
        """Get an existing chat thread."""
        if thread_id not in self.threads:
            raise ValueError(f"Thread {thread_id} does not exist")
        return self.threads[thread_id]
    
    def delete_thread(self, thread_id: str):
        """Delete a chat thread."""
        if thread_id in self.threads:
            del self.threads[thread_id]
    
    def save_threads(self, filepath: str):
        """Save all threads to a JSON file."""
        data = {
            thread_id: thread.to_dict()
            for thread_id, thread in self.threads.items()
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_threads(self, filepath: str):
        """Load threads from a JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.threads = {
            thread_id: ChatThread.from_dict(thread_data)
            for thread_id, thread_data in data.items()
        }
