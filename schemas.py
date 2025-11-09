"""
Database Schemas

Define your MongoDB collection schemas here using Pydantic models.
These schemas are used for data validation in your application.

Each Pydantic model represents a collection in your database.
Model name is converted to lowercase for the collection name:
- User -> "user" collection
- Product -> "product" collection
- BlogPost -> "blogs" collection
"""

from pydantic import BaseModel, Field
from typing import Optional, List

class Document(BaseModel):
    """
    Documents collection schema
    Collection name: "document"
    """
    filename: str = Field(..., description="Original file name")
    content_type: str = Field(..., description="MIME type")
    size: int = Field(..., ge=0, description="File size in bytes")
    title: Optional[str] = Field(None, description="Optional document title")
    num_pages: Optional[int] = Field(None, ge=0, description="Number of pages")
    num_chunks: Optional[int] = Field(None, ge=0, description="Number of text chunks extracted")

class Chunk(BaseModel):
    """
    Chunks collection schema
    Collection name: "chunk"
    """
    document_id: str = Field(..., description="Reference to the parent document id as string")
    index: int = Field(..., ge=0, description="Chunk index in document")
    text: str = Field(..., description="Chunk text content")
    page: Optional[int] = Field(None, ge=0, description="Source page number if known")

class Conversation(BaseModel):
    """
    Conversations collection schema
    Collection name: "conversation"
    """
    document_id: str = Field(..., description="Associated document id as string")
    title: Optional[str] = Field(None, description="Conversation title")

class Message(BaseModel):
    """
    Messages collection schema
    Collection name: "message"
    """
    conversation_id: str = Field(..., description="Parent conversation id as string")
    role: str = Field(..., description="user or assistant")
    content: str = Field(..., description="Message content")

# Example schemas kept for reference but not used in this app
class User(BaseModel):
    name: str
    email: str
    address: str
    age: Optional[int] = Field(None, ge=0, le=120)
    is_active: bool = True

class Product(BaseModel):
    title: str
    description: Optional[str] = None
    price: float = Field(..., ge=0)
    category: str
    in_stock: bool = True
