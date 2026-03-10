from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from auth import get_current_user, get_db
from db import ChatSession, ChatMessage, User


router = APIRouter(prefix="/chat", tags=["Chat"])


class ChatSessionCreate(BaseModel):
    document_id: str
    title: Optional[str] = None


class ChatSessionOut(BaseModel):
    id: int
    document_id: str
    title: Optional[str]
    is_archived: bool

    class Config:
        from_attributes = True


class ChatMessageOut(BaseModel):
    id: int
    role: str
    content: str
    metadata: Optional[dict] = Field(default=None, alias="message_metadata")

    class Config:
        from_attributes = True
        validate_by_name = True


class ChatSessionDetail(ChatSessionOut):
    messages: List[ChatMessageOut]


@router.get("/sessions", response_model=List[ChatSessionOut])
def list_sessions(
    document_id: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    q = db.query(ChatSession).filter(
        ChatSession.user_id == current_user.id,
        ChatSession.is_archived.is_(False),
    )
    if document_id:
        q = q.filter(ChatSession.document_id == document_id)
    sessions = q.order_by(ChatSession.updated_at.desc()).all()
    return sessions


@router.post("/sessions", response_model=ChatSessionOut, status_code=status.HTTP_201_CREATED)
def create_session(
    payload: ChatSessionCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    session = ChatSession(
        user_id=current_user.id,
        document_id=payload.document_id,
        title=payload.title,
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def _get_user_session_or_404(
    db: Session, current_user: User, session_id: int
) -> ChatSession:
    session = (
        db.query(ChatSession)
        .filter(
            ChatSession.id == session_id,
            ChatSession.user_id == current_user.id,
            ChatSession.is_archived.is_(False),
        )
        .first()
    )
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    return session


@router.get("/sessions/{session_id}", response_model=ChatSessionDetail)
def get_session(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    session = _get_user_session_or_404(db, current_user, session_id)
    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session.id)
        .order_by(ChatMessage.created_at.asc())
        .all()
    )
    # Convert ORM objects to response schema (Pydantic does not auto-convert nested ORM lists)
    messages_out = [
        ChatMessageOut(
            id=m.id,
            role=m.role,
            content=m.content,
            message_metadata=m.message_metadata,
        )
        for m in messages
    ]
    return ChatSessionDetail(
        id=session.id,
        document_id=session.document_id,
        title=session.title,
        is_archived=session.is_archived,
        messages=messages_out,
    )


@router.get("/sessions/{session_id}/messages", response_model=List[ChatMessageOut])
def list_session_messages(
    session_id: int,
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    session = _get_user_session_or_404(db, current_user, session_id)
    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session.id)
        .order_by(ChatMessage.created_at.asc())
        .limit(limit)
        .all()
    )
    return messages


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_session(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    session = _get_user_session_or_404(db, current_user, session_id)
    session.is_archived = True
    db.add(session)
    db.commit()
    return

