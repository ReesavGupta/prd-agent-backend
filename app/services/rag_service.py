"""
RAG and PRD summarization helpers for chat mode.
"""

from __future__ import annotations

import hashlib
from typing import List, Tuple, Optional

from app.services.ai_service import ai_service
from app.services.cache_service import cache_service
from app.core.config import settings


async def get_prd_summary(project_id: str, user_id: str, prd_markdown: Optional[str]) -> str:
    """Return PRD summary from cache; if missing and prd_markdown provided, compute ephemeral summary and cache it.

    - Persisted summary is cached by save-artifacts as key: prd_summary:{project_id}:{etag}
    - Ephemeral (unsaved) summary is cached by content hash with short TTL
    """
    # Try persisted summary (any etag)
    try:
        if await cache_service.is_connected():
            keys = await cache_service.redis.keys(f"prd_summary:{project_id}:*")
            if keys:
                val = await cache_service.redis.get(keys[-1])
                if val:
                    return val
    except Exception:
        pass

    # Ephemeral summarization
    if prd_markdown:
        try:
            sys = {"role": "system", "content": "Summarize the PRD concisely in 8-12 bullet points. No code fences."}
            usr = {"role": "user", "content": prd_markdown[:18000]}
            resp = await ai_service.generate_response(
                user_id=user_id,
                messages=[sys, usr],
                temperature=0.2,
                max_tokens=800,
                use_cache=False,
            )
            summary_text = (resp.content or "").strip()
            try:
                if await cache_service.is_connected():
                    h = hashlib.md5(prd_markdown.encode("utf-8")).hexdigest()
                    await cache_service.redis.setex(
                        f"prd_summary_ephemeral:{project_id}:{h}",
                        settings.PRD_SUMMARY_EPHEMERAL_TTL_SECONDS,
                        summary_text,
                    )
            except Exception:
                pass
            return summary_text
        except Exception:
            return "(No PRD summary available.)"

    return "(No PRD summary available.)"


async def select_sections(prd_markdown: str, question: str, user_id: str) -> str:
    """Use a lightweight LLM classifier to select up to 3 PRD sections most relevant to the question.
    Fallback to keyword overlap. Returns concatenated sections (~<=2400 chars).
    """
    try:
        lines = prd_markdown.splitlines()
        sections: List[Tuple[str, str]] = []
        current_head: Optional[str] = None
        current_body: List[str] = []
        for ln in lines:
            if ln.strip().startswith("### "):
                if current_head is not None:
                    sections.append((current_head, "\n".join(current_body).strip()))
                    current_body = []
                current_head = ln.strip().lstrip("# ")
            else:
                if current_head is not None:
                    current_body.append(ln)
        if current_head is not None:
            sections.append((current_head, "\n".join(current_body).strip()))

        headings = [h for h, _ in sections]
        chosen_indices: List[int] = []
        if headings:
            try:
                sys = {
                    "role": "system",
                    "content": (
                        "Given a user question and a list of section headings, select at most 3 most relevant sections. "
                        "Return ONLY a minified JSON array of integer indices (0-based), no prose."
                    ),
                }
                heading_list = "\n".join([f"[{i}] {t}" for i, t in enumerate(headings)])
                usr = {"role": "user", "content": f"Question: {question}\nHeadings:\n{heading_list}"}
                resp = await ai_service.generate_response(
                    user_id=user_id,
                    messages=[sys, usr],
                    temperature=0.1,
                    max_tokens=64,
                    use_cache=False,
                )
                import json as _json
                raw = (resp.content or "").strip()
                start = raw.find('[')
                end = raw.rfind(']')
                if start != -1 and end != -1:
                    raw = raw[start : end + 1]
                arr = _json.loads(raw)
                if isinstance(arr, list):
                    for v in arr:
                        try:
                            idx = int(v)
                            if 0 <= idx < len(sections):
                                chosen_indices.append(idx)
                        except Exception:
                            continue
                chosen_indices = chosen_indices[:3]
            except Exception:
                chosen_indices = []

            if not chosen_indices:
                import re as _re
                q = _re.findall(r"[a-zA-Z0-9]+", question.lower())
                qset = set(q)
                scores = []
                for i, (h, b) in enumerate(sections):
                    text = (h + "\n" + (b or "")).lower()
                    toks = set(_re.findall(r"[a-zA-Z0-9]+", text))
                    scores.append((i, len(qset & toks)))
                scores.sort(key=lambda x: x[1], reverse=True)
                chosen_indices = [i for i, s in scores[:3] if s > 0]

            out_blocks: List[str] = []
            total_chars = 0
            for idx in chosen_indices:
                head, body = sections[idx]
                block = f"### {head}\n{body}".strip()
                if total_chars + len(block) > 2400:
                    break
                out_blocks.append(block)
                total_chars += len(block)
            if out_blocks:
                return "\n\n".join(out_blocks)
    except Exception:
        pass
    return ""


async def retrieve_attachment(project_id: str, file_id: str, query: str) -> str:
    """Retrieve top chunks from Pinecone for the given file_id within the project namespace.
    Returns concatenated text (~<=4500 chars). Uses the same embedding/vector configuration as indexing.
    """
    try:
        # Use the shared rag_service vector store to keep embeddings consistent with indexing
        store = rag_service._vector(namespace=str(project_id))
        k = rag_service.config.top_k if hasattr(rag_service, "config") else 6
        docs = None
        try:
            docs = await store.asimilarity_search(query, k=k, filter={"file_id": file_id})  # type: ignore
        except Exception:
            pass
        if docs is None:
            docs = store.similarity_search(query, k=k, filter={"file_id": file_id})
        acc: List[str] = []
        total_chars = 0
        for d in docs or []:
            t = (getattr(d, 'page_content', '') or '').strip()
            if not t:
                continue
            if total_chars + len(t) > 4500:
                break
            acc.append(t)
            total_chars += len(t)
        return "\n\n".join(acc)
    except Exception:
        return ""

import asyncio
import io
import os
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, Iterable, List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
try:
    from langchain_nomic.embeddings import NomicEmbeddings  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    NomicEmbeddings = None  # type: ignore
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

from app.core.config import settings
import logging

logger = logging.getLogger(__name__)


@dataclass
class RAGConfig:
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 6
    openai_model: str = settings.OPENAI_MODEL or "gpt-4o-mini"


class RAGService:
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        # Embeddings and vector store are initialized lazily per call to allow namespacing
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            add_start_index=True,
        )

        # Defer hard failures until first use to avoid boot-time crashes
        self._validated = False
        self._pc = None  # Pinecone client
        # Quiet Pinecone plugin interface logs (we don't use inference plugins)
        try:
            logging.getLogger("pinecone_plugin_interface").setLevel(logging.WARNING)
        except Exception:
            pass

    def _use_nomic(self) -> bool:
        return bool(settings.NOMIC_API_KEY and NomicEmbeddings is not None)

    def _embeddings(self):
        # Prefer Nomic embeddings if available, else OpenAI
        if self._use_nomic():
            return NomicEmbeddings(model="nomic-embed-text-v1.5", nomic_api_key=settings.NOMIC_API_KEY)  # type: ignore
        return OpenAIEmbeddings(model="text-embedding-3-large")

    def _llm(self) -> ChatOpenAI:
        return ChatOpenAI(model=self.config.openai_model, temperature=0.0)

    def _llm_streaming(self) -> ChatOpenAI:
        # Streaming must be enabled via constructor for OpenAI; do not pass `streaming` at call time
        return ChatOpenAI(model=self.config.openai_model, temperature=0.0, streaming=True)

    def _ensure_index(self) -> None:
        # Create Pinecone index if it doesn't exist (serverless default)
        from pinecone import Pinecone, ServerlessSpec
        self._pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        name = settings.PINECONE_INDEX_NAME
        try:
            def _index_exists() -> bool:
                try:
                    # v6: returns iterable of index summaries
                    lst = self._pc.list_indexes()
                    # could be list of objects or dict with 'indexes'
                    if isinstance(lst, (list, tuple)):
                        try:
                            return any(getattr(x, 'name', None) == name for x in lst)
                        except Exception:
                            return any((isinstance(x, dict) and x.get('name') == name) for x in lst)
                    if isinstance(lst, dict) and 'indexes' in lst:
                        return any(i.get('name') == name for i in lst['indexes'])
                except Exception:
                    pass
                # Fallback older client: names() accessor
                try:
                    names = self._pc.list_indexes().names()  # type: ignore[attr-defined]
                    return name in names
                except Exception:
                    return False

            if not _index_exists():
                dimension = 768 if self._use_nomic() else 3072
                self._pc.create_index(
                    name=name,
                    dimension=dimension,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                )
                logger.info(f"Created Pinecone index '{name}' (dim={dimension})")
        except Exception:
            # Best-effort; retrieval may fail later with explicit error
            logger.exception("Failed to ensure Pinecone index exists")

    def _validate(self) -> None:
        if self._validated:
            return
        if not settings.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set")
        if not settings.PINECONE_API_KEY:
            raise RuntimeError("PINECONE_API_KEY is not set")
        self._ensure_index()
        self._validated = True

    def _index(self):
        # Ensure validated and index exists
        self._validate()
        assert self._pc is not None
        name = settings.PINECONE_INDEX_NAME
        try:
            return self._pc.Index(name)
        except Exception:
            # If Index construction fails due to missing index, try create-on-demand
            self._ensure_index()
            return self._pc.Index(name)

    def _vector(self, namespace: str) -> PineconeVectorStore:
        index = self._index()
        return PineconeVectorStore(index=index, embedding=self._embeddings(), namespace=namespace)

    # ---------- Indexing ----------
    async def index_prd(self, project_id: str, prd_markdown: str) -> None:
        if not prd_markdown:
            return
        self._validate()
        # Extract sections by Markdown headings to improve retrieval granularity
        sections: List[Document] = []
        current_title = "Introduction"
        current_buf: List[str] = []
        def flush():
            if current_buf:
                sections.append(Document(page_content="\n".join(current_buf).strip(), metadata={"filename": "prd.md", "source": "prd", "section": current_title}))
        for line in prd_markdown.splitlines():
            if line.lstrip().startswith('#'):
                flush()
                current_title = line.lstrip('#').strip() or current_title
                current_buf = []
            else:
                current_buf.append(line)
        flush()

        splits = self.text_splitter.split_documents(sections)
        store = self._vector(namespace=project_id)
        # add_documents upserts vectors by id internally
        try:
            await store.aadd_documents(splits)
            logger.info(f"RAG indexed PRD into namespace={project_id}: {len(splits)} chunks")
        except Exception:
            logger.exception("Failed to index PRD into Pinecone")

    async def index_upload_bytes(
        self,
        project_id: str,
        filename: str,
        content_type: str,
        data: bytes,
    ) -> None:
        if not data:
            return
        self._validate()

        text = ""
        name_lower = (filename or "").lower()
        ctype = (content_type or "").lower()

        # Convert to markdown/plain text as required
        if ctype == "application/pdf" or name_lower.endswith(".pdf"):
            try:
                from pypdf import PdfReader

                reader = PdfReader(io.BytesIO(data))
                pages = []
                for i, page in enumerate(reader.pages):
                    pages.append(page.extract_text() or "")
                text = "\n\n".join(pages)
            except Exception:
                text = ""
        elif name_lower.endswith(".docx") or ctype in (
            "application/msword",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ):
            try:
                import docx2txt

                text = docx2txt.process(io.BytesIO(data)) or ""
            except Exception:
                text = ""
        else:
            try:
                text = data.decode("utf-8", errors="ignore")
            except Exception:
                text = ""

        if not text:
            return

        # Try to infer lightweight sections for md/txt by headings
        docs: List[Document] = []
        if (filename or "").lower().endswith('.md'):
            title = "Introduction"
            buf: List[str] = []
            def flush_md():
                if buf:
                    docs.append(Document(page_content="\n".join(buf).strip(), metadata={"filename": filename or "upload.md", "source": "upload", "section": title}))
            for line in text.splitlines():
                if line.lstrip().startswith('#'):
                    flush_md()
                    title = line.lstrip('#').strip() or title
                    buf = []
                else:
                    buf.append(line)
            flush_md()
        else:
            docs.append(Document(page_content=text, metadata={"filename": filename or "upload", "source": "upload"}))

        splits = self.text_splitter.split_documents(docs)
        store = self._vector(namespace=project_id)
        try:
            await store.aadd_documents(splits)
            logger.info(f"RAG indexed upload into namespace={project_id}: file={filename} chunks={len(splits)}")
        except Exception:
            logger.exception("Failed to index upload into Pinecone")

    # ---------- Retrieval & QA ----------
    async def retrieve(
        self, project_id: str, query: str, k: Optional[int] = None
    ) -> List[Document]:
        if not query:
            return []
        store = self._vector(namespace=project_id)
        retriever = store.as_retriever(search_kwargs={"k": k or self.config.top_k})
        results = await retriever.ainvoke(query)
        return results

    async def answer(
        self,
        project_id: str,
        question: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        retrieved = await self.retrieve(project_id, question)
        context_lines: List[str] = []
        for i, d in enumerate(retrieved, 1):
            filename = d.metadata.get("filename") or "unknown"
            start_index = d.metadata.get("start_index")
            section = d.metadata.get("section")
            header = f"{i}) filename: {filename} | section: {section or '-'} | idx: {start_index if start_index is not None else '-'}"
            context_lines.append(header + "\n" + (d.page_content or ""))
        context = "\n\n".join(context_lines)

        from langchain_core.prompts import PromptTemplate

        prompt = PromptTemplate.from_template(
            (
                "You are a PRD-QA assistant. Answer ONLY using the supplied project context.\n"
                "If unknown, say: 'I don't know based on the PRD and uploaded docs.'\n"
                "Be concise (<= 6 sentences). Prefer bullet points.\n"
                "Do NOT invent facts.\n\n"
                "Question: {question}\n\nContext:\n{context}\n\nAnswer:"
            )
        )

        llm = self._llm()
        formatted = prompt.format(question=question, context=context)
        resp = await llm.ainvoke(formatted)
        return {"answer": getattr(resp, "content", str(resp))}

    async def stream_answer(
        self,
        project_id: str,
        question: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        # Decide mode: uploads RAG vs PRD summary
        from app.services.project_service import ProjectService  # only for type
        has_uploads = False
        try:
            # Basic heuristic: try listing uploads count via repository
            from app.dependencies import get_project_service
            ps: ProjectService = get_project_service()
            uploads = await ps.list_project_files(project_id, user_id="system")  # user_id not checked in repo layer
            has_uploads = bool(uploads)
        except Exception:
            has_uploads = False

        if has_uploads:
            # RAG from uploads only
            retrieved = await self.retrieve(project_id, question)
            docs_content = "\n\n".join((d.page_content or "") for d in retrieved)
            from langchain_core.prompts import PromptTemplate
            prompt = PromptTemplate.from_template(
                (
                    "You are a PRD-QA assistant. Answer ONLY using the supplied project context.\n"
                    "If unknown, say: 'I don't know based on the PRD and uploaded docs.'\n"
                    "Be concise (<= 6 sentences).\n\n"
                    "Question: {question}\n\nContext:\n{context}\n\nAnswer:"
                )
            )
            llm = self._llm_streaming()
            formatted = prompt.format(question=question, context=docs_content)
            buffer: List[str] = []
            async for chunk in llm.astream(formatted):
                text = getattr(chunk, "content", None)
                if not text:
                    continue
                buffer.append(text)
                yield {"type": "delta", "text": text}
            full_text = "".join(buffer)
            yield {"type": "complete", "text": full_text}
            return

        # Fallback: PRD summary stuffed into prompt (no embeddings)
        summary_text = ""
        try:
            from app.services.cache_service import cache_service
            # We need the latest ETag; try simple probe using repository (not strictly required if a single summary per project is fine)
            from app.dependencies import get_project_service
            ps = get_project_service()
            project = await ps.get_project(project_id, user_id="system")  # type: ignore
            etag = None
            try:
                # naive: attempt to read current ETag from project metadata or via storage checksum; skip if unavailable
                etag = None
            except Exception:
                etag = None
            # If we cannot resolve ETag reliably, cache can be keyed by project only
            key = f"prd:summary:{project_id}:{etag or 'current'}"
            if await cache_service.is_connected():
                found = await cache_service.redis.get(key)  # type: ignore
                if found:
                    summary_text = str(found)
        except Exception:
            pass

        if not summary_text:
            # As a safety fallback (should be rare if save path cached summary), use a direct small summary now
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model=self.config.openai_model, temperature=0.2)
            summary_text = (await llm.ainvoke(f"Summarize this PRD in ~800 words:\n\n{project_id}"))
            summary_text = getattr(summary_text, "content", "") or ""

        from langchain_core.prompts import PromptTemplate
        prompt = PromptTemplate.from_template(
            (
                "You are a PRD-QA assistant. Answer ONLY using the supplied project context.\n"
                "If unknown, say: 'I don't know based on the PRD and uploaded docs.'\n"
                "Be concise (<= 6 sentences).\n\n"
                "Question: {question}\n\nContext (PRD summary):\n{context}\n\nAnswer:"
            )
        )
        llm = self._llm_streaming()
        formatted = prompt.format(question=question, context=summary_text)
        buffer: List[str] = []
        async for chunk in llm.astream(formatted):
            text = getattr(chunk, "content", None)
            if not text:
                continue
            buffer.append(text)
            yield {"type": "delta", "text": text}
        full_text = "".join(buffer)
        yield {"type": "complete", "text": full_text}


rag_service = RAGService()


