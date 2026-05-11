import asyncio as _asyncio

import time as _time
from observability.observability_wrapper import (
    trace_agent, trace_step, trace_step_sync, trace_model_call, trace_tool_call,
)
from config import settings as _obs_settings

import logging as _obs_startup_log
from contextlib import asynccontextmanager
from observability.instrumentation import initialize_tracer

_obs_startup_logger = _obs_startup_log.getLogger(__name__)

from modules.guardrails.content_safety_decorator import with_content_safety

GUARDRAILS_CONFIG = {
    'content_safety_enabled': True,
    'runtime_enabled': True,
    'content_safety_severity_threshold': 3,
    'check_toxicity': True,
    'check_jailbreak': True,
    'check_pii_input': False,
    'check_credentials_output': True,
    'check_output': True,
    'check_toxic_code_output': True,
    'sanitize_pii': False
}

import logging
import json
from typing import List, Optional, Dict, Any
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator

from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery
import openai

from config import Config

# =========================
# Constants and Prompts
# =========================

SYSTEM_PROMPT = (
    "You are a professional assistant specializing in extracting and summarizing information from a curated collection of resumes. "
    "Your task is to answer user questions by searching the provided resume documents using Azure AI Search and delivering clear, accurate, and concise responses based strictly on the retrieved content. "
    "Do not speculate or provide information that is not present in the resumes. If the answer cannot be found, politely inform the user. Always maintain a formal and professional tone."
)
OUTPUT_FORMAT = (
    "Provide a direct, well-structured answer to the user's question. If relevant, include specific details such as job titles, skills, achievements, or education as found in the resumes. "
    "Do not include personal contact information or sensitive data."
)
FALLBACK_RESPONSE = "I'm sorry, I could not find relevant information in the available resumes to answer your question."

SELECTED_DOCUMENT_TITLES = [
    "resumes_collection1.pdf",
    "resumes_collection2.pdf",
    "resumes_collection3.pdf",
    "resumes_collection4.pdf"
]

ENRICHED_FIELDS = ["entities", "keyphrases", "relationships"]

VALIDATION_CONFIG_PATH = Config.VALIDATION_CONFIG_PATH or str(Path(__file__).parent / "validation_config.json")

_logger = logging.getLogger(__name__)

# =========================
# Observability Lifespan
# =========================

@asynccontextmanager
async def _obs_lifespan(application):
    """Initialise observability on startup, clean up on shutdown."""
    try:
        _obs_startup_logger.info('')
        _obs_startup_logger.info('========== Agent Configuration Summary ==========')
        _obs_startup_logger.info(f'Environment: {getattr(Config, "ENVIRONMENT", "N/A")}')
        _obs_startup_logger.info(f'Agent: {getattr(Config, "AGENT_NAME", "N/A")}')
        _obs_startup_logger.info(f'Project: {getattr(Config, "PROJECT_NAME", "N/A")}')
        _obs_startup_logger.info(f'LLM Provider: {getattr(Config, "MODEL_PROVIDER", "N/A")}')
        _obs_startup_logger.info(f'LLM Model: {getattr(Config, "LLM_MODEL", "N/A")}')
        _cs_endpoint = getattr(Config, 'AZURE_CONTENT_SAFETY_ENDPOINT', None)
        _cs_key = getattr(Config, 'AZURE_CONTENT_SAFETY_KEY', None)
        if _cs_endpoint and _cs_key:
            _obs_startup_logger.info('Content Safety: Enabled (Azure Content Safety)')
            _obs_startup_logger.info(f'Content Safety Endpoint: {_cs_endpoint}')
        else:
            _obs_startup_logger.info('Content Safety: Not Configured')
        _obs_startup_logger.info('Observability Database: Azure SQL')
        _obs_startup_logger.info(f'Database Server: {getattr(Config, "OBS_AZURE_SQL_SERVER", "N/A")}')
        _obs_startup_logger.info(f'Database Name: {getattr(Config, "OBS_AZURE_SQL_DATABASE", "N/A")}')
        _obs_startup_logger.info('===============================================')
        _obs_startup_logger.info('')
    except Exception as _e:
        _obs_startup_logger.warning('Config summary failed: %s', _e)

    _obs_startup_logger.info('')
    _obs_startup_logger.info('========== Content Safety & Guardrails ==========')
    if GUARDRAILS_CONFIG.get('content_safety_enabled'):
        _obs_startup_logger.info('Content Safety: Enabled')
        _obs_startup_logger.info(f'  - Severity Threshold: {GUARDRAILS_CONFIG.get("content_safety_severity_threshold", "N/A")}')
        _obs_startup_logger.info(f'  - Check Toxicity: {GUARDRAILS_CONFIG.get("check_toxicity", False)}')
        _obs_startup_logger.info(f'  - Check Jailbreak: {GUARDRAILS_CONFIG.get("check_jailbreak", False)}')
        _obs_startup_logger.info(f'  - Check PII Input: {GUARDRAILS_CONFIG.get("check_pii_input", False)}')
        _obs_startup_logger.info(f'  - Check Credentials Output: {GUARDRAILS_CONFIG.get("check_credentials_output", False)}')
    else:
        _obs_startup_logger.info('Content Safety: Disabled')
    _obs_startup_logger.info('===============================================')
    _obs_startup_logger.info('')

    _obs_startup_logger.info('========== Initializing Agent Services ==========')
    # 1. Observability DB schema (imports are inside function — only needed at startup)
    try:
        from observability.database.engine import create_obs_database_engine
        from observability.database.base import ObsBase
        import observability.database.models  # noqa: F401
        _obs_engine = create_obs_database_engine()
        ObsBase.metadata.create_all(bind=_obs_engine, checkfirst=True)
        _obs_startup_logger.info('✓ Observability database connected')
    except Exception as _e:
        _obs_startup_logger.warning('✗ Observability database connection failed (metrics will not be saved)')
    # 2. OpenTelemetry tracer (initialize_tracer is pre-injected at top level)
    try:
        _t = initialize_tracer()
        if _t is not None:
            _obs_startup_logger.info('✓ Telemetry monitoring enabled')
        else:
            _obs_startup_logger.warning('✗ Telemetry monitoring disabled')
    except Exception as _e:
        _obs_startup_logger.warning('✗ Telemetry monitoring failed to initialize')
    _obs_startup_logger.info('=================================================')
    _obs_startup_logger.info('')
    yield

app = FastAPI(
    title="Resume Insights Answer Agent",
    description="Answers questions about professional resumes using Azure AI Search and GPT-4.1, with strict privacy compliance.",
    version=Config.SERVICE_VERSION if hasattr(Config, "SERVICE_VERSION") else "1.0.0",
    lifespan=_obs_lifespan
)

# =========================
# Input/Output Models
# =========================

class QueryRequest(BaseModel):
    query: str = Field(..., description="User question about the resumes")

    @model_validator(mode="after")
    def validate_content(self):
        if not self.query or not self.query.strip():
            raise ValueError("Query must be non-empty.")
        if len(self.query.strip()) > 50000:
            raise ValueError("Query exceeds maximum length.")
        self.query = self.query.strip()
        return self

class QueryResponse(BaseModel):
    success: bool = Field(..., description="Whether the query was processed successfully")
    answer: Optional[str] = Field(None, description="Agent's answer to the query")
    error: Optional[str] = Field(None, description="Error message, if any")
    tool_calls_made: Optional[List[str]] = Field(None, description="List of tool calls made (always empty for this agent)")

# =========================
# LLM Output Sanitizer
# =========================

import re as _re

_FENCE_RE = _re.compile(r"```(?:\w+)?\s*\n(.*?)```", _re.DOTALL)
_LONE_FENCE_START_RE = _re.compile(r"^```\w*$")
_WRAPPER_RE = _re.compile(
    r"^(?:"
    r"Here(?:'s| is)(?: the)? (?:the |your |a )?(?:code|solution|implementation|result|explanation|answer)[^:]*:\s*"
    r"|Sure[!,.]?\s*"
    r"|Certainly[!,.]?\s*"
    r"|Below is [^:]*:\s*"
    r")",
    _re.IGNORECASE,
)
_SIGNOFF_RE = _re.compile(
    r"^(?:Let me know|Feel free|Hope this|This code|Note:|Happy coding|If you)",
    _re.IGNORECASE,
)
_BLANK_COLLAPSE_RE = _re.compile(r"\n{3,}")

def _strip_fences(text: str, content_type: str) -> str:
    """Extract content from Markdown code fences."""
    fence_matches = _FENCE_RE.findall(text)
    if fence_matches:
        if content_type == "code":
            return "\n\n".join(block.strip() for block in fence_matches)
        for match in fence_matches:
            fenced_block = _FENCE_RE.search(text)
            if fenced_block:
                text = text[:fenced_block.start()] + match.strip() + text[fenced_block.end():]
        return text
    lines = text.splitlines()
    if lines and _LONE_FENCE_START_RE.match(lines[0].strip()):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()

def _strip_trailing_signoffs(text: str) -> str:
    """Remove conversational sign-off lines from the end of code output."""
    lines = text.splitlines()
    while lines and _SIGNOFF_RE.match(lines[-1].strip()):
        lines.pop()
    return "\n".join(lines).rstrip()

@with_content_safety(config=GUARDRAILS_CONFIG)
def sanitize_llm_output(raw: str, content_type: str = "code") -> str:
    """
    Generic post-processor that cleans common LLM output artefacts.
    Args:
        raw: Raw text returned by the LLM.
        content_type: 'code' | 'text' | 'markdown'.
    Returns:
        Cleaned string ready for validation, formatting, or direct return.
    """
    if not raw:
        return ""
    text = _strip_fences(raw.strip(), content_type)
    text = _WRAPPER_RE.sub("", text, count=1).strip()
    if content_type == "code":
        text = _strip_trailing_signoffs(text)
    return _BLANK_COLLAPSE_RE.sub("\n\n", text).strip()

# =========================
# Compliance Manager
# =========================

class ComplianceManager:
    """
    Redacts or omits PII (email, phone, address) from LLM responses.
    """
    EMAIL_RE = _re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
    PHONE_RE = _re.compile(r"\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b")
    ADDRESS_RE = _re.compile(
        r"\d{1,5}\s+\w+(?:\s+\w+)*\s+(?:Street|St|Avenue|Ave|Boulevard|Blvd|Road|Rd|Lane|Ln|Drive|Dr|Court|Ct|Circle|Cir|Way|Place|Pl|Square|Sq|Loop|Trail|Trl|Parkway|Pkwy|Commons|Cmns)\b",
        _re.IGNORECASE
    )

    def sanitize_response(self, response: str) -> str:
        """
        Redact emails, phone numbers, and addresses from the response.
        """
        if not response:
            return response
        sanitized = self.EMAIL_RE.sub("[EMAIL REDACTED]", response)
        sanitized = self.PHONE_RE.sub("[PHONE REDACTED]", sanitized)
        sanitized = self.ADDRESS_RE.sub("[ADDRESS REDACTED]", sanitized)
        return sanitized

# =========================
# Error Handler
# =========================

class ErrorHandler:
    """
    Handles errors, retries, timeouts, and fallback responses; logs incidents.
    """
    def handle_error(self, error: Exception) -> str:
        _logger.error("Error occurred: %s", error, exc_info=True)
        return FALLBACK_RESPONSE

# =========================
# Tool Registry (No tools for this agent)
# =========================

class BaseTool:
    """
    Defines interface for tool integration; used for function-calling.
    """
    @trace_agent(agent_name=_obs_settings.AGENT_NAME, project_name=_obs_settings.PROJECT_NAME)
    def execute(self, params: dict) -> dict:
        raise NotImplementedError("BaseTool is abstract.")

class ToolRegistry:
    """
    Manages available tools for OpenAI function-calling; registers BaseTool instances.
    """
    def __init__(self):
        self._tools: List[BaseTool] = []

    def get_tools(self) -> List[BaseTool]:
        return self._tools.copy()

# =========================
# Chunk Retriever
# =========================

_enriched_available = None  # None = not yet checked, True/False after first search

class ChunkRetriever:
    """
    Queries Azure AI Search using vector semantic search and OData filter for selected document titles.
    """
    def __init__(self):
        self._search_client = None

    def _get_search_client(self):
        if self._search_client is None:
            self._search_client = SearchClient(
                endpoint=Config.AZURE_SEARCH_ENDPOINT,
                index_name=Config.AZURE_SEARCH_INDEX_NAME,
                credential=AzureKeyCredential(Config.AZURE_SEARCH_API_KEY),
            )
        return self._search_client

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def retrieve_chunks(self, query: str, document_titles: List[str], top_k: int) -> List[Dict[str, Any]]:
        """
        Retrieve top-k relevant chunks from Azure AI Search, filtered by document titles.
        Returns list of dicts (each chunk with possible enriched fields).
        """
        global _enriched_available
        search_client = self._get_search_client()

        # Get embedding for the query
        embedding = await self._get_embedding(query)
        vector_query = VectorizedQuery(vector=embedding, k_nearest_neighbors=top_k, fields="vector")

        base_fields = ["chunk", "title"]
        select_fields = base_fields + ENRICHED_FIELDS if _enriched_available is not False else base_fields

        search_kwargs = {
            "search_text": query,
            "vector_queries": [vector_query],
            "top": top_k,
            "select": select_fields,
        }
        if document_titles:
            odata_parts = [f"title eq '{t}'" for t in document_titles]
            search_kwargs["filter"] = " or ".join(odata_parts)

        from azure.core.exceptions import HttpResponseError

        _t0 = _time.time()
        try:
            results = list(search_client.search(**search_kwargs))
            if _enriched_available is None:
                _enriched_available = True
                _logger.info("Enriched index fields are AVAILABLE — using: %s", ENRICHED_FIELDS)
            try:
                trace_tool_call(
                    tool_name="search_client.search",
                    latency_ms=int((_time.time() - _t0) * 1000),
                    output=str(results)[:200] if results is not None else None,
                    status="success",
                )
            except Exception:
                pass
            return results
        except HttpResponseError as e:
            if "Could not find a property named" in str(e) and _enriched_available is not False:
                _enriched_available = False
                _logger.warning("Enriched index fields NOT available in this index — falling back to base fields: %s", base_fields)
                search_kwargs["select"] = base_fields
                _t0 = _time.time()
                results = list(search_client.search(**search_kwargs))
                try:
                    trace_tool_call(
                        tool_name="search_client.search",
                        latency_ms=int((_time.time() - _t0) * 1000),
                        output=str(results)[:200] if results is not None else None,
                        status="success",
                    )
                except Exception:
                    pass
                return results
            try:
                trace_tool_call(
                    tool_name="search_client.search",
                    latency_ms=int((_time.time() - _t0) * 1000),
                    output=str(e)[:200],
                    status="error",
                    error=e,
                )
            except Exception:
                pass
            raise

    async def _get_embedding(self, text: str) -> List[float]:
        """
        Get vector embedding for the given text using Azure OpenAI.
        """
        client = openai.AsyncAzureOpenAI(
            api_key=Config.AZURE_OPENAI_API_KEY,
            api_version="2024-02-01",
            azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
        )
        model = Config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT or "text-embedding-ada-002"
        _t0 = _time.time()
        resp = await client.embeddings.create(
            input=text,
            model=model
        )
        try:
            trace_tool_call(
                tool_name="openai_client.embeddings.create",
                latency_ms=int((_time.time() - _t0) * 1000),
                output=str(resp)[:200],
                status="success",
            )
        except Exception:
            pass
        return resp.data[0].embedding

# =========================
# LLM Service
# =========================

class LLMService:
    """
    Calls Azure OpenAI GPT-4.1 with enhanced system prompt, user query, and retrieved chunks.
    """
    def __init__(self):
        self._client = None
        self.model = Config.LLM_MODEL or "gpt-4.1"

    def _get_client(self):
        if self._client is None:
            api_key = Config.AZURE_OPENAI_API_KEY
            if not api_key:
                raise ValueError("AZURE_OPENAI_API_KEY not configured")
            self._client = openai.AsyncAzureOpenAI(
                api_key=api_key,
                api_version="2024-02-01",
                azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
            )
        return self._client

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def generate_response(self, query: str, chunks: List[Dict[str, Any]], tools: List[BaseTool] = None) -> str:
        """
        Call LLM with system prompt, user query, and retrieved chunks.
        """
        # Build context for LLM
        context = self._format_context(chunks)
        system_message = SYSTEM_PROMPT + "\n\nOutput Format: " + OUTPUT_FORMAT
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"{query}\n\nContext:\n{context}"}
        ]
        client = self._get_client()
        _llm_kwargs = Config.get_llm_kwargs()
        _t0 = _time.time()
        response = await client.chat.completions.create(
            model=self.model,
            messages=messages,
            **_llm_kwargs
        )
        content = response.choices[0].message.content
        try:
            trace_model_call(
                provider="azure",
                model_name=self.model,
                prompt_tokens=getattr(getattr(response, "usage", None), "prompt_tokens", 0) or 0,
                completion_tokens=getattr(getattr(response, "usage", None), "completion_tokens", 0) or 0,
                latency_ms=int((_time.time() - _t0) * 1000),
                response_summary=content[:200] if content else "",
            )
        except Exception:
            pass
        return content

    def _format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Format retrieved chunks and enriched fields as LLM context.
        """
        context_parts = []
        for r in chunks:
            part = r.get("chunk", "")
            if _enriched_available:
                for field in ENRICHED_FIELDS:
                    value = r.get(field)
                    if value:
                        # Serialize as JSON if list/dict
                        part += f"\n{field}: {json.dumps(value) if isinstance(value, (list, dict)) else value}"
            context_parts.append(part)
        return "\n\n---\n\n".join(context_parts)

# =========================
# Agent Orchestrator
# =========================

class AgentOrchestrator:
    """
    Coordinates the flow: receives user query, invokes ChunkRetriever, passes context to LLMService, applies compliance checks, returns response.
    """
    def __init__(self):
        self.chunk_retriever = ChunkRetriever()
        self.llm_service = LLMService()
        self.compliance_manager = ComplianceManager()
        self.error_handler = ErrorHandler()
        self.tool_registry = ToolRegistry()

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def process_user_query(self, query: str) -> Dict[str, Any]:
        """
        Entry point for handling user queries; orchestrates retrieval, LLM call, compliance, and error handling.
        """
        async with trace_step(
            "process_user_query",
            step_type="process",
            decision_summary="Orchestrate retrieval, LLM, compliance, and error handling",
            output_fn=lambda r: f"success={r.get('success', False)}"
        ) as step:
            try:
                # 1. Retrieve chunks
                async with trace_step(
                    "retrieve_chunks",
                    step_type="llm_call",
                    decision_summary="Retrieve relevant chunks from Azure AI Search",
                    output_fn=lambda r: f"chunks={len(r) if r else 0}"
                ) as step_chunks:
                    chunks = await self.chunk_retriever.retrieve_chunks(
                        query=query,
                        document_titles=SELECTED_DOCUMENT_TITLES,
                        top_k=5
                    )
                    step_chunks.capture({"chunks": len(chunks)})
                if not chunks or len(chunks) == 0:
                    return {
                        "success": True,
                        "answer": FALLBACK_RESPONSE,
                        "error": None,
                        "tool_calls_made": [],
                    }

                # 2. LLM call
                async with trace_step(
                    "llm_generate_response",
                    step_type="llm_call",
                    decision_summary="Generate answer using LLM and retrieved context",
                    output_fn=lambda r: f"answer={str(r)[:80]}"
                ) as step_llm:
                    raw_response = await self.llm_service.generate_response(
                        query=query,
                        chunks=chunks,
                        tools=self.tool_registry.get_tools()
                    )
                    step_llm.capture({"answer": str(raw_response)[:80]})

                # 3. Sanitize LLM output
                sanitized = sanitize_llm_output(raw_response, content_type="text")
                sanitized = self.compliance_manager.sanitize_response(sanitized)

                return {
                    "success": True,
                    "answer": sanitized,
                    "error": None,
                    "tool_calls_made": [],
                }
            except Exception as e:
                error_msg = self.error_handler.handle_error(e)
                return {
                    "success": False,
                    "answer": None,
                    "error": error_msg,
                    "tool_calls_made": [],
                }

# =========================
# FastAPI Endpoints
# =========================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

@app.post("/query", response_model=QueryResponse)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def query_endpoint(req: QueryRequest):
    """
    Main endpoint for resume insights question answering.
    """
    agent = AgentOrchestrator()
    try:
        result = await agent.process_user_query(req.query)
        # Defensive: sanitize output before returning
        if result.get("answer"):
            result["answer"] = sanitize_llm_output(result["answer"], content_type="text")
        return QueryResponse(**result)
    except Exception as e:
        _logger.error("Unhandled error in /query endpoint: %s", e, exc_info=True)
        return QueryResponse(
            success=False,
            answer=None,
            error="An unexpected error occurred. Please try again later.",
            tool_calls_made=[]
        )

# =========================
# JSON Error Handling
# =========================

@app.exception_handler(Exception)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def generic_exception_handler(request: Request, exc: Exception):
    """
    Handles uncaught exceptions and malformed JSON.
    """
    _logger.error("Exception in request: %s", exc, exc_info=True)
    if isinstance(exc, ValueError):
        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "answer": None,
                "error": f"Invalid input: {str(exc)}. Please check your request and try again.",
                "tool_calls_made": []
            }
        )
    if hasattr(exc, "errors") and callable(getattr(exc, "errors", None)):
        # Pydantic validation error
        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "answer": None,
                "error": "Malformed request. Please check your JSON formatting (quotes, commas, etc.) and try again.",
                "tool_calls_made": []
            }
        )
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "answer": None,
            "error": "An unexpected error occurred. Please try again later.",
            "tool_calls_made": []
        }
    )

# =========================
# Entrypoint
# =========================

async def _run_agent():
    """Entrypoint: runs the agent with observability (trace collection only)."""
    import uvicorn

    # Unified logging config — routes uvicorn, agent, and observability through
    # the same handler so all telemetry appears in a single consistent stream.
    _LOG_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": "%(levelprefix)s %(name)s: %(message)s",
                "use_colors": None,
            },
            "access": {
                "()": "uvicorn.logging.AccessFormatter",
                "fmt": '%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
            "access": {
                "formatter": "access",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "uvicorn":        {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.error":  {"level": "INFO"},
            "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
            "agent":          {"handlers": ["default"], "level": "INFO", "propagate": False},
            "__main__":       {"handlers": ["default"], "level": "INFO", "propagate": False},
            "observability": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "config": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "azure":   {"handlers": ["default"], "level": "WARNING", "propagate": False},
            "urllib3": {"handlers": ["default"], "level": "WARNING", "propagate": False},
        },
    }

    config = uvicorn.Config(
        "agent:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info",
        log_config=_LOG_CONFIG,
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    _asyncio.run(_run_agent())