"""配置贯通与文件日志验证；不需要真实模型密钥。"""

import subprocess
import sys


def test_env_config_reaches_retrieval_defaults(tmp_path, monkeypatch):
    from search_agent.core.config import get_config
    from search_agent.core.retrieval import RetrievalResult, Retriever
    from search_agent.core.tools.retrieve import create_retrieve_tool

    (tmp_path / ".env").write_text(
        "EMBEDDING__DENSE_MODEL=test-encoder\nCHROMA__PATH=custom-chroma\n"
        "CHROMA__DENSE_COLLECTION=custom-dense\nRETRIEVAL__TOP_K=8\n"
        "RETRIEVAL__METHOD=dense\nINDEXING__CHUNK_SIZE=900\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RETRIEVAL__TOP_K", "3")
    config = get_config()
    assert config.indexing.chunk_size == 900

    def retrieve(self, query, top_k=None, method=None, instruction=None):
        assert self.config.embedding.dense_model == "test-encoder"
        assert str(self.config.chroma.path) == "custom-chroma"
        assert self.config.chroma.dense_collection == "custom-dense"
        assert (query, top_k, method) == ("question", 3, "dense")
        return [RetrievalResult(doc_id="d1", score=1, snippet="Evidence")]

    monkeypatch.setattr(Retriever, "retrieve", retrieve)
    result = create_retrieve_tool(config).invoke({"query": "question"})
    assert result["result"][0]["doc_id"] == "d1"


def test_root_logger_is_file_only_and_setup_is_idempotent(tmp_path):
    path = tmp_path / "logs" / "agent.log"
    program = f"""
import logging
from pathlib import Path
from search_agent.core.config import LoggingConfig
from search_agent.core.logging_setup import setup_logger
logging.getLogger().addHandler(logging.StreamHandler())
settings = LoggingConfig(path=Path({str(path)!r}), level='INFO')
setup_logger(settings)
logging.getLogger('agent').info('first-record')
setup_logger(settings)
logging.getLogger('evaluation').warning('second-record')
assert len(logging.getLogger().handlers) == 1
logging.shutdown()
"""
    result = subprocess.run([sys.executable, "-c", program], capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout == b"" and result.stderr == b""
    text = path.read_text(encoding="utf-8")
    assert text.count("first-record") == 1
    assert text.count("second-record") == 1
    assert "INFO | agent |" in text
    assert "WARNING | evaluation |" in text
