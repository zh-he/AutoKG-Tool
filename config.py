import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any

# 定位项目根目录并加载 .env 文件
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


class Config:
    # 领域
    DOMAIN = os.environ.get("DOMAIN", "计算机科学")

    # LIGHTRAG工作目录配置
    DEFAULT_RAG_DIR: str = str(BASE_DIR / "LightRAG_KG_QA")
    WORKING_DIR: str = os.environ.get("RAG_DIR", DEFAULT_RAG_DIR)

    # 知识图谱本体和输出JSON文件配置
    ONTOLOGY_FILE = os.environ.get("ONTOLOGY_FILE", "ontology.json")
    OUTPUT_FILE = os.environ.get("OUTPUT_FILE", "knowledge_graph.json")

    # 知识图谱的根节点
    ROOT_TYPE = "computer_science"

    # LLM 配置，可扩展为支持多种 LLM，使用 deepseek-chat 作为默认
    LLM_TYPE: str = os.environ.get("LLM_MODEL", "deepseek-chat")
    LLM_CONFIG: Dict[str, Dict[str, Any]] = {
        "deepseek-chat": {
            "model": "deepseek-chat",
            "base_url": os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            "api_key": os.environ.get("DEEPSEEK_API_KEY", ""),
        },

        "gpt-4o": {
            "model": "gpt-4o",
            "base_url": os.environ.get("BASE_URL", "https://api.openai.com/v1"),
            "api_key": os.environ.get("OPENAI_API_KEY", "")
        },
    }

    @classmethod
    def get_llm_config(cls) -> Dict[str, Any]:
        config = cls.LLM_CONFIG.get(cls.LLM_TYPE)
        if config is None:
            raise ValueError(f"未找到 LLM_TYPE({cls.LLM_TYPE}) 对应的配置")
        return config

    # 嵌入模型配置
    EMBEDDING_MODEL: str = os.environ.get("EMBEDDING_MODEL", "your embedding path")
    EMBEDDING_MAX_TOKEN_SIZE: int = int(os.environ.get("EMBEDDING_MAX_TOKEN_SIZE", "8192"))

    # 知识图谱配置
    KG_FILE_PATH: str = os.environ.get("KG_FILE_PATH", "./lightrag_formatted_kg.json")

    # Neo4j 配置
    NEO4J_URI: str = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USERNAME: str = os.environ.get("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD: str = os.environ.get("NEO4J_PASSWORD", "password")


def validate_config() -> bool:
    """
    验证配置完整性，确保关键配置已设置：
      - 验证当前 LLM 配置中的 API_KEY 已配置
      - 检查自定义嵌入模型路径是否已设置（警告默认值）
      - 确保工作目录存在，不存在则自动创建
    """
    llm_config = Config.get_llm_config()
    if not llm_config.get("api_key"):
        raise ValueError(
            f"未设置 {Config.LLM_TYPE} 的 API_KEY，请在 .env 文件中配置对应变量"
        )

    if Config.EMBEDDING_MODEL == "your_default_embedding_path":
        print("警告: 未设置自定义嵌入模型路径，正在使用默认值")

    if not os.path.exists(Config.WORKING_DIR):
        print(f"创建工作目录: {Config.WORKING_DIR}")
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

    return True


if __name__ == "__main__":
    if validate_config():
        print("配置验证通过！")
        print("当前使用的 LLM 配置：", Config.get_llm_config())
