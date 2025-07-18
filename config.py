import os
from pathlib import Path
from typing import Dict, Any, Optional

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

dotenv_path = BASE_DIR / ".env"
if dotenv_path.exists():
    load_dotenv(dotenv_path)
else:
    print(f"警告: 未找到 .env 文件于 {dotenv_path}。将仅使用环境变量或默认值。")


class Config:
    """
    核心应用配置类。
    从环境变量或 .env 文件加载设置。
    """
    WORKING_DIR: Path = BASE_DIR
    LLM_TYPE: str = os.environ.get("LLM_TYPE", "deepseek-v3").lower()
    LLM_CONFIG: Dict[str, Dict[str, Any]] = {
        "deepseek-v3": {
            "model": "deepseek-v3",
            "base_url": "https://cloud.infini-ai.com/maas/v1/",
            "api_key": os.environ.get("GENSTUDIO_API_KEY"),
            "max_concurrency": int(os.environ.get("MAX_CONCURRENCY", "10")),
            "timeout_seconds": int(os.environ.get("LLM_TIMEOUT_SECONDS", "30")),
            "retry_attempts": int(os.environ.get("LLM_RETRY_ATTEMPTS", "5")),
        },
        "deepseek-chat": {
            "model": "deepseek-chat",
            "base_url": os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/"),
            "api_key": os.environ.get("DEEPSEEK_API_KEY"),
            "max_concurrency": int(os.environ.get("MAX_CONCURRENCY", "10")),
            "timeout_seconds": int(os.environ.get("LLM_TIMEOUT_SECONDS", "30")),
            "retry_attempts": int(os.environ.get("LLM_RETRY_ATTEMPTS", "5")),
        }
        # 根据需要在此处添加更多 LLM 配置
    }

    @classmethod
    def get_llm_config(cls) -> Dict[str, Any]:
        """
        获取当前选定 LLM 的配置。

        返回:
            Dict[str, Any]: LLM 配置字典的副本。

        引发:
            ValueError: 如果 LLM_TYPE 在 LLM_CONFIG 中未找到。
        """
        config = cls.LLM_CONFIG.get(cls.LLM_TYPE)
        if config is None:
            supported_models = list(cls.LLM_CONFIG.keys())
            raise ValueError(
                f"错误: 未知的 LLM_MODEL (LLM_TYPE) '{cls.LLM_TYPE}'。 "
                f"请在您的 .env 文件中将 LLM_MODEL 设置为以下之一: {', '.join(supported_models)}"
            )
        return config.copy()

    EMBEDDING_MODEL: str = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")
    EMBEDDING_MAX_TOKEN_SIZE: int = int(os.environ.get("EMBEDDING_MAX_TOKEN_SIZE", "8192"))  # 嵌入模型的最大 token 数

    NEO4J_URI: str = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USERNAME: str = os.environ.get("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD: Optional[str] = os.environ.get("NEO4J_PASSWORD")


def validate_config() -> None:
    """
    验证应用程序配置，确保关键设置已存在。
    如果验证失败，则引发 ValueError 或 FileNotFoundError。
    """
    print("正在验证配置...")

    try:
        llm_config = Config.get_llm_config()
        if not llm_config.get("api_key"):
            raise ValueError(
                f"错误: LLM '{Config.LLM_TYPE}' 的 API_KEY 未设置。 "
            )
    except ValueError as e:
        raise e

    if not Config.EMBEDDING_MODEL:
        raise ValueError(
            "错误: EMBEDDING_MODEL 未设置。 "
            "请在您的 .env 文件中指定一个有效的嵌入模型路径或名称。"
        )

    if Config.NEO4J_URI == "bolt://localhost:7687":
        print(f"警告: NEO4J_URI 正在使用默认值 '{Config.NEO4J_URI}'。请确保这是预期的行为。")
    if Config.NEO4J_USERNAME == "neo4j":
        print(f"警告: NEO4J_USERNAME 正在使用默认值 '{Config.NEO4J_USERNAME}'。请确保这是预期的行为。")
    if not Config.NEO4J_PASSWORD:
        print("警告: NEO4J_PASSWORD 未设置。如果您的 Neo4j 实例需要密码，连接可能会失败。")

    if not Config.WORKING_DIR.exists():
        print(f"工作目录 {Config.WORKING_DIR} 未找到。正在创建它...")
        try:
            Config.WORKING_DIR.mkdir(parents=True, exist_ok=True)
            print(f"成功创建工作目录: {Config.WORKING_DIR}")
        except OSError as e:
            raise OSError(f"错误: 无法创建工作目录 {Config.WORKING_DIR}: {e}")

    print("配置验证成功！")


if __name__ == "__main__":
    try:
        validate_config()
        print("\n--- 当前应用配置 ---")
        print(f"工作目录 (Working Directory): {Config.WORKING_DIR}")

        print("\n--- LLM 配置 ---")
        current_llm_config = Config.get_llm_config()
        print(f"选定的 LLM 类型 (Selected LLM Type): {Config.LLM_TYPE}")
        print(f"  模型名称 (Model Name): {current_llm_config.get('model')}")
        print(f"  基础 URL (Base URL): {current_llm_config.get('base_url')}")
        print(f"  API 密钥已设置 (API Key Set): {'是 (Yes)' if current_llm_config.get('api_key') else '否 (No) (如果未使用但需要则会报错!)'}")
        print(f"  最大并发数 (Max Concurrency): {current_llm_config.get('max_concurrency')}")
        print(f"  超时 (秒) (Timeout (seconds)): {current_llm_config.get('timeout_seconds')}")
        print(f"  重试次数 (Retry Attempts): {current_llm_config.get('retry_attempts')}")

        print("\n--- 嵌入模型配置 ---")
        print(f"嵌入模型 (Embedding Model): {Config.EMBEDDING_MODEL}")
        print(f"嵌入模型最大 Token 数 (Embedding Max Token Size): {Config.EMBEDDING_MAX_TOKEN_SIZE}")

        print("\n--- Neo4j 配置 ---")
        print(f"Neo4j URI: {Config.NEO4J_URI}")
        print(f"Neo4j 用户名 (Username): {Config.NEO4J_USERNAME}")
        print(f"Neo4j 密码 (Password): {'********' if Config.NEO4J_PASSWORD else '未设置 (Not set)'}")

    except (ValueError, FileNotFoundError, OSError) as e:
        print(f"\n配置错误 (Configuration Error): {e}")
        exit(1)
