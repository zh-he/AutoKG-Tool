# LightRAG_comparison_qa.py
import asyncio
import json
import os

import nest_asyncio
import numpy as np
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc
from sentence_transformers import SentenceTransformer

from config import Config

nest_asyncio.apply()

# 配置
DEFAULT_RAG_DIR = Config.DEFAULT_RAG_DIR
WORKING_DIR = Config.WORKING_DIR
llm_config = Config.get_llm_config()
LLM_MODEL = llm_config.get("model")
BASE_URL = llm_config.get("base_url")
API_KEY = llm_config.get("api_key")
EMBEDDING_MODEL = Config.EMBEDDING_MODEL
EMBEDDING_MAX_TOKEN_SIZE = Config.EMBEDDING_MAX_TOKEN_SIZE
KG_FILE_PATH = Config.KG_FILE_PATH

if not os.path.exists(WORKING_DIR):
    os.makedirs(WORKING_DIR)

sem = asyncio.Semaphore(2)


async def llm_model_func(prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs) -> str:
    async with sem:
        await asyncio.sleep(0.5)
        jitter = np.random.uniform(0.1, 0.3)
        await asyncio.sleep(jitter)
        return await openai_complete_if_cache(
            model=LLM_MODEL,
            prompt=prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            base_url=BASE_URL,
            api_key=API_KEY,
            **kwargs,
        )


embedding_model = SentenceTransformer(EMBEDDING_MODEL)


async def embedding_func(texts: list[str]) -> np.ndarray:
    embeddings = embedding_model.encode(texts)
    return np.array(embeddings)


async def get_embedding_dim():
    test_text = ["This is a test sentence."]
    embedding = await embedding_func(test_text)
    embedding_dim = embedding.shape[1]
    print(f"embedding_dim = {embedding_dim}")
    return embedding_dim


async def initialize_lightrag():
    embedding_dim = await get_embedding_dim()
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=EMBEDDING_MAX_TOKEN_SIZE,
            func=embedding_func,
        ),
        chunk_token_size=1000,
        chunk_overlap_token_size=100,
    )
    return rag


if __name__ == "__main__":
    # --- 异步初始化 LightRAG ---
    rag = asyncio.run(initialize_lightrag())
    # --- 加载并插入知识图谱 ---
    print(f"正在加载知识图谱文件: {KG_FILE_PATH}")
    try:
        # 确保文件存在
        if not os.path.exists(KG_FILE_PATH):
            print(f"错误：知识图谱文件 {KG_FILE_PATH} 不存在！请检查路径和文件名。")
            exit()  # 如果KG文件不存在，无法继续

        # 修正编码错误，并直接使用文件对象
        with open(KG_FILE_PATH, "r", encoding="utf-8") as f:
            domain_kg = json.load(f)
        print("知识图谱加载成功，正在插入 LightRAG...")
        rag.insert_custom_kg(domain_kg)  # 假设这个方法是同步的
        print("知识图谱插入完成。")
    except json.JSONDecodeError:
        print(f"错误：无法解析知识图谱文件 {KG_FILE_PATH}，请检查JSON格式。")
        exit()
    except FileNotFoundError:
        print(f"错误：知识图谱文件 {KG_FILE_PATH} 未找到。")
        exit()
    except Exception as e:
        print(f"加载或插入知识图谱时出错: {e}")
        exit()

    # --- 进入问答循环 ---
    print("\n" + "=" * 10 + " 欢迎使用 LightRAG 问答系统 " + "=" * 10)
    print("您可以开始提问了。")
    print("输入 'quit'、'exit' 或直接按 Enter 键退出程序。")
    print("-" * 50)

    while True:
        try:
            # 1. 获取用户输入
            user_question = input("请输入您的问题: ")
            cleaned_input = user_question.strip()

            # 2. 检查退出条件
            if cleaned_input in ["quit", "exit", ""]:
                print("感谢使用，再见！")
                break

            # 3. 执行查询
            print("正在查询，请稍候...")
            result = rag.query(
                user_question,  # 使用原始问题进行查询
                param=QueryParam(mode="hybrid")
            )

            # 4. 打印结果
            print("\n--- 问答结果 ---")
            # 检查 result 的类型并尝试友好地打印
            if hasattr(result, 'answer'):  # 如果结果对象有 'answer' 属性
                print(result.answer)
            elif isinstance(result, dict) and 'answer' in result:  # 如果结果是字典且有 'answer' 键
                print(result['answer'])
            elif isinstance(result, str):  # 如果结果直接是字符串
                print(result)
            else:  # 其他情况，直接打印对象本身
                print(result)
            print("-" * 50)

        except KeyboardInterrupt:  # 允许使用 Ctrl+C 退出循环
            print("\n检测到中断信号，正在退出...")
            break
        except Exception as e:
            print(f"\n处理查询时发生错误: {e}")
            print("-" * 50)
