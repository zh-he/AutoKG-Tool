import asyncio
import colorsys
import json
import logging
import sys
import uuid
import random
from collections import defaultdict, deque
from pathlib import Path

from openai import AsyncOpenAI, APIError, Timeout

from config import Config, validate_config

# ---------------- 日志配置 ----------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# ---------------- 工具函数 ----------------
def parse_response_json(content: str) -> dict | list | None:
    """
    尝试从可能包含```json```标记的字符串中解析JSON。
    更稳健地处理不同的代码块标记和潜在的额外文本。
    """
    text = content.strip()
    json_string = None

    if text.startswith("```json"):
        parts = text.split("```json", 1)
        if len(parts) > 1:
            json_string = parts[1].split("```", 1)[0].strip()
    elif text.startswith("```"):
        parts = text.split("```", 1)
        if len(parts) > 1:
            json_string = parts[1].split("```", 1)[0].strip()
    elif text.startswith("{") and text.endswith("}"):
        json_string = text
    elif text.startswith("[") and text.endswith("]"):
        json_string = text

    if not json_string:
        logger.warning(f"未在响应中找到可能的JSON字符串。\n原始内容: {content[:200]}...")
        return None

    try:
        parsed_data = json.loads(json_string)
        if isinstance(parsed_data, (dict, list)):
            logger.debug(f"JSON解析成功，类型: {type(parsed_data)}")
            return parsed_data
        else:
            logger.warning(f"JSON解析成功但内容不是字典或列表: {type(parsed_data)}.\n原始内容: {json_string[:100]}...")
            return None
    except json.JSONDecodeError as e:
        logger.warning(f"JSON解码失败: {e}\n尝试解析的字符串:\n{json_string[:200]}...")
        return None
    except Exception as e:
        logger.warning(f"JSON解析时发生意外错误: {e}\n尝试解析的字符串:\n{json_string[:200]}...")
        return None


def generate_unique_id() -> str:
    """生成一个带前缀的唯一ID。"""
    return f"N_{uuid.uuid4().hex}"


def generate_dynamic_color(node_type: str, color_map: dict) -> str:
    """根据节点类型生成或获取颜色。"""
    if node_type in color_map:
        return color_map[node_type]
    hue = (len(color_map) * 0.618033988749895) % 1.0
    rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
    color = '#{:02x}{:02x}{:02x}'.format(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
    color_map[node_type] = color
    return color


# ---------------- 异步 LLM 服务 (使用 AsyncOpenAI 客户端) ----------------
class LLMService:
    """LLM服务类，用于与大语言模型进行异步交互。"""

    def __init__(self, llm_config: dict):
        """
        初始化LLM服务。

        参数:
            llm_config (dict): 包含LLM特定配置的字典，如 model, api_key, base_url等。
        """
        self.config = llm_config
        self.model_name = self.config.get("model", Config.LLM_TYPE)
        self.api_key = self.config.get("api_key")
        self.base_url = self.config.get("base_url")
        self.max_concurrency = self.config.get("max_concurrency")
        self.timeout_seconds = self.config.get("timeout_seconds")
        self.retry_attempts = self.config.get("retry_attempts")

        try:
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout_seconds,
            )
            logger.info(
                f"已为模型 '{self.model_name}' 初始化 AsyncOpenAI 客户端，URL: '{self.base_url or '默认'}'")
        except Exception as e:
            logger.error(f"初始化 AsyncOpenAI 客户端失败: {e}")
            raise

        self.semaphore = asyncio.Semaphore(self.max_concurrency)
        self.cache: dict[str, str] = {}

    async def _call_model(self, prompt: str) -> str:
        """
        对LLM进行底层异步调用。

        参数:
            prompt (str): 发送给LLM的提示。

        返回:
            str: LLM生成的文本响应。

        引发:
            Timeout, APIError: 如果发生特定于OpenAI的错误。
            Exception: 其他LLM调用错误。
        """
        messages = [{"role": "user", "content": prompt}]
        try:
            resp = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
            )
            return resp.choices[0].message.content.strip()
        except Timeout as e:
            logger.warning(f"LLM 调用超时: {e}")
            raise
        except APIError as e:
            logger.warning(f"LLM API 错误 (状态码: {e.status_code}): {e.message}")
            raise
        except Exception as e:
            logger.warning(f"LLM 调用发生未知错误: {e}")
            raise

    async def generate_async(self, prompts: list[str]) -> list[str | None]:
        """
        批量异步生成文本，具有并发控制和重试机制。

        参数:
            prompts (list[str]): 提示字符串列表。

        返回:
            list[str | None]: 对应每个提示的生成结果列表，如果某个提示最终失败则为None。
        """

        async def task_fn(p: str) -> str | None:
            """处理单个提示的函数。"""
            prompt_key = p
            if prompt_key in self.cache:
                logger.debug(f"缓存命中: {prompt_key[:50]}...")
                return self.cache[prompt_key]

            async with self.semaphore:
                for i in range(self.retry_attempts):
                    try:
                        logger.debug(
                            f"尝试Prompt {i + 1}/{self.retry_attempts} (并发: {self.max_concurrency - self.semaphore._value}/{self.max_concurrency}): {prompt_key[:50]}...")
                        res = await self._call_model(p)
                        self.cache[prompt_key] = res
                        logger.debug(f"Prompt成功: {prompt_key[:50]}...")
                        return res
                    except Exception as e:
                        logger.warning(
                            f"Prompt尝试 {i + 1}/{self.retry_attempts} 失败: {e} (Prompt: {prompt_key[:50]}...)")
                        if i < self.retry_attempts - 1:
                            wait_time = 1 * (2 ** i) + (0.5 * random.random())  # 指数退避加抖动
                            logger.info(f"等待 {wait_time:.2f} 秒后重试...")
                            await asyncio.sleep(wait_time)
                        else:
                            logger.error(f"Prompt重试次数已用尽，放弃: {prompt_key[:100]}...")
                            return None

        tasks = [asyncio.create_task(task_fn(p)) for p in prompts]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        return results

    async def close(self):
        """关闭异步客户端连接。"""
        if hasattr(self.client, 'close'):
            try:
                await self.client.close()
            except Exception as e:
                logger.warning(f"关闭LLM客户端时发生错误: {e}")


# ---------------- 本体加载 ----------------
def load_ontology(path: Path) -> dict:
    """
    从文件加载本体并组织其结构。

    参数:
        path (Path): 本体JSON文件的路径。

    返回:
        dict: 包含解析后本体数据的字典。

    引发:
        FileNotFoundError: 如果本体文件未找到。
        json.JSONDecodeError: 如果JSON文件格式错误。
        Exception: 其他加载错误。
    """
    if not path.exists():
        logger.error(f"本体文件未找到: {path}")
        raise FileNotFoundError(f"本体文件未找到: {path}")
    try:
        with open(path, encoding='utf-8') as f:
            data = json.load(f)

        entity_types = {e['type']: e for e in data.get('entity_types', []) if isinstance(e, dict) and 'type' in e}

        relations_by_src = defaultdict(list)
        relationships_list = data.get('relationships', [])
        if not isinstance(relationships_list, list):
            logger.warning(f"本体文件中的 'relationships' 不是列表，默认为空列表。")
            relationships_list = []

        valid_relationships = []
        for r in relationships_list:
            if isinstance(r, dict) and all(k in r for k in ['src_type', 'tgt_type', 'relation']):
                if r['src_type'] in entity_types and r['tgt_type'] in entity_types:
                    relations_by_src[r['src_type']].append(r)
                    valid_relationships.append(r)
                else:
                    logger.warning(
                        f"本体关系定义中使用了未定义的实体类型 (src: {r.get('src_type')}, tgt: {r.get('tgt_type')}), 跳过关系: {r.get('relation', '未知')}")
            else:
                logger.warning(f"本体文件中的关系定义结构不完整，跳过: {r}")

        props_data = data.get('properties', {})
        if not isinstance(props_data, dict):
            logger.warning(f"本体文件中的 'properties' 不是字典，默认为空字典。")
            props_data = {}

        cleaned_props = {}
        for ent_type, prop_list in props_data.items():
            if ent_type not in entity_types:
                logger.warning(f"本体属性定义中使用了未在实体类型中定义的类型 '{ent_type}'，跳过此属性模板。")
                continue
            if isinstance(prop_list, list) and all(isinstance(item, str) for item in prop_list):
                cleaned_props[ent_type] = prop_list
            else:
                logger.warning(f"本体属性定义 '{ent_type}' 格式不正确 (应为字符串列表)，将使用默认属性 ['description']。")
                cleaned_props[ent_type] = ['description']

        logger.info(
            f"本体加载成功: {len(entity_types)} 个实体类型, {len(valid_relationships)} 个有效关系定义, {len(cleaned_props)} 个属性模板.")
        return {
            'original': data,
            'entity_types': entity_types,
            'relationships_by_src': relations_by_src,
            'relationships': valid_relationships,
            'properties': cleaned_props
        }
    except json.JSONDecodeError as e:
        logger.exception(f"无法解析本体文件，请检查JSON格式: {path}")
        raise
    except Exception as e:
        logger.exception(f"加载本体文件时发生错误: {e}")
        raise


# ---------------- Agent ----------------
class NodeGeneratorAgent:
    """节点生成智能体，负责根据父节点和本体生成子节点候选项。"""

    def __init__(self, ontology: dict, llm: LLMService):
        self.ontology = ontology
        self.llm = llm

    async def generate_children_batch(self, parent_label: str, parent_type: str, rels: list, path_context: str) -> dict:
        """
        批量异步生成子节点候选项。

        参数:
            parent_label (str): 父节点标签。
            parent_type (str): 父节点类型。
            rels (list): 本体中定义的从父节点类型出发的可能关系列表。
            path_context (str): 当前图谱路径上下文。

        返回:
            dict: 格式为 { (child_type, relation): [child_label1, child_label2], ... }
        """
        prompts = []
        prompt_key_map = {}  # 将prompt索引映射到 (child_type, relation)

        valid_rels = [r for r in rels if r['tgt_type'] in self.ontology['entity_types']]
        if not valid_rels:
            logger.debug(f"父节点类型 '{parent_type}' 在本体中没有找到有效的出度关系。")
            return {}

        logger.info(f"为父节点 '{parent_label}' ({parent_type}) 准备生成 {len(valid_rels)} 个子节点 prompts")

        for idx, r in enumerate(valid_rels):  # idx 从0开始
            type_desc = self.ontology['entity_types'].get(r['tgt_type'], {}).get('description', '无描述')
            rel_desc = r.get('description', '无描述')
            parent_type_desc = self.ontology['entity_types'].get(parent_type, {}).get('description', '无描述')

            prompt_text = (
                f"基于以下信息，为父节点 '{parent_label}' (类型 {parent_type}, 含义: {parent_type_desc}) 生成其直接子节点列表。\n"
                f"期望生成的子节点类型: {r['tgt_type']} (含义: {type_desc})\n"
                f"父子关系类型: {r['relation']} (含义: {rel_desc})\n"
                f"当前上下文路径 : {path_context}\n"
                f"\n请以 JSON 列表的形式返回子节点的名称 (label) 字符串，例如: [\"子节点名称1\", \"子节点名称2\", ...]。\n"
                f"请严格按照 JSON 列表格式输出，不要包含任何其他前置、后置文本或解释。\n"
                f"如果无法生成任何合理的子节点，返回空列表 []。"
            )
            prompts.append(prompt_text)
            prompt_key_map[idx] = (r['tgt_type'], r['relation'])

        if not prompts: return {}
        texts = await self.llm.generate_async(prompts)

        out = {}
        for i, txt in enumerate(texts):
            key = prompt_key_map[i]
            if txt is None:
                logger.warning(f"LLM生成子节点失败 (Prompt {i + 1}, 对应关系: {key})")
                out[key] = []
                continue

            arr = parse_response_json(txt)
            if isinstance(arr, list):
                cleaned_labels = [item.strip() for item in arr if isinstance(item, str) and item.strip()]
                logger.debug(f"为关系 {key} 生成的子节点: {cleaned_labels}")
                out[key] = cleaned_labels
            else:
                logger.warning(
                    f"LLM生成子节点返回格式错误 (Prompt {i + 1}, 对应关系: {key}): 返回内容非列表。\n原始内容:\n{txt[:200]}...")
                out[key] = []
        return out


class NodeValidationAgent:
    """节点验证智能体，负责验证节点间关系的合理性。"""
    def __init__(self, ontology: dict, llm: LLMService):
        self.ontology = ontology
        self.llm = llm
        self._rel_descriptions = {}
        for rel in self.ontology.get('relationships', []):
            key = (rel.get('src_type'), rel.get('tgt_type'), rel.get('relation'))  # 同一对实体之间可能有多种关系
            self._rel_descriptions[key] = rel.get('description', '无描述')

    async def validate_nodes_batch(self, reqs: list) -> dict:
        """
        批量异步验证节点和关系的合理性。

        参数:
            reqs (list): 字典列表，每个字典包含 'parent_node', 'parent_type', 'child_node', 'child_type', 'relationship'。

        返回:
            dict: 格式为 { req_index_str (从'0'开始的字符串索引): {'valid':true/false, 'reason':?}, ... }
        """
        if not reqs: return {}
        prompts = []
        logger.info(f"准备验证 {len(reqs)} 个节点关系 prompts")

        for r in reqs:
            rel_key = (r['parent_type'], r['child_type'], r['relationship'])
            rel_desc = self._rel_descriptions.get(rel_key, '本体中未找到此关系描述')

            parent_type_desc = self.ontology['entity_types'].get(r['parent_type'], {}).get('description', '无描述')
            child_type_desc = self.ontology['entity_types'].get(r['child_type'], {}).get('description', '无描述')

            prompt_text = (
                f"请判断以下父子节点及其关系是否合理，并符合知识图谱的常识:\n"
                f"父节点: '{r['parent_node']}' (类型 {r['parent_type']}, 含义: {parent_type_desc})\n"
                f"子节点: '{r['child_node']}' (类型 {r['child_type']}, 含义: {child_type_desc})\n"
                f"关系类型: {r['relationship']} (含义: {rel_desc})\n"
                f"\n请以 JSON 对象的形式返回判断结果，例如:\n"
                f"{{ \"valid\": true, \"reason\": \"判断理由或空字符串\" }}\n"
                f"请严格按照此 JSON 格式输出，不要包含任何其他文本或解释。\n"
                f"如果认为关系合理且节点存在意义，valid为 true；否则 valid为 false，并在 reason 中简要说明原因。"
            )
            prompts.append(prompt_text)

        if not prompts: return {}
        texts = await self.llm.generate_async(prompts)

        out = {}
        for i, txt in enumerate(texts):
            req_key = str(i)
            if txt is None:
                logger.warning(f"LLM验证节点失败 (Prompt {i + 1})")
                out[req_key] = {'valid': False, 'reason': 'LLM调用失败'}
                continue

            res = parse_response_json(txt)
            if res is None or not isinstance(res, dict) or 'valid' not in res:
                logger.warning(
                    f"LLM验证节点返回格式错误 (Prompt {i + 1}): 返回内容非预期JSON对象。\n原始内容:\n{txt[:200]}...")
                out[req_key] = {'valid': False, 'reason': 'LLM返回格式异常'}
            else:
                out[req_key] = {
                    'valid': bool(res.get('valid', False)),
                    'reason': str(res.get('reason', ''))
                }
                logger.debug(f"节点验证结果 (Prompt {i + 1}): {out[req_key]}")
        return out


class PropertyGeneratorAgent:
    """属性生成智能体，负责为节点生成属性。"""

    def __init__(self, ontology: dict, llm: LLMService):
        self.ontology = ontology
        self.llm = llm

    async def generate_properties_batch(self, reqs: list) -> dict:
        """
        批量异步生成节点属性。

        参数:
            reqs (list): 字典列表，每个字典包含 'node_label', 'node_type', 'full_path'。
        返回:
            dict: 格式为 { node_label: {prop_name: prop_value, ...}, ... }
        """
        if not reqs: return {}
        prompts = []
        prompt_label_map = {}

        logger.info(f"准备生成 {len(reqs)} 个节点属性 prompts")

        for idx, r in enumerate(reqs):
            node_type = r['node_type']
            template_props = self.ontology['properties'].get(node_type, ['description'])
            template_str = json.dumps(template_props, ensure_ascii=False)
            node_type_desc = self.ontology['entity_types'].get(node_type, {}).get('description', '无描述')

            prompt_text = (
                f"请为以下知识图谱节点生成属性值:\n"
                f"节点名称: '{r['node_label']}'\n"
                f"节点类型: {node_type} (含义: {node_type_desc})\n"
                f"知识图谱路径 : {r['full_path']}\n"
                f"需要生成的属性列表 : {template_str}\n"
                f"\n请以 JSON 对象的形式返回属性及其对应的值，例如:\n"
                f"{{ \"属性名1\": \"属性值1\", \"属性名2\": \"属性值2\", ... }}\n"
                f"请严格按照此 JSON 格式输出，不要包含任何其他文本或解释。\n"
                f"确保包含所有需要生成的属性，如果某个属性无法确定，可以赋值为空字符串或\"未知\"。"
            )
            prompts.append(prompt_text)
            prompt_label_map[idx] = r['node_label']

        if not prompts: return {}
        texts = await self.llm.generate_async(prompts)

        out = {}
        for i, txt in enumerate(texts):
            node_label = prompt_label_map[i]
            if txt is None:
                logger.warning(f"LLM生成属性失败 (Prompt {i + 1}, 节点: {node_label})")
                out[node_label] = {}
                continue

            props_data = parse_response_json(txt)
            if props_data is None or not isinstance(props_data, dict):
                logger.warning(
                    f"LLM生成属性返回格式错误 (Prompt {i + 1}, 节点: {node_label}): 返回内容非预期JSON对象。\n原始内容:\n{txt[:200]}...")
                out[node_label] = {}
            else:
                cleaned_props = {
                    k: v for k, v in props_data.items()
                    if isinstance(k, str) and (v is None or isinstance(v, (str, int, float, bool)))
                }
                out[node_label] = cleaned_props
                logger.debug(f"为节点 '{node_label}' 生成的属性: {out[node_label]}")
        return out


class PropertyValidationAgent:
    """属性验证智能体，负责验证生成的节点属性是否符合本体要求。"""
    def __init__(self, ontology: dict, llm: LLMService):
        self.ontology = ontology
        self.llm = llm

    async def validate_properties_batch(self, reqs: list) -> dict:
        """
        批量异步验证生成的节点属性。

        参数:
            reqs (list): 字典列表，每个字典包含 'node_label', 'node_type', 'properties'。

        返回:
            dict: 格式为 { node_label: {'valid':true/false, 'reason':?}, ... }
        """
        if not reqs: return {}
        prompts = []
        prompt_label_map = {}

        logger.info(f"准备验证 {len(reqs)} 组节点属性 prompts")

        for idx, r in enumerate(reqs):
            node_type = r['node_type']
            template_props = self.ontology['properties'].get(node_type, [])
            prop_str = json.dumps(r['properties'], ensure_ascii=False)
            template_str = json.dumps(template_props, ensure_ascii=False)
            node_type_desc = self.ontology['entity_types'].get(node_type, {}).get('description', '无描述')

            prompt_text = (
                f"请验证以下节点的属性是否合理，并确保包含了针对类型 '{node_type}' 的所有必需属性:\n"
                f"节点名称: '{r['node_label']}'\n"
                f"节点类型: {node_type} (含义: {node_type_desc})\n"
                f"生成的属性: {prop_str}\n"
                f"必需的属性列表: {template_str}\n"
                f"注意: 如果必需属性列表为空 ([])，则认为属性是有效的 (除非属性本身明显不合理)。\n"
                f"\n请以 JSON 对象的形式返回判断结果，例如:\n"
                f"{{ \"valid\": true, \"reason\": \"判断理由或空字符串\" }}\n"
                f"请严格按照此 JSON 格式输出，不要包含任何其他文本或解释。\n"
                f"如果属性合理且包含所有必需属性，valid为 true；否则 valid为 false，并在 reason 中简要说明原因。"
            )
            prompts.append(prompt_text)
            prompt_label_map[idx] = r['node_label']

        if not prompts: return {}
        texts = await self.llm.generate_async(prompts)

        out = {}
        for i, txt in enumerate(texts):
            node_label = prompt_label_map[i]
            if txt is None:
                logger.warning(f"LLM验证属性失败 (Prompt {i + 1}, 节点: {node_label})")
                out[node_label] = {'valid': False, 'reason': 'LLM调用失败'}
                continue

            res = parse_response_json(txt)
            if res is None or not isinstance(res, dict) or 'valid' not in res:
                logger.warning(
                    f"LLM验证属性返回格式错误 (Prompt {i + 1}, 节点: {node_label}): 返回内容非预期JSON对象。\n原始内容:\n{txt[:200]}...")
                out[node_label] = {'valid': False, 'reason': 'LLM返回格式异常'}
            else:
                out[node_label] = {
                    'valid': bool(res.get('valid', False)),
                    'reason': str(res.get('reason', ''))
                }
                logger.debug(f"属性验证结果 (Prompt {i + 1}, 节点 '{node_label}'): {out[node_label]}")
        return out


# ---------------- 构建器 ----------------
class KnowledgeGraphBuilder:
    """知识图谱构建器类，负责协调整个构建过程。"""
    def __init__(self, domain: str = None, root_type: str = None):
        """
        初始化知识图谱构建器。

        参数:
            domain (str): 知识图谱的领域名称
            root_type (str): 根节点类型
        """
        self.ontology = None
        self.llm = None
        self.domain = domain
        self.root_type = root_type
        self.nodes = {}
        self.relationships = []
        self.color_map = {}
        self.processed_nodes = set()  # 用于追踪已处理的节点，避免重复
        self.batch_size = 8  # 批处理大小
        self.node_generator = None
        self.node_validator = None
        self.property_generator = None
        self.property_validator = None


    async def _process_node_properties(self, node_label: str, node_type: str, path_context: str) -> dict:
        """处理单个节点的属性生成和验证。"""
        if not self.property_generator or not self.property_validator:
            raise RuntimeError("属性生成器和验证器未初始化")

        property_reqs = [{
            "node_label": node_label,
            "node_type": node_type,
            "full_path": path_context
        }]
        generated_props = await self.property_generator.generate_properties_batch(property_reqs)

        validation_reqs = [{
            "node_label": node_label,
            "node_type": node_type,
            "properties": generated_props.get(node_label, {})
        }]
        validated_props = await self.property_validator.validate_properties_batch(validation_reqs)

        validation_result = validated_props.get(node_label, {})
        if not validation_result.get('valid', False):
            logger.warning(f"节点 '{node_label}' 的属性验证失败: {validation_result.get('reason', '')}")

        return generated_props.get(node_label, {})

    async def _process_properties_in_batches(self, nodes_to_process: list):
        """批量处理节点属性（每批最多 self.batch_size 个）"""
        if not nodes_to_process:
            return

        batches = []
        current_batch = []
        for node in nodes_to_process:
            current_batch.append(node)
            if len(current_batch) >= self.batch_size:
                batches.append(current_batch)
                current_batch = []
        if current_batch:
            batches.append(current_batch)

        for batch_idx, batch in enumerate(batches):
            logger.info(f"处理属性批次 {batch_idx + 1}/{len(batches)}，包含 {len(batch)} 个节点")

            # 批量生成属性
            prop_requests = []
            for node_id, node_label, node_type, full_path in batch:
                prop_requests.append({
                    "node_label": node_label,
                    "node_type": node_type,
                    "full_path": full_path
                })

            generated_props = await self.property_generator.generate_properties_batch(prop_requests)

            # 批量验证属性
            val_requests = []
            for node_id, node_label, node_type, _ in batch:
                props = generated_props.get(node_label, {})
                val_requests.append({
                    "node_label": node_label,
                    "node_type": node_type,
                    "properties": props
                })

            validated_props = await self.property_validator.validate_properties_batch(val_requests)

            # 应用属性到节点
            for node_id, node_label, node_type, _ in batch:
                props = generated_props.get(node_label, {})
                validation_result = validated_props.get(node_label, {})
                if not validation_result.get('valid', True):
                    logger.warning(f"节点 '{node_label}' 的属性验证失败: {validation_result.get('reason', '')}")
                self.nodes[node_id]['properties'] = props

    async def build(self) -> dict:
        """
        构建知识图谱。

        返回:
            dict: 包含节点和关系的知识图谱数据。
        """
        try:
            # 验证必要参数
            if not self.domain:
                raise ValueError("领域名称(domain)不能为空")
            if not self.root_type:
                raise ValueError("根节点类型(root_type)不能为空")
            if not hasattr(self, 'ontology') or self.ontology is None:
                raise ValueError("本体文件未加载，请先调用 build_with_ontology() 方法")
            entity_types = {et["type"] for et in self.ontology["entity_types"]}
            if self.root_type not in entity_types:
                raise ValueError(f"指定的根节点类型 '{self.root_type}' 不存在于本体定义中。可用的类型有: {', '.join(entity_types)}")

            # 初始化LLM服务
            if not self.llm:
                llm_config = Config.get_llm_config()
                self.llm = LLMService(llm_config)

            if not self.node_generator:
                self.node_generator = NodeGeneratorAgent(self.ontology, self.llm)
                self.node_validator = NodeValidationAgent(self.ontology, self.llm)
                self.property_generator = PropertyGeneratorAgent(self.ontology, self.llm)
                self.property_validator = PropertyValidationAgent(self.ontology, self.llm)

            logger.info(f'开始构建 "{self.domain}" 领域知识图谱')

            # 如果还没有根节点，创建根节点
            if not self.nodes:
                root_id = generate_unique_id()
                root_node = {
                    "id": root_id,
                    "label": self.domain,
                    "type": self.root_type,
                    "color": generate_dynamic_color(self.root_type, self.color_map),
                    "properties": {}
                }
                self.nodes[root_id] = root_node
                self.processed_nodes.add(self.domain)

                # 处理根节点的属性
                logger.info("处理根节点属性")
                root_properties = await self._process_node_properties(self.domain, self.root_type, self.domain)
                self.nodes[root_id]["properties"] = root_properties

            # 使用BFS构建知识图谱
            queue = deque()

            # 将根节点加入队列
            for node_id, node_info in self.nodes.items():
                if node_info.get('type') == self.root_type:
                    queue.append((node_id, node_info['label'], node_info['type'], node_info['label']))
                    break

            # BFS逐层扩展
            while queue:
                current_id, current_label, current_type, full_path = queue.popleft()
                logger.info(f"处理节点: {current_label} (类型: {current_type})")
                logger.info(f"   路径: {full_path}")

                # 获取当前节点类型的可能关系
                possible_relations = self.ontology['relationships_by_src'].get(current_type, [])
                if not possible_relations:
                    logger.info(f"节点类型 {current_type} 无子节点关系")
                    continue

                # 生成子节点候选项
                gen_result = await self.node_generator.generate_children_batch(
                    current_label, current_type, possible_relations, full_path
                )

                # 准备验证请求
                validation_requests = []
                for (child_type, relationship), children in gen_result.items():
                    for child_name in children:
                        validation_requests.append({
                            "child_node": child_name,
                            "parent_node": current_label,
                            "parent_type": current_type,
                            "child_type": child_type,
                            "relationship": relationship
                        })

                # 批量验证子节点
                if validation_requests:
                    valid_result = await self.node_validator.validate_nodes_batch(validation_requests)

                    # 处理验证结果，添加有效节点
                    valid_children = []
                    added_nodes = 0

                    for req_idx, req in enumerate(validation_requests):
                        child = req["child_node"]
                        key = str(req_idx)
                        valid_info = valid_result.get(key, {"valid": True})

                        if valid_info.get("valid", True):
                            if child not in self.processed_nodes:
                                # 创建新节点
                                child_id = generate_unique_id()
                                child_color = generate_dynamic_color(req["child_type"], self.color_map)
                                self.nodes[child_id] = {
                                    "id": child_id,
                                    "label": child,
                                    "type": req["child_type"],
                                    "properties": {},
                                    "color": child_color
                                }

                                # 添加关系
                                self.relationships.append({
                                    "src_id": current_id,
                                    "tgt_id": child_id,
                                    "type": req["relationship"]
                                })

                                # 添加到队列和待处理列表
                                new_path = f"{full_path} -- {req['relationship']} --> {child}"
                                valid_children.append((child_id, child, req["child_type"], new_path))
                                queue.append((child_id, child, req["child_type"], new_path))
                                self.processed_nodes.add(child)
                                added_nodes += 1

                            else:
                                # 节点已存在，只添加关系
                                existing_node_id = None
                                for node_id, node_info in self.nodes.items():
                                    if node_info["label"] == child:
                                        existing_node_id = node_id
                                        break

                                if existing_node_id is not None:
                                    # 检查关系是否已存在
                                    relationship_exists = any(
                                        rel for rel in self.relationships
                                        if (rel["src_id"] == current_id and
                                            rel["tgt_id"] == existing_node_id and
                                            rel["type"] == req["relationship"])
                                    )
                                    if not relationship_exists:
                                        self.relationships.append({
                                            "src_id": current_id,
                                            "tgt_id": existing_node_id,
                                            "type": req["relationship"]
                                        })
                        else:
                            logger.debug(f"节点验证失败: {child}, 原因: {valid_info.get('reason', '')}")

                    if added_nodes > 0:
                        logger.info(f"添加了 {added_nodes} 个子节点（经过验证）")
                        # 批量处理新节点的属性
                        await self._process_properties_in_batches(valid_children)

                # 添加延迟以避免过于频繁的API调用
                await asyncio.sleep(1)

            logger.info(f"知识图谱构建完成！总节点数: {len(self.nodes)}，总关系数: {len(self.relationships)}")

            return {
                "nodes": self.nodes,
                "relationships": self.relationships
            }

        except Exception as e:
            logger.error(f"构建知识图谱时发生错误: {e}")
            raise
        finally:
            if self.llm:
                await self.llm.close()

    async def build_with_ontology(self, ontology_path: str) -> dict:
        """
        使用指定的本体文件路径构建知识图谱

        参数:
            ontology_path (str): 本体文件的完整路径

        返回:
            dict: 包含节点和关系的知识图谱数据
        """
        try:
            if not self.domain:
                raise ValueError("领域名称(domain)不能为空")
            if not self.root_type:
                raise ValueError("根节点类型(root_type)不能为空")

            self.ontology = load_ontology(Path(ontology_path))

            logger.info(f"本体数据结构: {type(self.ontology)}")
            logger.info(f"本体keys: {self.ontology.keys()}")
            logger.info(f"entity_types类型: {type(self.ontology.get('entity_types', 'NOT_FOUND'))}")
            logger.info(f"entity_types内容: {self.ontology.get('entity_types', 'NOT_FOUND')}")

            entity_types_dict = self.ontology.get("entity_types", {})
            if isinstance(entity_types_dict, dict):
                entity_types = set(entity_types_dict.keys())
            else:
                entity_types = {et["type"] for et in entity_types_dict if isinstance(et, dict) and "type" in et}

            logger.info(f"提取到的实体类型: {entity_types}")

            if self.root_type not in entity_types:
                raise ValueError(
                    f"指定的根节点类型 '{self.root_type}' 不存在于本体定义中。可用的类型有: {', '.join(entity_types)}")

            llm_config = Config.get_llm_config()
            self.llm = LLMService(llm_config)

            self.node_generator = NodeGeneratorAgent(self.ontology, self.llm)
            self.node_validator = NodeValidationAgent(self.ontology, self.llm)
            self.property_generator = PropertyGeneratorAgent(self.ontology, self.llm)
            self.property_validator = PropertyValidationAgent(self.ontology, self.llm)

            logger.info(f'开始构建 "{self.domain}" 领域知识图谱')

            # 如果还没有根节点，创建根节点
            if not self.nodes:
                root_id = generate_unique_id()
                root_node = {
                    "id": root_id,
                    "label": self.domain,
                    "type": self.root_type,
                    "color": generate_dynamic_color(self.root_type, self.color_map),
                    "properties": {}
                }
                self.nodes[root_id] = root_node
                self.processed_nodes.add(self.domain)

                # 处理根节点的属性
                logger.info("处理根节点属性")
                root_properties = await self._process_node_properties(self.domain, self.root_type, self.domain)
                self.nodes[root_id]["properties"] = root_properties

            # 使用BFS构建知识图谱
            queue = deque()

            # 将根节点加入队列
            for node_id, node_info in self.nodes.items():
                if node_info.get('type') == self.root_type:
                    queue.append((node_id, node_info['label'], node_info['type'], node_info['label']))
                    break

            # BFS逐层扩展（复制原build方法的核心逻辑）
            while queue:
                current_id, current_label, current_type, full_path = queue.popleft()
                logger.info(f"处理节点: {current_label} (类型: {current_type})")
                logger.info(f"   路径: {full_path}")

                # 获取当前节点类型的可能关系
                possible_relations = self.ontology['relationships_by_src'].get(current_type, [])
                if not possible_relations:
                    logger.info(f"节点类型 {current_type} 无子节点关系")
                    continue

                # 生成子节点候选项
                gen_result = await self.node_generator.generate_children_batch(
                    current_label, current_type, possible_relations, full_path
                )

                # 准备验证请求
                validation_requests = []
                for (child_type, relationship), children in gen_result.items():
                    for child_name in children:
                        validation_requests.append({
                            "child_node": child_name,
                            "parent_node": current_label,
                            "parent_type": current_type,
                            "child_type": child_type,
                            "relationship": relationship
                        })

                # 批量验证子节点
                if validation_requests:
                    valid_result = await self.node_validator.validate_nodes_batch(validation_requests)

                    # 处理验证结果，添加有效节点
                    valid_children = []
                    added_nodes = 0

                    for req_idx, req in enumerate(validation_requests):
                        child = req["child_node"]
                        key = str(req_idx)
                        valid_info = valid_result.get(key, {"valid": True})

                        if valid_info.get("valid", True):
                            if child not in self.processed_nodes:
                                # 创建新节点
                                child_id = generate_unique_id()
                                child_color = generate_dynamic_color(req["child_type"], self.color_map)
                                self.nodes[child_id] = {
                                    "id": child_id,
                                    "label": child,
                                    "type": req["child_type"],
                                    "properties": {},
                                    "color": child_color
                                }

                                # 添加关系
                                self.relationships.append({
                                    "src_id": current_id,
                                    "tgt_id": child_id,
                                    "type": req["relationship"]
                                })

                                # 添加到队列和待处理列表
                                new_path = f"{full_path} -- {req['relationship']} --> {child}"
                                valid_children.append((child_id, child, req["child_type"], new_path))
                                queue.append((child_id, child, req["child_type"], new_path))
                                self.processed_nodes.add(child)
                                added_nodes += 1

                            else:
                                # 节点已存在，只添加关系（如果不存在的话）
                                existing_node_id = None
                                for node_id, node_info in self.nodes.items():
                                    if node_info["label"] == child:
                                        existing_node_id = node_id
                                        break

                                if existing_node_id is not None:
                                    # 检查关系是否已存在
                                    relationship_exists = any(
                                        rel for rel in self.relationships
                                        if (rel["src_id"] == current_id and
                                            rel["tgt_id"] == existing_node_id and
                                            rel["type"] == req["relationship"])
                                    )
                                    if not relationship_exists:
                                        self.relationships.append({
                                            "src_id": current_id,
                                            "tgt_id": existing_node_id,
                                            "type": req["relationship"]
                                        })
                        else:
                            logger.debug(f"节点验证失败: {child}, 原因: {valid_info.get('reason', '')}")

                    if added_nodes > 0:
                        logger.info(f"添加了 {added_nodes} 个子节点（经过验证）")
                        # 批量处理新节点的属性
                        await self._process_properties_in_batches(valid_children)

                # 添加延迟以避免过于频繁的API调用
                await asyncio.sleep(1)

            logger.info(f"知识图谱构建完成！总节点数: {len(self.nodes)}，总关系数: {len(self.relationships)}")

            return {
                "nodes": self.nodes,
                "relationships": self.relationships
            }

        except Exception as e:
            logger.error(f"构建知识图谱时发生错误: {e}")
            raise
        finally:
            if self.llm:
                await self.llm.close()

    def save(self, path: Path = None):
        """保存知识图谱到文件。"""
        if path is None:
            path = Path(f"KG/{self.domain}_kg.json")

        try:
            path.parent.mkdir(parents=True, exist_ok=True)

            kg_copy = {
                "domain": self.domain,
                "nodes": {},
                "relationships": self.relationships
            }

            for node_id, node in self.nodes.items():
                kg_copy['nodes'][node_id] = node.copy()
                if 'color' not in kg_copy['nodes'][node_id]:
                    kg_copy['nodes'][node_id]['color'] = self.color_map.get(node['type'], '#CCCCCC')

            with open(path, 'w', encoding='utf-8') as f:
                json.dump(kg_copy, f, ensure_ascii=False, indent=2)
            logger.info(f"知识图谱已保存到: {path}")
        except Exception as e:
            logger.error(f"保存知识图谱时发生错误: {e}")
            raise

