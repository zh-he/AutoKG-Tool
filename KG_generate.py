import colorsys
import json
import os
import time
import uuid
from collections import deque, defaultdict

from openai import OpenAI

from neo4j_import import Neo4jImporter  # 自定义模块，用于导入到 Neo4j
from config import Config  # 使用统一的配置管理

# ---------------- 使用 Config 配置 ----------------
NEO4J_URI = Config.NEO4J_URI
NEO4J_USERNAME = Config.NEO4J_USERNAME
NEO4J_PASSWORD = Config.NEO4J_PASSWORD

llm_config = Config.get_llm_config()
API_KEY = llm_config.get("api_key")
BASE_URL = llm_config.get("base_url")

# 创建 OpenAI 客户端
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL  # 根据实际使用的 API 更改
)

# ---------------- 通用工具函数 ----------------
def parse_response_json(content: str):
    """
    尝试提取 API 返回文本中的 JSON 部分，并解析为字典。
    支持 "```json" 或 "```" 包围的格式。
    """
    try:
        if content.startswith("```json"):
            content = content.split("```json", 1)[1].split("```", 1)[0].strip()
        elif content.startswith("```"):
            content = content.split("```", 1)[1].split("```", 1)[0].strip()
        return json.loads(content)
    except Exception as e:
        print(f"⚠️ JSON解析失败: {e}")
        print(f"响应内容: {content[:100]}...")
        return None


def ask_chat_model(prompt: str, max_retries: int = 10, temperature: float = 0.1, sleep_seconds: float = 1.5):
    """
    通用的聊天模型调用函数，内置最多 max_retries 次重试。
    返回成功解析后的 JSON (dict)，若多次失败则返回 None。
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=Config.LLM_TYPE,  # 直接使用 Config 中的 LLM_TYPE
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature
            )
            content = response.choices[0].message.content.strip()
            parsed = parse_response_json(content)
            if parsed is not None:
                return parsed
            else:
                print(f"⚠️ JSON 解析失败或返回空，尝试第 {attempt + 1}/{max_retries} 次...")
        except Exception as e:
            print(f"⚠️ 调用大模型失败: {e} - 第 {attempt + 1}/{max_retries} 次重试")
        time.sleep(sleep_seconds)
    print("❌ 已达最大重试次数，依然无法获得有效结果，返回 None。")
    return None


def generate_unique_id(label: str) -> str:
    """使用 UUID 生成唯一节点ID"""
    return f"N_{uuid.uuid4().hex}"


def generate_dynamic_color(node_type: str, color_map: dict) -> str:
    """
    生成动态且唯一的颜色。
    如果节点类型已有颜色，则直接返回；否则使用 HSV 色彩空间生成新颜色。
    """
    if node_type in color_map:
        return color_map[node_type]

    # 使用黄金分割角度生成颜色，确保颜色区分度高
    hue = (len(color_map) * 0.618033988749895) % 1.0
    saturation = 0.7
    value = 0.9

    rgb = colorsys.hsv_to_rgb(hue, saturation, value)
    color = '#{:02x}{:02x}{:02x}'.format(
        int(rgb[0] * 255),
        int(rgb[1] * 255),
        int(rgb[2] * 255)
    )
    color_map[node_type] = color
    return color

# ---------------- 加载本体层定义 ----------------
def load_ontology(file_path: str = "ontology.json"):
    """加载并解析本体层定义文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            ontology_data = json.load(f)
        # 检查必要键
        for key in ['entity_types', 'relationships']:
            if key not in ontology_data:
                raise ValueError(f"本体层定义缺少必要的键: {key}")
        entity_types = {entity['type']: entity for entity in ontology_data['entity_types']}
        relationships_by_src = defaultdict(list)
        for rel in ontology_data['relationships']:
            relationships_by_src[rel['src_type']].append(rel)
        valid_relations = {rel['relation'] for rel in ontology_data['relationships']}
        valid_entity_types = {entity['type'] for entity in ontology_data['entity_types']}

        return {
            'original': ontology_data,
            'entity_types': entity_types,
            'relationships_by_src': dict(relationships_by_src),
            'valid_relations': valid_relations,
            'valid_entity_types': valid_entity_types,
            'properties': ontology_data.get('properties', {})
        }
    except Exception as e:
        print(f"加载本体层失败: {e}")
        raise

# ---------------- 定义各代理类 ----------------
class NodeGeneratorAgent:
    """批量生成子节点代理，根据多个关系请求生成候选子节点"""

    def __init__(self, domain: str, ontology: dict):
        self.domain = domain
        self.ontology = ontology

    def generate_children_batch(self, parent_node: str, parent_type: str, relation_requests: list, full_path: str):
        """调用大模型，根据给定关系请求生成子节点列表"""
        valid_requests = []
        # 筛选只保留在本体层中定义的关系
        for req in relation_requests:
            child_type = req["child_type"]
            relationship = req["relationship"]
            valid = False
            for rel in self.ontology['original']['relationships']:
                if (rel['src_type'] == parent_type and
                    rel['tgt_type'] == child_type and
                    rel['relation'] == relationship):
                    req["relation_description"] = rel.get("description", "")
                    req["type_description"] = self.ontology['entity_types'].get(child_type, {}).get("description", "")
                    valid = True
                    break
            if valid:
                valid_requests.append(req)
            else:
                print(f"⚠️ 请求无效: {parent_type} --[{relationship}]--> {child_type} 未在本体中定义")
        if not valid_requests:
            return {}

        # 构造 Prompt
        prompt_lines = [
            f"你是一位专家级{self.domain}知识图谱构建助手，请根据以下请求批量生成子节点。"
        ]
        for idx, req in enumerate(valid_requests, start=1):
            prompt_lines.append(
                f"请求{idx}: 父节点: \"{parent_node}\" (类型: {parent_type}); "
                f"子节点类型: {req['child_type']} - {req['type_description']}; "
                f"关系: \"{req['relationship']}\" - {req['relation_description']}; "
                f"节点完整路径: \"{full_path}\". "
            )
        prompt_lines.append(
            "要求生成相互独立、不互相包含、清晰的候选子节点，不要进行任何的编造。"
            "请按照请求编号输出JSON对象，格式为：{ \"1\": [\"子节点1\", \"子节点2\", ...], \"2\": [...], ... }。不要包含其他文本。"
        )
        prompt = "\n".join(prompt_lines)
        # 调用大模型获取候选子节点
        result = ask_chat_model(prompt, max_retries=10, temperature=0.1, sleep_seconds=2)
        if result is None:
            return {}
        # 整理输出格式
        output = {}
        for idx, req in enumerate(valid_requests, start=1):
            key = str(idx)
            output[(req["child_type"], req["relationship"])] = result.get(key, [])
        return output


class NodeValidationAgent:
    """批量验证子节点代理"""

    def __init__(self, domain: str, ontology: dict):
        self.domain = domain
        self.ontology = ontology

    def validate_nodes_batch(self, validation_requests: list):
        """调用大模型，验证给定父子节点关系是否合理"""
        if not validation_requests:
            return {}

        prompt_lines = [f"您是{self.domain}领域的知识图谱验证专家，请批量评估以下节点关系是否合理。"]
        for idx, req in enumerate(validation_requests, start=1):
            parent_node = req["parent_node"]
            child_node = req["child_node"]
            parent_type = req["parent_type"]
            child_type = req["child_type"]
            relationship = req["relationship"]
            type_description = self.ontology['entity_types'].get(child_type, {}).get("description", "")
            relation_description = ""
            for rel in self.ontology['original']['relationships']:
                if (rel['src_type'] == parent_type and
                    rel['tgt_type'] == child_type and
                    rel['relation'] == relationship):
                    relation_description = rel.get("description", "")
                    break
            prompt_lines.append(
                f"请求{idx}: 父节点: \"{parent_node}\" (类型: {parent_type});"
                f"子节点: \"{child_node}\" (类型: {child_type} - {type_description}); "
                f"关系: \"{relationship}\" - {relation_description}. "
                f"请回答：该关系是否合理？输出格式为：{{\"{idx}\": {{\"valid\": true}}}} 或 "
                f"{{\"{idx}\": {{\"valid\": false, \"reason\": \"详细原因\"}}}}。"
            )
        prompt_lines.append("请输出一个完整的JSON对象，不包含其他文本。")
        prompt = "\n".join(prompt_lines)
        result = ask_chat_model(prompt, max_retries=10, temperature=0.1, sleep_seconds=2)
        if result is None:
            # 重试失败时默认全部通过
            return {str(i + 1): {"valid": True} for i in range(len(validation_requests))}
        return result


class PropertyGeneratorAgent:
    """批量生成节点属性代理"""

    def __init__(self, domain: str, ontology: dict):
        self.domain = domain
        self.ontology = ontology

    def generate_properties_batch(self, property_requests: list):
        """调用大模型，生成节点的属性"""
        if not property_requests:
            return {}

        groups = defaultdict(list)
        for req in property_requests:
            groups[req["node_type"]].append(req)

        output = {}
        for node_type, requests in groups.items():
            property_template = self.ontology['properties'].get(node_type, ["description"])
            type_description = self.ontology['entity_types'].get(node_type, {}).get("description", "")

            prompt_lines = [
                f"您是{self.domain}领域的知识图谱专家，请为下列节点批量生成属性信息。",
                f"要求为每个节点提供以下属性：{', '.join(property_template)}，内容必须专业、准确，不要进行任何的编造。"
            ]
            for idx, req in enumerate(requests, start=1):
                prompt_lines.append(
                    f"请求{idx}: 节点名称: \"{req['node']}\" (类型: {node_type} - {type_description}); "
                    f"所在路径: \"{req['full_path']}\"."
                )
            prompt_lines.append(
                "请以JSON格式输出，格式为：{ \"1\": {\"属性1\": \"值1\", ...}, \"2\": {...}, ... }。不要包含其他文本。"
            )
            prompt = "\n".join(prompt_lines)
            result = ask_chat_model(prompt, max_retries=10, temperature=0.1, sleep_seconds=2)
            if result is None:
                # 重试失败则给出占位信息
                for req in requests:
                    output[req['node']] = {
                        prop: ""
                        for prop in property_template
                    }
                continue

            for idx, req in enumerate(requests, start=1):
                props = result.get(str(idx), {})
                # 确保每个属性都有值
                for prop in property_template:
                    if prop not in props:
                        props[prop] = ""
                output[req['node']] = props

        return output


class PropertyValidationAgent:
    """批量验证节点属性代理"""

    def __init__(self, domain: str, ontology: dict):
        self.domain = domain
        self.ontology = ontology

    def validate_properties_batch(self, validation_requests: list):
        """调用大模型，验证给定节点属性是否合理"""
        if not validation_requests:
            return {}

        prompt_lines = [f"您是{self.domain}领域的知识图谱属性验证专家，请批量评估下列节点属性是否合理。"]
        for idx, req in enumerate(validation_requests, start=1):
            node = req["node"]
            node_type = req["node_type"]
            type_description = self.ontology['entity_types'].get(node_type, {}).get("description", "")
            properties = req["properties"]
            prop_str = "\n".join([f"{k}: {v}" for k, v in properties.items()])
            template = self.ontology['properties'].get(node_type, ["description"])
            prompt_lines.append(
                f"请求{idx}: 节点名称: \"{node}\" (类型: {node_type} - {type_description}); 属性内容:\n{prop_str}\n"
                f"要求判断是否包含必要属性({', '.join(template)})，并且内容是否准确。"
                f"请输出格式：{{\"{idx}\": {{\"valid\": true}}}} 或 "
                f"{{\"{idx}\": {{\"valid\": false, \"reason\": \"详细原因\"}}}}。"
            )
        prompt_lines.append("请输出一个完整的JSON对象，不包含其他文本。")
        prompt = "\n".join(prompt_lines)
        result = ask_chat_model(prompt, max_retries=10, temperature=0.1, sleep_seconds=2)
        if result is None:
            # 重试失败时默认全部通过
            output = {}
            for req in validation_requests:
                output[req["node"]] = {"valid": True}
            return output

        output = {}
        for idx, req in enumerate(validation_requests, start=1):
            res = result.get(str(idx), {})
            output[req["node"]] = res
        return output


# ---------------- 知识图谱构建器 ----------------
class KnowledgeGraphBuilder:
    """知识图谱构建器：协调各代理批量构建完整的知识图谱"""

    def __init__(self, domain: str, ontology_file: str = "ontology.json"):
        self.domain = domain
        self.ontology = load_ontology(ontology_file)
        self.node_generator = NodeGeneratorAgent(domain, self.ontology)
        self.node_validator = NodeValidationAgent(domain, self.ontology)
        self.property_generator = PropertyGeneratorAgent(domain, self.ontology)
        self.property_validator = PropertyValidationAgent(domain, self.ontology)
        self.knowledge_graph = {
            "domain": domain,
            "nodes": {},
            "relationships": []
        }
        self.processed_nodes = set()
        self.batch_size = 5
        self.node_color_map = {}  # 存储节点类型与颜色的映射

    def process_node_properties(self, node_id: str, node_label: str, node_type: str, full_path: str):
        """处理单个节点的属性生成和验证"""
        print(f"🔍 生成并验证节点属性: {node_label} (类型: {node_type})")
        prop_req = [{"node": node_label, "node_type": node_type, "full_path": full_path}]
        gen_props = self.property_generator.generate_properties_batch(prop_req)
        props = gen_props.get(node_label, {})
        val_req = [{"node": node_label, "node_type": node_type, "properties": props}]
        val_props = self.property_validator.validate_properties_batch(val_req)
        if not val_props.get(node_label, {}).get("valid", True):
            print(f"⚠️ 属性验证问题: {val_props.get(node_label, {}).get('reason', '')}")
        self.knowledge_graph['nodes'][node_id]['properties'] = props
        return props

    def process_properties_in_batches(self, nodes_to_process: list):
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
            print(f"\n⏳ 处理属性批次 {batch_idx + 1}/{len(batches)}，包含 {len(batch)} 个节点")
            prop_requests = []
            for node_id, node_label, node_type, full_path in batch:
                prop_requests.append({
                    "node": node_label,
                    "node_type": node_type,
                    "full_path": full_path
                })
            gen_props = self.property_generator.generate_properties_batch(prop_requests)
            val_requests = []
            for node_id, node_label, node_type, _ in batch:
                props = gen_props.get(node_label, {})
                val_requests.append({
                    "node": node_label,
                    "node_type": node_type,
                    "properties": props
                })
            val_props = self.property_validator.validate_properties_batch(val_requests)
            for node_id, node_label, node_type, _ in batch:
                props = gen_props.get(node_label, {})
                if not val_props.get(node_label, {}).get("valid", True):
                    print(f"⚠️ 节点 '{node_label}' 的属性验证问题: {val_props.get(node_label, {}).get('reason', '')}")
                self.knowledge_graph['nodes'][node_id]['properties'] = props

    def build(self):
        print(f'🚀 开始构建 "{self.domain}" 领域知识图谱')
        if self.domain and self.ontology['valid_entity_types']:
            root_id = generate_unique_id(self.domain)
            root_label = self.domain
            root_type = Config.ROOT_TYPE
            root_color = generate_dynamic_color(root_type, self.node_color_map)
            self.knowledge_graph['nodes'][root_id] = {
                "id": root_id,
                "label": root_label,
                "type": root_type,
                "properties": {},
                "color": root_color
            }
            print("\n🔍 处理根节点属性")
            self.process_node_properties(root_id, root_label, root_type, root_label)
            queue = deque([(root_id, root_label, root_type, root_label)])
            self.processed_nodes.add(root_label)
        else:
            print(f"⚠️ 未定义 '{self.domain}' 类型，请检查配置")
            return self.knowledge_graph

        # BFS逐层扩展
        while queue:
            current_id, current_label, current_type, full_path = queue.popleft()
            print(f"\n🔍 处理节点: {current_label} (类型: {current_type})")
            print(f"   路径: {full_path}")

            possible_relations = self.ontology['relationships_by_src'].get(current_type, [])
            if not possible_relations:
                print(f"ℹ️ 节点类型 {current_type} 无子节点关系")
                continue

            relation_requests = [
                {"child_type": rel["tgt_type"], "relationship": rel["relation"]}
                for rel in possible_relations
            ]

            # 生成候选子节点
            gen_result = self.node_generator.generate_children_batch(
                current_label, current_type, relation_requests, full_path
            )

            # 验证子节点候选关系
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
            valid_result = self.node_validator.validate_nodes_batch(validation_requests)
            valid_children = []
            added_nodes = 0
            for req_idx, req in enumerate(validation_requests, 1):
                child = req["child_node"]
                key = str(req_idx)
                valid_info = valid_result.get(key, {"valid": True})
                if valid_info.get("valid", True):
                    if child not in self.processed_nodes:
                        child_id = generate_unique_id(child)
                        child_color = generate_dynamic_color(req["child_type"], self.node_color_map)
                        self.knowledge_graph['nodes'][child_id] = {
                            "id": child_id,
                            "label": child,
                            "type": req["child_type"],
                            "properties": {},
                            "color": child_color
                        }
                        self.knowledge_graph['relationships'].append({
                            "src_id": current_id,
                            "tgt_id": child_id,
                            "type": req["relationship"]
                        })
                        new_path = f"{full_path} -- {req['relationship']} --> {child}"
                        valid_children.append((child_id, child, req["child_type"], new_path))
                        queue.append((child_id, child, req["child_type"], new_path))
                        self.processed_nodes.add(child)
                        added_nodes += 1
                    else:
                        # 已存在节点则仅添加新的关系（避免重复节点）
                        existing_node_id = None
                        for node_id, node_info in self.knowledge_graph['nodes'].items():
                            if node_info["label"] == child:
                                existing_node_id = node_id
                                break
                        if existing_node_id is not None:
                            relationship_exists = any(
                                rel for rel in self.knowledge_graph['relationships']
                                if (rel["src_id"] == current_id and
                                    rel["tgt_id"] == existing_node_id and
                                    rel["type"] == req["relationship"])
                            )
                            if not relationship_exists:
                                self.knowledge_graph['relationships'].append({
                                    "src_id": current_id,
                                    "tgt_id": existing_node_id,
                                    "type": req["relationship"]
                                })
            if added_nodes > 0:
                print(f"添加了 {added_nodes} 个子节点（经过验证）")
                # 批量生成并验证子节点属性
                self.process_properties_in_batches(valid_children)
                time.sleep(1)
        print(f"\n✅ 知识图谱构建完成！总节点数: {len(self.knowledge_graph['nodes'])}，总关系数: {len(self.knowledge_graph['relationships'])}")
        return self.knowledge_graph

    def save(self, output_file: str = "knowledge_graph.json"):
        """
        将知识图谱保存到本地 JSON 文件，并为每个节点添加 color 属性
        """
        kg_copy = json.loads(json.dumps(self.knowledge_graph))
        for node_id, node in kg_copy['nodes'].items():
            node['color'] = self.node_color_map.get(node['type'], '#CCCCCC')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(kg_copy, f, ensure_ascii=False, indent=2)
        print(f"📝 知识图谱已保存到: {output_file}")
        return output_file

    def import_to_neo4j(self, uri: str, user: str, password: str):
        """
        将知识图谱导入 Neo4j 数据库（由 neo4j_import.Neo4jImporter 实现）
        """
        importer = Neo4jImporter(uri, user, password)
        try:
            importer.import_knowledge_graph(self.knowledge_graph)
            print("✅ 成功将知识图谱导入 Neo4j 数据库！")
        finally:
            importer.close()


# ---------------- 主函数 ----------------
if __name__ == '__main__':
    ontology_file = Config.ONTOLOGY_FILE
    if not os.path.exists(ontology_file):
        raise ValueError("请提供本体层设计文件：ontology.json")

    # 定义要构建的领域
    domain = Config.DOMAIN
    builder = KnowledgeGraphBuilder(domain, ontology_file)
    kg = builder.build()

    output_file = Config.OUTPUT_FILE
    builder.save(output_file)

    # 导入 Neo4j 数据库（确保服务已启动且配置正确）
    builder.import_to_neo4j(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
