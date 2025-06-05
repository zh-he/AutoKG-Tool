import asyncio
from datetime import datetime
import json
import os

from flask import Flask, request, jsonify, send_from_directory

from KG_generate import KnowledgeGraphBuilder
from config import Config, validate_config
from neo4j_import import Neo4jImporter

app = Flask(__name__)

UPLOAD_FOLDER = 'ontology'
KG_FOLDER = 'KG'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(KG_FOLDER, exist_ok=True)


def validate_ontology(ontology_data):
    """验证本体层设计的合理性"""
    # 检查必需字段
    required_fields = ["entity_types", "relationships", "properties"]
    for field in required_fields:
        if field not in ontology_data:
            return False, f"缺少必需字段: {field}"

    # 检查entity_types
    if not isinstance(ontology_data["entity_types"], list):
        return False, "entity_types必须是列表类型"

    for entity_type in ontology_data["entity_types"]:
        if not isinstance(entity_type, dict):
            return False, "entity_types中的每个元素必须是字典类型"
        if "type" not in entity_type:
            return False, "entity_types中的每个元素必须包含'type'字段"

    # 检查relationships
    if not isinstance(ontology_data["relationships"], list):
        return False, "relationships必须是列表类型"

    for rel in ontology_data["relationships"]:
        if not isinstance(rel, dict):
            return False, "relationships中的每个元素必须是字典类型"
        required_rel_fields = ["src_type", "tgt_type", "relation"]
        for field in required_rel_fields:
            if field not in rel:
                return False, f"relationships中的每个元素必须包含'{field}'字段"

    # 检查properties
    if not isinstance(ontology_data["properties"], dict):
        return False, "properties必须是字典类型"

    # 检查properties中的每个实体类型是否在entity_types中定义
    entity_types = {et["type"] for et in ontology_data["entity_types"]}
    for entity_type in ontology_data["properties"]:
        if entity_type not in entity_types:
            return False, f"properties中定义了未在entity_types中声明的类型: {entity_type}"

    return True, "本体层设计验证通过"


@app.route('/')
def index():
    return send_from_directory('.', 'knowledge_graph.html')


@app.route('/KG/<path:filename>')
def serve_kg(filename):
    return send_from_directory(KG_FOLDER, filename)


@app.route('/upload-ontology-and-generate-kg', methods=['POST'])
def upload_ontology_and_generate_kg():
    """合并上传本体和生成知识图谱的功能"""
    if 'ontology_file' not in request.files:
        return jsonify({"error": "未找到文件"}), 400

    file = request.files['ontology_file']
    domain_name = request.form.get('domain_name', '').strip()
    root_type = request.form.get('root_type', '').strip()

    if not domain_name:
        return jsonify({"error": "请输入领域名称"}), 400

    if not root_type:
        return jsonify({"error": "请输入根节点类型"}), 400

    if file.filename == '':
        return jsonify({"error": "未选择文件"}), 400

    if not file.filename.endswith('.json'):
        return jsonify({"error": "只支持JSON格式的文件"}), 400

    try:
        # 读取并解析JSON文件
        ontology_data = json.loads(file.read())

        # 验证本体层设计
        is_valid, message = validate_ontology(ontology_data)
        if not is_valid:
            return jsonify({"error": f"本体层设计验证失败: {message}"}), 400

        # 验证root_type是否存在于entity_types中
        entity_types = {et["type"] for et in ontology_data["entity_types"]}
        if root_type not in entity_types:
            return jsonify({
                               "error": f"指定的根节点类型 '{root_type}' 不存在于本体定义中。可用的类型有: {', '.join(entity_types)}"}), 400

        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存文件
        ontology_filename = f"{root_type}_{timestamp}_ontology.json"
        ontology_path = os.path.join(UPLOAD_FOLDER, ontology_filename)

        with open(ontology_path, 'w', encoding='utf-8') as f:
            json.dump(ontology_data, f, ensure_ascii=False, indent=2)
        print(f"本体文件已保存: {ontology_path}")

        # 生成知识图谱文件名
        kg_filename = f"{root_type}_{timestamp}_kg.json"
        kg_path = os.path.join(KG_FOLDER, kg_filename)

        try:
            # 创建知识图谱构建器实例
            builder = KnowledgeGraphBuilder(domain=domain_name, root_type=root_type)

            try:
                # 直接使用保存的本体文件，不再需要临时文件
                kg_data = asyncio.run(builder.build_with_ontology(ontology_path))
                print(
                    f"知识图谱构建完成，节点数: {len(kg_data.get('nodes', {}))}, 关系数: {len(kg_data.get('relationships', []))}")

                # 保存生成的知识图谱
                with open(kg_path, 'w', encoding='utf-8') as f:
                    json.dump(kg_data, f, ensure_ascii=False, indent=2)

                return jsonify({
                    "success": True,
                    "kg_path": f"KG/{kg_filename}",  # 返回相对路径
                    "ontology_path": f"ontology/{ontology_filename}",  # 返回本体文件路径
                    "message": f"知识图谱 {kg_filename} 生成成功，本体文件已保存为 {ontology_filename}"
                })
            except Exception as e:
                # 如果构建失败，删除已保存的本体文件
                if os.path.exists(ontology_path):
                    os.unlink(ontology_path)
                    print(f"构建失败，已删除本体文件: {ontology_path}")

                print(f"处理文件时发生错误: {e}")
                import traceback
                print(f"错误详细信息: {traceback.format_exc()}")
                return jsonify({"error": f"处理文件时发生错误: {str(e)}"}), 500

        except Exception as e:
            # 如果构建器创建失败，删除已保存的本体文件
            if os.path.exists(ontology_path):
                os.unlink(ontology_path)
                print(f"构建器创建失败，已删除本体文件: {ontology_path}")
            raise e

    except json.JSONDecodeError:
        return jsonify({"error": "无效的JSON格式"}), 400
    except Exception as e:
        return jsonify({"error": f"处理文件时发生错误: {str(e)}"}), 500


@app.route('/list-knowledge-graphs', methods=['GET'])
def list_kgs():
    files = [f for f in os.listdir(KG_FOLDER) if f.endswith('.json')]
    return jsonify(files)


@app.route('/get-knowledge-graph/<filename>', methods=['GET'])
def get_kg(filename):
    return send_from_directory(KG_FOLDER, filename)


@app.route('/import-to-neo4j', methods=['POST'])
def import_to_neo4j():
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                "success": False,
                "error": "请求数据为空"
            }), 400

        kg_filename = data.get('kg_filename')
        if not kg_filename:
            return jsonify({
                "success": False,
                "error": "未提供知识图谱文件名"
            }), 400

        kg_path = os.path.join(KG_FOLDER, kg_filename)

        if not os.path.exists(kg_path):
            return jsonify({
                "success": False,
                "error": f"找不到知识图谱文件: {kg_filename}"
            }), 404

        with open(kg_path, 'r', encoding='utf-8') as f:
            kg_data = json.load(f)

        # 使用Neo4jImporter类导入数据
        importer = Neo4jImporter(Config.NEO4J_URI, Config.NEO4J_USERNAME, Config.NEO4J_PASSWORD)
        try:
            importer.import_knowledge_graph(kg_data)
            return jsonify({
                "success": True,
                "message": f"✅ 成功将知识图谱 {kg_filename} 导入到Neo4j数据库！"
            })
        finally:
            importer.close()

    except json.JSONDecodeError:
        return jsonify({
            "success": False,
            "error": "知识图谱文件格式错误，请检查JSON格式"
        }), 400
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"导入Neo4j时发生错误: {str(e)}"
        }), 500


if __name__ == '__main__':
    try:
        # 验证配置
        validate_config()
        print("✅ 配置验证成功，启动Flask应用...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        exit(1)
