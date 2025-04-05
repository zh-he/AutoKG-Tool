import json
import os
import uuid # 导入 uuid 模块

def format_kg_for_lightrag(input_filepath="knowledge_graph.json", output_filepath="lightrag_formatted_kg.json"):
    """
    根据最终确认的需求，将自定义知识图谱JSON转换为LightRAG接受的格式。
    确保 'entities' 中的 'description' 字段是一个 JSON 字符串。

    格式要求 :
    顶级键: chunks, entities, relationships (都是列表)

    - 全局生成一个 source_id: global_source_id = f"chunk-{uuid.uuid4().hex}"

    1. chunks (List[Dict]):
        - 只包含一个元素: [{"content": "", "source_id": global_source_id}]
    2. entities (List[Dict]):
        - "entity_name": 原始节点的 'label'
        - "entity_type": 原始节点的 'type'
        - "description": **原始节点 'properties' 字典序列化后的 JSON 字符串**
        - "source_id": global_source_id
    3. relationships (List[Dict]):
        - "src_id": 原始源节点的 'label'
        - "tgt_id": 原始目标节点的 'label'
        - "description": f"{源节点label}和{目标节点label}的关系是{原始关系type}"
        - "keywords": "" (空字符串)
        - "weight": 1 (数字)
        - "source_id": global_source_id

    Args:
        input_filepath (str): 输入的原始知识图谱JSON文件路径。
        output_filepath (str): 输出的符合LightRAG格式的JSON文件路径。

    Returns:
        bool: 如果成功转换并保存则返回 True，否则返回 False。
    """
    print(f"开始根据最终确认格式转换知识图谱文件: {input_filepath}")

    try:
        # 1. 读取原始知识图谱JSON文件
        if not os.path.exists(input_filepath):
            print(f"错误：输入文件不存在 {input_filepath}")
            return False

        with open(input_filepath, 'r', encoding='utf-8') as f:
            original_kg = json.load(f)
        print("原始知识图谱加载成功。")

        # 检查基本结构
        if 'nodes' not in original_kg or 'relationships' not in original_kg:
            print("错误：输入的JSON文件缺少 'nodes' 或 'relationships' 键。")
            return False

        # --- 生成全局唯一的 source_id ---
        global_source_id = f"chunk-{uuid.uuid4().hex}"
        print(f"生成的全局 source_id (用于单一 Chunk): {global_source_id}")
        # ---

        # --- 预处理：创建 node_id 到 label 的映射 ---
        print("正在创建 node ID 到 label 的映射...")
        node_id_to_label = {}
        original_nodes_data = original_kg.get('nodes', {})
        for node_id, node_data in original_nodes_data.items():
            if isinstance(node_data, dict) and 'label' in node_data:
                actual_id = node_data.get('id', node_id)
                node_id_to_label[actual_id] = node_data['label']
            else:
                print(f"警告：节点 {node_id} 数据格式不正确或缺少 'label'，无法添加到映射。")
        print(f"ID 到 label 映射创建完成，共 {len(node_id_to_label)} 个条目。")
        # --- 预处理结束 ---

        # 2. 初始化LightRAG格式的数据结构
        lightrag_chunks = [{"content": "", "source_id": global_source_id}]
        lightrag_entities = []
        lightrag_relationships = []

        # 3. 处理节点 (生成 entities 列表) - 确保 description 是 JSON 字符串
        print(f"开始处理 {len(original_nodes_data)} 个节点 (生成 entities 列表)...")
        processed_node_count = 0
        for node_id, node_data in original_nodes_data.items():
            if not isinstance(node_data, dict):
                print(f"警告：跳过无效的节点数据（非字典）: ID {node_id}")
                continue

            # 获取原始节点信息
            original_label = node_data.get('label', f"UnknownLabel_{node_id}")
            original_type = node_data.get('type', 'Unknown')
            original_properties = node_data.get('properties', {}) # 获取属性字典

            # *** 关键步骤：将原始属性字典序列化为 JSON 字符串 ***
            properties_str = json.dumps(original_properties, ensure_ascii=False, separators=(',', ':'))

            # 构建entities 对象 - 确保 description 赋值正确
            new_entity = {
                "entity_name": original_label,
                "entity_type": original_type,
                "description": properties_str,
                "source_id": global_source_id
            }
            lightrag_entities.append(new_entity)
            processed_node_count += 1
        print(f"成功处理 {processed_node_count} 个节点 (生成 entities 列表)。")


        # 4. 处理关系 (Relationships)
        print(f"开始处理 {len(original_kg.get('relationships', []))} 条关系...")
        processed_relation_count = 0
        missing_label_count = 0
        for rel_data in original_kg.get('relationships', []):
            if not isinstance(rel_data, dict):
                print(f"警告：跳过无效的关系数据（非字典）: {rel_data}")
                continue

            original_src_id = rel_data.get('src_id')
            original_tgt_id = rel_data.get('tgt_id')
            original_rel_type = rel_data.get('type')

            if not original_src_id or not original_tgt_id or not original_rel_type:
                print(f"警告：跳过缺少 src_id, tgt_id 或 type 的关系: {rel_data}")
                continue

            src_label = node_id_to_label.get(original_src_id, None)
            tgt_label = node_id_to_label.get(original_tgt_id, None)

            if src_label is None or tgt_label is None:
                missing_label_count += 1
                print(f"警告：关系 {rel_data} 的源或目标节点 label 未找到，跳过。")
                if src_label is None: print(f"  - 源 ID '{original_src_id}' label 未找到。")
                if tgt_label is None: print(f"  - 目标 ID '{original_tgt_id}' label 未找到。")
                continue

            rel_description = f"{src_label}和{tgt_label}的关系是{original_rel_type}"

            new_relationship = {
                "src_id": src_label,
                "tgt_id": tgt_label,
                "description": rel_description,
                "keywords": "",
                "weight": 1,
                "source_id": global_source_id
            }
            lightrag_relationships.append(new_relationship)
            processed_relation_count += 1

        print(f"成功处理 {processed_relation_count} 条关系 。")
        if missing_label_count > 0:
            print(f"注意：有 {missing_label_count} 条关系因无法找到源/目标节点的 label 而被跳过。")

        # 5. 组合最终的LightRAG输入JSON
        lightrag_kg_input = {
            "chunks": lightrag_chunks,
            "entities": lightrag_entities,
            "relationships": lightrag_relationships
        }

        # 6. 保存转换后的JSON文件
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(lightrag_kg_input, f, ensure_ascii=False, indent=2)

        print(f"转换完成！符合最终确认格式的知识图谱已保存到: {output_filepath}")
        return True

    except FileNotFoundError:
        print(f"错误：输入文件未找到 {input_filepath}")
        return False
    except json.JSONDecodeError:
        print(f"错误：无法解析输入的JSON文件 {input_filepath}，请检查文件格式是否正确。")
        return False
    except Exception as e:
        print(f"转换过程中发生未知错误: {e}")
        # import traceback; traceback.print_exc()
        return False

# --- 主程序入口 ---
if __name__ == "__main__":
    input_kg_file = "knowledge_graph.json"
    output_lightrag_file = "lightrag_formatted_kg.json" # 格式输出

    success = format_kg_for_lightrag(input_kg_file, output_lightrag_file)

    if success:
        print("\n转换成功JSON字符串格式。")
        # 预览检查
        try:
             with open(output_lightrag_file, 'r', encoding='utf-8') as f:
                 preview_data = json.load(f)
                 print("\n转换后数据预览:")
                 print("--- Chunks ---")
                 print(json.dumps(preview_data.get('chunks', []), indent=2, ensure_ascii=False))
                 print("\n--- Entities ---")
                 # 打印第一个 entity 的 description 类型和内容，以供检查
                 first_entity = preview_data.get('entities', [])[0] if preview_data.get('entities') else None
                 if first_entity:
                     print(f"(第一个 Entity 的 description 类型: {type(first_entity.get('description'))})")
                 print(json.dumps(first_entity, indent=2, ensure_ascii=False))
                 print("\n--- Relationships ---")
                 print(json.dumps(preview_data.get('relationships', [])[:1], indent=2, ensure_ascii=False))
        except Exception as e:
            print(f"\n预览转换后数据时出错: {e}")
    else:
        print("\n知识图谱转换失败。请检查错误信息。")