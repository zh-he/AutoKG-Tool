import json
import sys
from neo4j import GraphDatabase
from config import Config

NEO4J_URI = Config.NEO4J_URI
NEO4J_USERNAME = Config.NEO4J_USERNAME
NEO4J_PASSWORD = Config.NEO4J_PASSWORD


class Neo4jImporter:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def import_knowledge_graph(self, kg_data):
        """导入知识图谱数据到Neo4j数据库"""
        with self.driver.session() as session:
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:KGNode) REQUIRE n.id IS UNIQUE")

            nodes_data = kg_data.get("nodes", {})
            for node_id, node_info in nodes_data.items():
                session.write_transaction(self._create_node, node_id, node_info)

            relationships_data = kg_data.get("relationships", [])
            for rel in relationships_data:
                session.write_transaction(self._create_relationship, rel)

    @staticmethod
    def _create_node(tx, node_id, node_info):
        """创建节点的静态方法"""
        properties_str = json.dumps(node_info.get("properties", {}), ensure_ascii=False)
        node_type = node_info.get("type", "Unknown")

        label = node_type.replace(" ", "_").replace("-", "_").replace(".", "_")
        if label and not label[0].isalpha():
            label = "Type_" + label

        query = f"""
        MERGE (n:KGNode:{label} {{id: $node_id}})
        ON CREATE SET 
            n.label = $label_str, 
            n.type = $type, 
            n.properties = $properties, 
            n.color = $color
        ON MATCH SET 
            n.label = $label_str, 
            n.type = $type, 
            n.properties = $properties, 
            n.color = $color
        """

        tx.run(
            query,
            node_id=node_id,
            label_str=node_info.get("label", ""),
            type=node_type,
            properties=properties_str,
            color=node_info.get("color", "#CCCCCC")
        )

    @staticmethod
    def _create_relationship(tx, rel):
        """创建关系的静态方法"""
        rel_type = rel.get("type", "RELATED")

        rel_type = rel_type.upper().replace(" ", "_").replace("-", "_").replace(".", "_")

        if rel_type and not rel_type[0].isalpha():
            rel_type = "REL_" + rel_type

        query = f"""
        MATCH (a:KGNode {{id: $from_id}}), (b:KGNode {{id: $to_id}})
        MERGE (a)-[r:{rel_type}]->(b)
        """

        tx.run(
            query,
            from_id=rel["src_id"],
            to_id=rel["tgt_id"],
        )

    def clear_database(self):
        """清空数据库中的所有数据"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("✅ 数据库已清空")

    def get_stats(self):
        """获取数据库统计信息"""
        with self.driver.session() as session:
            node_result = session.run("MATCH (n:KGNode) RETURN count(n) as count")
            node_count = node_result.single()["count"]

            rel_result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            rel_count = rel_result.single()["count"]

            return {"nodes": node_count, "relationships": rel_count}


def import_kg_from_file(kg_file_path: str) -> bool:
    """
    从指定文件导入知识图谱到Neo4j的便捷函数
    参数:
        kg_file_path (str): 知识图谱JSON文件路径
    返回:
        bool: 导入是否成功
    """
    try:
        with open(kg_file_path, "r", encoding="utf-8") as f:
            kg_data = json.load(f)

        importer = Neo4jImporter(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)

        try:
            print(f"开始导入知识图谱 {kg_file_path} 到Neo4j数据库...")
            importer.import_knowledge_graph(kg_data)

            # 获取并显示统计信息
            stats = importer.get_stats()
            print(f"✅ 成功将知识图谱导入 Neo4j 数据库！")
            print(f"   - 节点数量: {stats['nodes']}")
            print(f"   - 关系数量: {stats['relationships']}")
            return True

        finally:
            importer.close()

    except FileNotFoundError:
        print(f"❌ 错误: 找不到知识图谱文件 {kg_file_path}")
        return False
    except json.JSONDecodeError:
        print(f"❌ 错误: 知识图谱文件格式不正确 {kg_file_path}")
        return False
    except Exception as e:
        print(f"❌ 导入过程中发生错误: {e}")
        return False


def import_kg_from_data(kg_data: dict) -> bool:
    """
    直接从数据导入知识图谱到Neo4j的便捷函数
    参数:
        kg_data (dict): 知识图谱数据字典
    返回:
        bool: 导入是否成功
    """
    try:
        importer = Neo4jImporter(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)

        try:
            print("开始导入知识图谱数据到Neo4j数据库...")
            importer.import_knowledge_graph(kg_data)

            stats = importer.get_stats()
            print(f"✅ 成功将知识图谱导入 Neo4j 数据库！")
            print(f"   - 节点数量: {stats['nodes']}")
            print(f"   - 关系数量: {stats['relationships']}")
            return True

        finally:
            importer.close()

    except Exception as e:
        print(f"❌ 导入过程中发生错误: {e}")
        return False


def main():
    """主函数：支持命令行参数导入知识图谱"""
    if len(sys.argv) > 1:
        kg_file_path = sys.argv[1]
        success = import_kg_from_file(kg_file_path)
        if not success:
            sys.exit(1)
    else:
        print("使用方法: python neo4j_import.py <knowledge_graph_file.json>")
        print("示例: python neo4j_import.py KG/computer_science_kg.json")
        print("\n注意: Web应用会直接调用Neo4jImporter类，无需通过此脚本")


if __name__ == "__main__":
    main()