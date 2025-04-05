import json
from neo4j import GraphDatabase

from config import Config
# ====== 配置 Neo4j 连接信息 ======
NEO4J_URI = Config.NEO4J_URI
NEO4J_USERNAME = Config.NEO4J_USERNAME
NEO4J_PASSWORD = Config.NEO4J_PASSWORD
KNOWLEDGE_GRAPH_JSON = Config.OUTPUT_FILE


class Neo4jImporter:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def import_knowledge_graph(self, kg_data):
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
        properties_str = json.dumps(node_info.get("properties", {}), ensure_ascii=False)
        node_type = node_info.get("type", "Unknown")
        label = node_type.replace(" ", "_").replace("-", "_").replace(".", "_")
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
        query = """
        MATCH (a {id: $from_id}), (b {id: $to_id})
        MERGE (a)-[r:%s]->(b)
        """ % rel["type"]
        tx.run(
            query,
            from_id=rel["src_id"],  # 修改这里：使用 src_id
            to_id=rel["tgt_id"],  # 修改这里：使用 tgt_id
        )


def main():
    with open(KNOWLEDGE_GRAPH_JSON, "r", encoding="utf-8") as f:
        kg_data = json.load(f)

    importer = Neo4jImporter(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
    try:
        importer.import_knowledge_graph(kg_data)
        print("✅ 成功将知识图谱写入 Neo4j 数据库！")
    finally:
        importer.close()


if __name__ == "__main__":
    main()