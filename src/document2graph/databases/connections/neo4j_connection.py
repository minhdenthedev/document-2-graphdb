from neo4j import GraphDatabase


URI = "bolt://localhost:7689"
AUTH = ("neo4j", "minhminh")

with GraphDatabase.driver(URI, auth=AUTH) as driver:
    driver.verify_connectivity()
    summary = driver.execute_query(
        "CREATE (:Person {name: $name})",
        name="Alice",
        database_="neo4j",
    ).summary
    print("Created {nodes_created} nodes in {time} ms.".format(
        nodes_created=summary.counters.nodes_created,
        time=summary.result_available_after
    ))