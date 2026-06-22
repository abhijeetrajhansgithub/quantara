from quantara import Database
from quantara import OllamaEmbedding

# 5 long sentences about rivers
rivers = [
    "The Amazon River is one of the longest and most voluminous rivers in the world, flowing through several countries in South America and supporting an incredibly diverse rainforest ecosystem.",
    "The Nile River has played a crucial role in the development of ancient civilizations by providing fertile land, transportation routes, and a reliable source of water across northeastern Africa.",
    "The Yangtze River is the longest river in Asia and serves as an important transportation corridor, economic resource, and source of hydroelectric power for millions of people in China.",
    "The Mississippi River flows through the central United States, connecting numerous states while supporting agriculture, trade, transportation, and wildlife throughout its vast basin.",
    "The Danube River passes through multiple European countries, linking diverse cultures and economies while serving as a major waterway for commerce, tourism, and regional cooperation.",
]

# 5 long sentences about mountains
mountains = [
    "Mount Everest is the highest mountain in the world, attracting climbers from every continent who seek to reach its summit despite the extreme weather and challenging conditions.",
    "Mount Kilimanjaro is the highest mountain in Africa and is famous for its distinct ecological zones, which range from tropical forests at its base to glaciers near its peak.",
    "Mount Fuji is the highest mountain in Japan and is widely regarded as a cultural and spiritual symbol that has inspired artists, poets, and travelers for centuries.",
    "K2 is the second-highest mountain on Earth and is considered one of the most difficult and dangerous peaks to climb because of its steep slopes and unpredictable weather.",
    "The Alps stretch across several European countries and provide breathtaking landscapes, popular skiing destinations, and important habitats for a wide variety of plants and animals.",
]

# 5 long sentences about cities
cities = [
    "New York City is one of the most influential metropolitan areas in the world, known for its financial institutions, cultural diversity, iconic landmarks, and vibrant economy.",
    "Tokyo is the capital of Japan and is renowned for its advanced technology, efficient public transportation systems, bustling districts, and unique blend of tradition and modernity.",
    "Paris is the capital of France and attracts millions of visitors each year because of its historic architecture, world-famous museums, exquisite cuisine, and artistic heritage.",
    "London is a global center for finance, education, and culture, combining centuries of history with modern innovation and serving as a hub for international business.",
    "Singapore is a highly developed city-state known for its strategic location, clean urban environment, technological advancements, and strong emphasis on economic growth and sustainability.",
]

db = Database(db_name="test")

e = OllamaEmbedding()

collections = {
    "rivers": rivers,
    "mountains": mountains,
    "cities": cities,
}

for collection_name, sentences in collections.items():
    db.create_collection(collection=collection_name)

    for idx, sentence in enumerate(sentences):
        embedding = e.embed(text=sentence)

        db.insert_doc(
            collection=collection_name,
            name=f"{collection_name}_{idx}",
            vector=embedding,
            metadata={"text": sentence},
        )

db.export_to_json(json_path="test.json")


stats = db.stats()

print("\n\nStats:\n", stats)
# print collection names
print(db.list_collections())

# rename collections & print stats
db.rename_collection(old_name="rivers", new_name="rivers2")

db.rename_collection(old_name="mountains", new_name="mountains2")

db.rename_collection(old_name="cities", new_name="cities2")

stats = db.stats()

# clone collections & print stats
db.clone_collection(source="rivers2", target="rivers3")

db.clone_collection(source="mountains2", target="mountains3")

db.clone_collection(source="cities2", target="cities3")

stats = db.stats()

print("\n\nStats:\n", stats)

# print collection names
print(db.list_collections())

# delete collections & print stats
db.delete_collection("rivers2")
db.delete_collection("mountains2")
db.delete_collection("cities2")

stats = db.stats()

print("\n\nStats:\n", stats)
print(db.get_config())

# do vector search for river

query = "which is the longest river?"
e = OllamaEmbedding()

vector = e.embed(text=query)

results = db.search_doc(
    collection="rivers3", input_vector=vector, top_k=3, return_text_outputs=True
)

print("\n\nResults:\n", results)
