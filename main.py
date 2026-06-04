import ollama 
from quantara.database.database_v2 import Database

EMBEDDING_MODEL = "snowflake-arctic-embed:335m"

sentences = [
    "I am going to the United States of America to pursue advanced research in artificial intelligence and machine learning technologies.",
    "I will become an AI researcher at Anthropic or OpenAI and contribute to the development of safe, reliable, and beneficial AI systems.",
    "I want to build intelligent AI agents capable of reasoning, planning, learning from experience, and solving complex real-world problems.",
    "Rivers are the soul of any ancient civilization because they provide water, fertile land, transportation routes, and economic opportunities.",
    "The rapid growth of artificial intelligence is transforming industries by automating repetitive tasks and enabling data-driven decision making.",
    "Building scalable software systems requires a strong understanding of algorithms, data structures, distributed computing, and software architecture principles.",
    "Space exploration continues to inspire humanity by expanding our understanding of the universe and driving technological innovation on Earth.",
    "Modern databases are designed to handle massive volumes of data while ensuring consistency, reliability, performance, and fault tolerance.",
    "Machine learning models can identify hidden patterns in data that would be difficult or impossible for humans to discover manually.",
    "The development of renewable energy technologies is essential for reducing carbon emissions and addressing the challenges of climate change.",
    "Reading books from diverse fields of knowledge helps individuals develop critical thinking skills and gain broader perspectives on life.",
    "Effective communication is one of the most valuable professional skills because it enables collaboration, leadership, and problem solving.",
    "The history of scientific discovery demonstrates how curiosity, experimentation, and persistence can lead to groundbreaking innovations.",
    "Advancements in medical technology have significantly improved healthcare outcomes by enabling earlier diagnosis and more effective treatments.",
    "Cybersecurity has become increasingly important as organizations rely on digital infrastructure to store sensitive information and conduct operations.",
    "The emergence of cloud computing has revolutionized software deployment by providing scalable, cost-effective, and highly available infrastructure.",
    "Learning mathematics strengthens logical reasoning abilities and provides the foundation for many disciplines including engineering and computer science.",
    "Autonomous vehicles combine computer vision, sensor fusion, machine learning, and control systems to navigate complex environments safely.",
    "Open-source software communities accelerate innovation by allowing developers around the world to collaborate and share knowledge freely.",
    "The future of human-computer interaction will likely involve more natural interfaces powered by artificial intelligence and multimodal technologies."
]

def main():
    print("Hello from quantara!")

    db = Database("my_database", auto_persist=True)

    for sent in sentences:
        embedded = ollama.embeddings(model=EMBEDDING_MODEL, prompt=sent)
        db.insert_doc(
            name=sent,
            vector=embedded["embedding"],
            metadata={
                "doc_length": len(sent),
                "num_words": len(sent.split()),
                "vector_size": len(embedded["embedding"])
            }
        )

    query = "I want to build AI agents"
    embedded = ollama.embeddings(model=EMBEDDING_MODEL, prompt=query)
    results = db.search_doc(embedded["embedding"], top_k=3, return_text_outputs=True)
    print(results)
        



if __name__ == "__main__":
    main()
