import os
from llama_index.core import StorageContext, load_index_from_storage 
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.core import Settings

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
# Settings.embed_model = HuggingFaceEmbedding(
#     model_name="avsolatorio/GIST-small-Embedding-v0"
# )

llm = OpenAI(model="gpt-4o-mini", max_tokens=4096, temperature=0.5, top_p=0.9)
reader = SimpleDirectoryReader(input_dir=r"C:\Users\idarv\Work\temp\republic\republic\republic_store")
docs = reader.load_data()
print(f"Loaded {len(docs)} docs")

try:
    storage_context = StorageContext.from_defaults(persist_dir="./republic/republic_store");
    index = load_index_from_storage(storage_context)

    index_loaded = True
except:
    index_loaded = False

if not index_loaded:
    index = VectorStoreIndex.from_documents(docs)
    index.storage_context.persist(persist_dir="./republic/republic_store")


query_engine = index.as_query_engine(
    llm=llm,
    similarity_top_k=3,
    dense_similarity_top_k=3,
    sparse_similarity_top_k=3,
    alpha=0.5,
    enable_reranking=True,
)

response = query_engine.query("Get all monthwise invoice amount for cimmaron")

print(response)

