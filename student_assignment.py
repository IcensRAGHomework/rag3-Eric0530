import datetime
import chromadb
import traceback
import pandas as pd
import os , time

from chromadb.utils import embedding_functions

from model_configurations import get_model_configuration

gpt_emb_version = 'text-embedding-ada-002'
gpt_emb_config = get_model_configuration(gpt_emb_version)

dbpath = "./"

def generate_hw01():
    
     
    # 初始化 ChromaDB 並使用 SQLite 作為存儲
    #db_path = "./"
    chroma_client = chromadb.PersistentClient(path=dbpath)

    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key = gpt_emb_config['api_key'],
    api_base = gpt_emb_config['api_base'],
    api_type = gpt_emb_config['openai_type'],
    api_version = gpt_emb_config['api_version'],
    deployment_id = gpt_emb_config['deployment_name']
    )

    # 創建一個 collection
    collection = chroma_client.get_or_create_collection(
        name="TRAVEL",
        metadata={"hnsw:space": "cosine"},
        embedding_function=openai_ef
        )
    
    if collection.count() != 0:
        return collection
    
        # 讀取 CSV 檔案
    file_path = os.path.join(dbpath, "COA_OpenData.csv")
    df = pd.read_csv(file_path)

    # 轉換 CreateDate 欄位為 datetime
    df["CreateDate"] = pd.to_datetime(df["CreateDate"], errors="coerce")

    # 6️⃣ 轉換日期格式並準備資料
    records = []
    for index, row in df.iterrows():
            host_words = str(row.get("HostWords", "")).strip()  # 提取 `HostWords` 欄位作為文本數據
            if not host_words:  # 如果 `HostWords` 為空，則跳過
                continue

            # 將 CreateDate 轉換為 Unix 時間戳格式（秒）
            timestamp = int(time.mktime(row["CreateDate"].timetuple())) if pd.notnull(row["CreateDate"]) else None

            metadata = {
            "file_name": "COA_OpenData.csv",
            "name": row.get("Name", ""),
            "type": row.get("Type", ""),
            "address": row.get("Address", ""),
            "tel": row.get("Tel", ""),
            "city": row.get("City", ""),
            "town": row.get("Town", ""),
            "date": timestamp  # 存入 Unix 時間戳格式
            }

            records.append((str(index), host_words, metadata))  # 使用 index 作為 id

    if records:
        collection.add(
        ids=[r[0] for r in records],            # 每筆數據的 ID
        documents=[r[1] for r in records],       # `HostWords` 作為文本數據
        metadatas=[r[2] for r in records]        # Metadata 附加資訊
        )   

    print(collection.count())
    return collection
    

def generate_hw02(question, city, store_type, start_date, end_date):
    # 連接到已經建立的 ChromaDB SQLite
    chroma_client = chromadb.PersistentClient(path=dbpath)

    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key = gpt_emb_config['api_key'],
    api_base = gpt_emb_config['api_base'],
    api_type = gpt_emb_config['openai_type'],
    api_version = gpt_emb_config['api_version'],
    deployment_id = gpt_emb_config['deployment_name']
    )

    # 取得 Collection
    collection = chroma_client.get_or_create_collection(
        name="TRAVEL",
        metadata={"hnsw:space": "cosine"},
        embedding_function=openai_ef

        )
    question_embeding = openai_ef([question])

    result = collection.query(
        query_embeddings=question_embeding,
        n_results=10,
        where={"$and": [{"city":{"$in":city}},{"type":{"$in":store_type}}]}
        )
    print(result)
    return result


def generate_hw03(question, store_name, new_store_name, city, store_type):
    pass
    
def demo(question):
    chroma_client = chromadb.PersistentClient(path=dbpath)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key = gpt_emb_config['api_key'],
        api_base = gpt_emb_config['api_base'],
        api_type = gpt_emb_config['openai_type'],
        api_version = gpt_emb_config['api_version'],
        deployment_id = gpt_emb_config['deployment_name']
    )
    collection = chroma_client.get_or_create_collection(
        name="TRAVEL",
        metadata={"hnsw:space": "cosine"},
        embedding_function=openai_ef
    )
    
    return collection

generate_hw01()
city=["宜蘭縣", "新北市"]
type=["美食"]
start_date = datetime.datetime(2024, 1, 1)
End_date = datetime.datetime(2024, 12, 31)
#generate_hw02("我想要找有關茶餐點的店家",city,type,start_date,End_date)