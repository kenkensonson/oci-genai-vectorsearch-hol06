# 会話履歴保持の仕組みを取り入れたRAGの実装

### 構成
ドキュメント保持用ベクトルデータベース：Oracle DB23ai AI Vector Search  
会話履歴保持用データベース : OCI PosgreSQL Database Service  
大規模言語モデル : OCI Generative AI Service(Command-R-Plus)  

### 参考にしたサンプルコード  

LangChainの会話履歴の仕組みを追加するサンプルコード  
https://python.langchain.com/v0.2/docs/how_to/qa_chat_history_how_to/  

会話履歴をPosgreSQLに記録するコード  
https://hexacluster.ai/postgresql/postgres-for-chat-history-langchain-postgres-postgreschatmessagehistory/ 


# 構成

# 実装

### インストール

```python
!pip install --upgrade pip
!pip install -Uq oracledb pypdf cohere langchain langchain-community langchain-core langchain_postgres oci grandalf psycopg
```

### Oracle DB
データベースの接続、pdfファイルの埋め込みとロード

```python
import oracledb

username = "docuser"
password = "docuser"
dsn = "localhost/freepdb1"

try:
    connection = oracledb.connect(user=username, password=password, dsn=dsn)
    print("Connection successful!")
except Exception as e:
    print("Connection failed!")
```

```python
from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("/tmp/rocket.pdf")
documents = loader.load_and_split()


documents 

[Document(metadata={'source': '/tmp/rocket.pdf', 'page': 0}, page_content='当社が開発したロケットエンジンである OraBooster は、次世代の宇宙探査を支える先進的な推進技術の\n象徴です。その独自の設計は、高性能と革新性を融合させ、人類の宇宙進出を加速させるための革命的\nな一歩となります。\nこのエンジンの核となるのは、量子ダイナミックス・プラズマ・ブースターです。このブースターは、\n量子力学の原理に基づいてプラズマを生成し、超高速で加速させます。その結果、従来の化学反応より\nもはるかに高い推力を発生し、遠く離れた惑星や星系への探査を可能にします。\nさらに、エンジンの外殻にはナノファイバー製の超軽量かつ超強度の素材が使用されています。この素\n材は、宇宙空間の過酷な環境に耐え、高速での飛行中に生じる熱や衝撃からロケットを守ります。\nまた、ハイパーフォトン・ジャイロスコープが搭載されており、極めて高い精度でロケットの姿勢を維\n持し、目標を追跡します。これにより、長時間にわたる宇宙飛行中でも安定した飛行軌道を維持し、\nミッションの成功を確保します。\nさらに、バイオニック・リアクション・レスポンダーが統合されています。このシステムは、人工知能\nと生体認識技術を組み合わせ、ロケットの異常な振動や動きを検知し、自己修復機能を活性化します。\n総じて、この新開発のロケットエンジンは、革新的な技術と未来志向の設計によって、宇宙探査の新た\nな時代を切り開くことでしょう。その高い性能と信頼性は、人類の夢を実現するための力強い支援とな\nることでしょう。')]
```

```python
from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(separator="。", chunk_size=100, chunk_overlap=10)
docs = text_splitter.split_documents(documents)
print(docs)


[Document(metadata={'source': '/tmp/rocket.pdf', 'page': 0}, page_content='当社が開発したロケットエンジンである OraBooster は、次世代の宇宙探査を支える先進的な推進技術の\n象徴です'), Document(metadata={'source': '/tmp/rocket.pdf', 'page': 0}, page_content='その独自の設計は、高性能と革新性を融合させ、人類の宇宙進出を加速させるための革命的\nな一歩となります。\nこのエンジンの核となるのは、量子ダイナミックス・プラズマ・ブースターです'), Document(metadata={'source': '/tmp/rocket.pdf', 'page': 0}, page_content='このブースターは、\n量子力学の原理に基づいてプラズマを生成し、超高速で加速させます。その結果、従来の化学反応より\nもはるかに高い推力を発生し、遠く離れた惑星や星系への探査を可能にします'), Document(metadata={'source': '/tmp/rocket.pdf', 'page': 0}, page_content='さらに、エンジンの外殻にはナノファイバー製の超軽量かつ超強度の素材が使用されています。この素\n材は、宇宙空間の過酷な環境に耐え、高速での飛行中に生じる熱や衝撃からロケットを守ります'), Document(metadata={'source': '/tmp/rocket.pdf', 'page': 0}, page_content='また、ハイパーフォトン・ジャイロスコープが搭載されており、極めて高い精度でロケットの姿勢を維\n持し、目標を追跡します'), Document(metadata={'source': '/tmp/rocket.pdf', 'page': 0}, page_content='これにより、長時間にわたる宇宙飛行中でも安定した飛行軌道を維持し、\nミッションの成功を確保します。\nさらに、バイオニック・リアクション・レスポンダーが統合されています'), Document(metadata={'source': '/tmp/rocket.pdf', 'page': 0}, page_content='このシステムは、人工知能\nと生体認識技術を組み合わせ、ロケットの異常な振動や動きを検知し、自己修復機能を活性化します'), Document(metadata={'source': '/tmp/rocket.pdf', 'page': 0}, page_content='総じて、この新開発のロケットエンジンは、革新的な技術と未来志向の設計によって、宇宙探査の新た\nな時代を切り開くことでしょう'), Document(metadata={'source': '/tmp/rocket.pdf', 'page': 0}, page_content='その高い性能と信頼性は、人類の夢を実現するための力強い支援とな\nることでしょう')]
```


```python
from langchain_community.vectorstores.oraclevs import OracleVS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.embeddings import OCIGenAIEmbeddings

embeddings = OCIGenAIEmbeddings(
    model_id="cohere.embed-multilingual-v3.0",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="<compartmentのOCID>",
)


vector_store_dot = OracleVS.from_documents(
    docs,
    embeddings,
    client=connection,
    table_name="doc_table",
    distance_strategy=DistanceStrategy.DOT_PRODUCT,
)
```


### PostgreSQL
データベースに接続、会話履歴をロードする表の作成

```python
import psycopg

conn_info = (
    "postgresql://<user>:<passwd>@<id adress>/<database name>"
    "?sslmode=require"
    "&sslrootcert=/home/opc/postgre/CaCertificate-postgresql.pub"
)

# PostgreSQLに接続
try:
    sync_connection = psycopg.connect(conn_info)
    print("Connection successful!")
except Exception as e:
    print(f"Connection failed!: {e}")
```

```python
from langchain_postgres import PostgresChatMessageHistory

# 履歴を保存する表を作成
table_name = "message_store"
PostgresChatMessageHistory.create_tables(sync_connection, table_name)
```



PostgreSQLに接続し、クエリを実行

```sql
(base) [opc@ol9 ~]$ psql -h 10.0.1.254 -p 5432 -U ksonoda -d postgres
Password for user ksonoda: 
psql (16.1, server 14.11)
SSL connection (protocol: TLSv1.2, cipher: ECDHE-RSA-AES256-GCM-SHA384, compression: off)
Type "help" for help.

postgres=> 


作成した表(message_store)を確認

postgres=> \dt;
            List of relations
 Schema |     Name      | Type  |  Owner  
--------+---------------+-------+---------
 public | message_store | table | ksonoda
(1 row)

クエリを実行(現時点では0件)

postgres=> select * from message_store;
 id | session_id | message | created_at 
----+------------+---------+------------
(0 rows)

postgres=> 
```



# RAGの実装

```python
from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI

llm = ChatOCIGenAI(
    #model_id="cohere.command-r-16k",
    model_id="cohere.command-r-plus",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="<compartmentのOCID>",
    model_kwargs={"temperature": 0.7, "max_tokens": 500},
)
```

```python
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever

# 会話コンテキストに沿ってクエリ変換用retrieverを定義
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
        
    ]
)

retriever = vector_store_dot.as_retriever(search_kwargs={"k": 3})

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)
```

```python
### 質問応答のチェーン

qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

{context}"""



from langchain.chains.combine_documents import create_stuff_documents_chain

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
```


```python
from langchain.chains import create_retrieval_chain

# クエリ変換用と質問応答チェーン用のretrieverからチェーンを定義
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
```

```python
from langchain_core.chat_history import BaseChatMessageHistory
import uuid

# 会話セッションのIDを設定
session_id = str(uuid.uuid4())

def get_session_history(session_id: str) -> BaseChatMessageHistory:
   return PostgresChatMessageHistory(
        table_name,
        session_id,
        sync_connection=sync_connection
   )
```


```python
from langchain_core.runnables.history import RunnableWithMessageHistory

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
    output_messages_key="answer",
)
```


```python
response = conversational_rag_chain.invoke(
   {"input": "OraBoosterとは何ですか？"},
   config={"configurable": {"session_id": session_id}},
)

response


{'input': 'OraBoosterとは何ですか？',
 'history': [],
 'context': [Document(metadata={'source': '/tmp/rocket.pdf', 'page': 0}, page_content='当社が開発したロケットエンジンである OraBooster は、次世代の宇宙探査を支える先進的な推進技術の\n象徴です'),
  Document(metadata={'source': '/tmp/rocket.pdf', 'page': 0}, page_content='その高い性能と信頼性は、人類の夢を実現するための力強い支援とな\nることでしょう'),
  Document(metadata={'source': '/tmp/rocket.pdf', 'page': 0}, page_content='また、ハイパーフォトン・ジャイロスコープが搭載されており、極めて高い精度でロケットの姿勢を維\n持し、目標を追跡します')],
 'answer': 'OraBooster は、次世代の宇宙探査を支えるために開発された先進的なロケットエンジンです。高い性能と信頼性、そしてハイパーフォトン・ジャイロスコープによる高精度の姿勢制御を特長とします。'}
```



 PostgreSQLでクエリ実行(inputとanswerの2件が追加されていることを確認)

```sql

postgres=> select * from message_store;
 id |              session_id              |                                                                                                                             
                                                                         message                                                                                         
                                                                                                              |          created_at           
----+--------------------------------------+-----------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------+-------------------------------
  1 | 5e2a952c-59a4-40e1-9355-f688a3dbf27e | {"data": {"id": null, "name": null, "type": "human", "content": "OraBoosterとは何ですか？", "example": false, "additional_kw
args": {}, "response_metadata": {}}, "type": "human"}                                                                                                                    
                                                                                                              | 2024-07-24 07:01:39.320933+00
  2 | 5e2a952c-59a4-40e1-9355-f688a3dbf27e | {"data": {"id": null, "name": null, "type": "ai", "content": "OraBooster は、次世代の宇宙探査を支える先進的な推進技術を備え
ロケットエンジンです。高い性能と信頼性、そしてハイパーフォトン・ジャイロスコープによる極めて高い姿勢制御精度を特長とします。", "example": false, "tool_calls": [], "usage
_metadata": null, "additional_kwargs": {}, "response_metadata": {}, "invalid_tool_calls": []}, "type": "ai"} | 2024-07-24 07:01:39.320933+00
(2 rows)

postgres=> 
```


```python
response  = conversational_rag_chain.invoke(
   {"input": "それは実在するものですか？"},
   config={"configurable": {"session_id": session_id}},
)

response


{'input': 'それは実在するものですか？',
 'history': [HumanMessage(content='OraBoosterとは何ですか？'),
  AIMessage(content='OraBooster は、次世代の宇宙探査を支えるために開発された先進的なロケットエンジンです。高い性能と信頼性、そしてハイパーフォトン・ジャイロスコープによる高精度の姿勢制御を特長とします。')],
 'context': [Document(metadata={'source': '/tmp/rocket.pdf', 'page': 0}, page_content='その高い性能と信頼性は、人類の夢を実現するための力強い支援とな\nることでしょう'),
  Document(metadata={'source': '/tmp/rocket.pdf', 'page': 0}, page_content='また、ハイパーフォトン・ジャイロスコープが搭載されており、極めて高い精度でロケットの姿勢を維\n持し、目標を追跡します'),
  Document(metadata={'source': '/tmp/rocket.pdf', 'page': 0}, page_content='これにより、長時間にわたる宇宙飛行中でも安定した飛行軌道を維持し、\nミッションの成功を確保します。\nさらに、バイオニック・リアクション・レスポンダーが統合されています')],
 'answer': 'いいえ、OraBooster は架空のロケットエンジンです。'}
```


```python
 response  = conversational_rag_chain.invoke(
   {"input": "それはいつ開発されましたか"},
   config={"configurable": {"session_id": session_id}},
)

response



{'input': 'それはいつ開発されましたか',
 'history': [HumanMessage(content='OraBoosterとは何ですか？'),
  AIMessage(content='OraBooster は、次世代の宇宙探査を支えるために開発された先進的なロケットエンジンです。高い性能と信頼性、そしてハイパーフォトン・ジャイロスコープによる高精度の姿勢制御を特長とします。'),
  HumanMessage(content='それは実在するものですか？'),
  AIMessage(content='いいえ、OraBooster は架空のロケットエンジンです。')],
 'context': [Document(metadata={'source': '/tmp/rocket.pdf', 'page': 0}, page_content='その高い性能と信頼性は、人類の夢を実現するための力強い支援とな\nることでしょう'),
  Document(metadata={'source': '/tmp/rocket.pdf', 'page': 0}, page_content='また、ハイパーフォトン・ジャイロスコープが搭載されており、極めて高い精度でロケットの姿勢を維\n持し、目標を追跡します'),
  Document(metadata={'source': '/tmp/rocket.pdf', 'page': 0}, page_content='総じて、この新開発のロケットエンジンは、革新的な技術と未来志向の設計によって、宇宙探査の新た\nな時代を切り開くことでしょう')],
 'answer': '開発時期は不明です。'}
```



 
PostgreSQLでクエリ実行(追加のinputとanswerが追加されていることを確認)

```sql
postgres=> select * from message_store;
 id |              session_id              |                                                                                                                             
                                                                         message                                                                                         
                                                                                                              |          created_at           
----+--------------------------------------+-----------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------+-------------------------------
  1 | 5e2a952c-59a4-40e1-9355-f688a3dbf27e | {"data": {"id": null, "name": null, "type": "human", "content": "OraBoosterとは何ですか？", "example": false, "additional_kw
args": {}, "response_metadata": {}}, "type": "human"}                                                                                                                    
                                                                                                              | 2024-07-24 07:01:39.320933+00
  2 | 5e2a952c-59a4-40e1-9355-f688a3dbf27e | {"data": {"id": null, "name": null, "type": "ai", "content": "OraBooster は、次世代の宇宙探査を支える先進的な推進技術を備え
ロケットエンジンです。高い性能と信頼性、そしてハイパーフォトン・ジャイロスコープによる極めて高い姿勢制御精度を特長とします。", "example": false, "tool_calls": [], "usage
_metadata": null, "additional_kwargs": {}, "response_metadata": {}, "invalid_tool_calls": []}, "type": "ai"} | 2024-07-24 07:01:39.320933+00
(2 rows)

postgres=> 
postgres=> select * from message_store;
 id |              session_id              |                                                                                                                             
                                                                         message                                                                                         
                                                                                                              |          created_at           
----+--------------------------------------+-----------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------+-------------------------------
  1 | 5e2a952c-59a4-40e1-9355-f688a3dbf27e | {"data": {"id": null, "name": null, "type": "human", "content": "OraBoosterとは何ですか？", "example": false, "additional_kw
args": {}, "response_metadata": {}}, "type": "human"}                                                                                                                    
                                                                                                              | 2024-07-24 07:01:39.320933+00
  2 | 5e2a952c-59a4-40e1-9355-f688a3dbf27e | {"data": {"id": null, "name": null, "type": "ai", "content": "OraBooster は、次世代の宇宙探査を支える先進的な推進技術を備え
ロケットエンジンです。高い性能と信頼性、そしてハイパーフォトン・ジャイロスコープによる極めて高い姿勢制御精度を特長とします。", "example": false, "tool_calls": [], "usage
_metadata": null, "additional_kwargs": {}, "response_metadata": {}, "invalid_tool_calls": []}, "type": "ai"} | 2024-07-24 07:01:39.320933+00
  3 | 5e2a952c-59a4-40e1-9355-f688a3dbf27e | {"data": {"id": null, "name": null, "type": "human", "content": "それは実在するものですか？", "example": false, "additional_
kwargs": {}, "response_metadata": {}}, "type": "human"}                                                                                                                  
                                                                                                              | 2024-07-24 07:03:40.984928+00
  4 | 5e2a952c-59a4-40e1-9355-f688a3dbf27e | {"data": {"id": null, "name": null, "type": "ai", "content": "いいえ、OraBooster は架空のロケットエンジンです。", "example":
 false, "tool_calls": [], "usage_metadata": null, "additional_kwargs": {}, "response_metadata": {}, "invalid_tool_calls": []}, "type": "ai"}                             
                                                                                                              | 2024-07-24 07:03:40.984928+00
(4 rows)

postgres=> 
```





```python
response  = conversational_rag_chain.invoke(
   {"input": "いつ使われる予定ですか？"},
   config={"configurable": {"session_id": session_id}},
)

response



{'input': 'いつ使われる予定ですか？',
 'history': [HumanMessage(content='OraBoosterとは何ですか？'),
  AIMessage(content='OraBooster は、次世代の宇宙探査を支えるために開発された先進的なロケットエンジンです。高い性能と信頼性、そしてハイパーフォトン・ジャイロスコープによる高精度の姿勢制御を特長とします。'),
  HumanMessage(content='それは実在するものですか？'),
  AIMessage(content='いいえ、OraBooster は架空のロケットエンジンです。'),
  HumanMessage(content='それはいつ開発されましたか'),
  AIMessage(content='開発時期は不明です。')],
 'context': [Document(metadata={'source': '/tmp/rocket.pdf', 'page': 0}, page_content='その高い性能と信頼性は、人類の夢を実現するための力強い支援とな\nることでしょう'),
  Document(metadata={'source': '/tmp/rocket.pdf', 'page': 0}, page_content='また、ハイパーフォトン・ジャイロスコープが搭載されており、極めて高い精度でロケットの姿勢を維\n持し、目標を追跡します'),
  Document(metadata={'source': '/tmp/rocket.pdf', 'page': 0}, page_content='総じて、この新開発のロケットエンジンは、革新的な技術と未来志向の設計によって、宇宙探査の新た\nな時代を切り開くことでしょう')],
 'answer': '私は使用予定について知りません。'}
```



```python
response  = conversational_rag_chain.invoke(
   {"input": "重量はどれくらいですか？"},
   config={"configurable": {"session_id": session_id}},
)

response


{'input': '重量はどれくらいですか？',
 'history': [HumanMessage(content='OraBoosterとは何ですか？'),
  AIMessage(content='OraBooster は、次世代の宇宙探査を支えるために開発された先進的なロケットエンジンです。高い性能と信頼性、そしてハイパーフォトン・ジャイロスコープによる高精度の姿勢制御を特長とします。'),
  HumanMessage(content='それは実在するものですか？'),
  AIMessage(content='いいえ、OraBooster は架空のロケットエンジンです。'),
  HumanMessage(content='それはいつ開発されましたか'),
  AIMessage(content='開発時期は不明です。'),
  HumanMessage(content='いつ使われる予定ですか？'),
  AIMessage(content='私は使用予定について知りません。')],
 'context': [Document(metadata={'source': '/tmp/rocket.pdf', 'page': 0}, page_content='その高い性能と信頼性は、人類の夢を実現するための力強い支援とな\nることでしょう'),
  Document(metadata={'source': '/tmp/rocket.pdf', 'page': 0}, page_content='また、ハイパーフォトン・ジャイロスコープが搭載されており、極めて高い精度でロケットの姿勢を維\n持し、目標を追跡します'),
  Document(metadata={'source': '/tmp/rocket.pdf', 'page': 0}, page_content='これにより、長時間にわたる宇宙飛行中でも安定した飛行軌道を維持し、\nミッションの成功を確保します。\nさらに、バイオニック・リアクション・レスポンダーが統合されています')],
 'answer': '私は重量の情報にアクセスできません。'}
```


```python
response  = conversational_rag_chain.invoke(
   {"input": "姿勢制御に使われている技術は何ですか？"},
   config={"configurable": {"session_id": session_id}},
)

response




{'input': '姿勢制御に使われている技術は何ですか？',
 'history': [HumanMessage(content='OraBoosterとは何ですか？'),
  AIMessage(content='OraBooster は、次世代の宇宙探査を支えるために開発された先進的なロケットエンジンです。高い性能と信頼性、そしてハイパーフォトン・ジャイロスコープによる高精度の姿勢制御を特長とします。'),
  HumanMessage(content='それは実在するものですか？'),
  AIMessage(content='いいえ、OraBooster は架空のロケットエンジンです。'),
  HumanMessage(content='それはいつ開発されましたか'),
  AIMessage(content='開発時期は不明です。'),
  HumanMessage(content='いつ使われる予定ですか？'),
  AIMessage(content='私は使用予定について知りません。'),
  HumanMessage(content='重量はどれくらいですか？'),
  AIMessage(content='私は重量の情報にアクセスできません。')],
 'context': [Document(metadata={'source': '/tmp/rocket.pdf', 'page': 0}, page_content='また、ハイパーフォトン・ジャイロスコープが搭載されており、極めて高い精度でロケットの姿勢を維\n持し、目標を追跡します'),
  Document(metadata={'source': '/tmp/rocket.pdf', 'page': 0}, page_content='これにより、長時間にわたる宇宙飛行中でも安定した飛行軌道を維持し、\nミッションの成功を確保します。\nさらに、バイオニック・リアクション・レスポンダーが統合されています'),
  Document(metadata={'source': '/tmp/rocket.pdf', 'page': 0}, page_content='その高い性能と信頼性は、人類の夢を実現するための力強い支援とな\nることでしょう')],
 'answer': '姿勢制御には、ハイパーフォトン・ジャイロスコープ技術が使用されています。'}
```


```python
response  = conversational_rag_chain.invoke(
   {"input": "それは重要なものですか？"},
   config={"configurable": {"session_id": session_id}},
)

response




{'input': 'それは重要なものですか？',
 'history': [HumanMessage(content='OraBoosterとは何ですか？'),
  AIMessage(content='OraBooster は、次世代の宇宙探査を支えるために開発された先進的なロケットエンジンです。高い性能と信頼性、そしてハイパーフォトン・ジャイロスコープによる高精度の姿勢制御を特長とします。'),
  HumanMessage(content='それは実在するものですか？'),
  AIMessage(content='いいえ、OraBooster は架空のロケットエンジンです。'),
  HumanMessage(content='それはいつ開発されましたか'),
  AIMessage(content='開発時期は不明です。'),
  HumanMessage(content='いつ使われる予定ですか？'),
  AIMessage(content='私は使用予定について知りません。'),
  HumanMessage(content='重量はどれくらいですか？'),
  AIMessage(content='私は重量の情報にアクセスできません。'),
  HumanMessage(content='姿勢制御に使われている技術は何ですか？'),
  AIMessage(content='姿勢制御には、ハイパーフォトン・ジャイロスコープ技術が使用されています。')],
 'context': [Document(metadata={'source': '/tmp/rocket.pdf', 'page': 0}, page_content='その高い性能と信頼性は、人類の夢を実現するための力強い支援とな\nることでしょう'),
  Document(metadata={'source': '/tmp/rocket.pdf', 'page': 0}, page_content='また、ハイパーフォトン・ジャイロスコープが搭載されており、極めて高い精度でロケットの姿勢を維\n持し、目標を追跡します'),
  Document(metadata={'source': '/tmp/rocket.pdf', 'page': 0}, page_content='これにより、長時間にわたる宇宙飛行中でも安定した飛行軌道を維持し、\nミッションの成功を確保します。\nさらに、バイオニック・リアクション・レスポンダーが統合されています')],
 'answer': 'はい、重要です。'}
```

```python
response  = conversational_rag_chain.invoke(
   {"input": "それがないとどうなりますか？"},
   config={"configurable": {"session_id": session_id}},
)

response



{'input': 'それがないとどうなりますか？',
 'history': [HumanMessage(content='OraBoosterとは何ですか？'),
  AIMessage(content='OraBooster は、次世代の宇宙探査を支えるために開発された先進的なロケットエンジンです。高い性能と信頼性、そしてハイパーフォトン・ジャイロスコープによる高精度の姿勢制御を特長とします。'),
  HumanMessage(content='それは実在するものですか？'),
  AIMessage(content='いいえ、OraBooster は架空のロケットエンジンです。'),
  HumanMessage(content='それはいつ開発されましたか'),
  AIMessage(content='開発時期は不明です。'),
  HumanMessage(content='いつ使われる予定ですか？'),
  AIMessage(content='私は使用予定について知りません。'),
  HumanMessage(content='重量はどれくらいですか？'),
  AIMessage(content='私は重量の情報にアクセスできません。'),
  HumanMessage(content='姿勢制御に使われている技術は何ですか？'),
  AIMessage(content='姿勢制御には、ハイパーフォトン・ジャイロスコープ技術が使用されています。'),
  HumanMessage(content='それは重要なものですか？'),
  AIMessage(content='はい、重要です。')],
 'context': [Document(metadata={'source': '/tmp/rocket.pdf', 'page': 0}, page_content='その高い性能と信頼性は、人類の夢を実現するための力強い支援とな\nることでしょう'),
  Document(metadata={'source': '/tmp/rocket.pdf', 'page': 0}, page_content='これにより、長時間にわたる宇宙飛行中でも安定した飛行軌道を維持し、\nミッションの成功を確保します。\nさらに、バイオニック・リアクション・レスポンダーが統合されています'),
  Document(metadata={'source': '/tmp/rocket.pdf', 'page': 0}, page_content='また、ハイパーフォトン・ジャイロスコープが搭載されており、極めて高い精度でロケットの姿勢を維\n持し、目標を追跡します')],
 'answer': 'ミッションの成功確保が難しくなります。'}
```
