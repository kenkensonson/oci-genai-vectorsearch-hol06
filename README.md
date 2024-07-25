{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "eb68a475-d037-4d97-99ab-3bcba3dd314c",
   "metadata": {},
   "source": [
    "# 会話履歴保持の仕組みを取り入れたRAGの実装\n",
    "\n",
    "### 構成\n",
    "ドキュメント保持用ベクトルデータベース：Oracle DB23ai AI Vector Search  \n",
    "会話履歴保持用データベース : OCI PosgreSQL Database Service  \n",
    "大規模言語モデル : OCI Generative AI Service(Command-R-Plus)  \n",
    "\n",
    "### 参考にしたサンプルコード  \n",
    "\n",
    "LangChainの会話履歴の仕組みを追加するサンプルコード  \n",
    "https://python.langchain.com/v0.2/docs/how_to/qa_chat_history_how_to/  \n",
    "\n",
    "会話履歴をPosgreSQLに記録するコード  \n",
    "https://hexacluster.ai/postgresql/postgres-for-chat-history-langchain-postgres-postgreschatmessagehistory/  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "52117618-8781-4626-9c3c-f825d0fb5009",
   "metadata": {},
   "outputs": [],
   "source": [
    "!pip install --upgrade pip\n",
    "!pip install -Uq oracledb pypdf cohere langchain langchain-community langchain-core langchain_postgres oci grandalf psycopg"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9ca7801c-a516-4277-bfd9-8e5e9b6d8a85",
   "metadata": {},
   "source": [
    "### Oracle DB\n",
    "データベースの接続、pdfファイルの埋め込みとロード"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "69973a3a-a35f-41fe-acdd-4bdab0d5ee1c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Connection successful!\n"
     ]
    }
   ],
   "source": [
    "import oracledb\n",
    "\n",
    "username = \"docuser\"\n",
    "password = \"docuser\"\n",
    "dsn = \"localhost/freepdb1\"\n",
    "\n",
    "try:\n",
    "    connection = oracledb.connect(user=username, password=password, dsn=dsn)\n",
    "    print(\"Connection successful!\")\n",
    "except Exception as e:\n",
    "    print(\"Connection failed!\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "19f14f6e-6449-4a9b-9831-5bb4f2516f86",
   "metadata": {},
   "outputs": [],
   "source": [
    "from langchain.document_loaders import PyPDFLoader\n",
    "\n",
    "loader = PyPDFLoader(\"/tmp/rocket.pdf\")\n",
    "documents = loader.load_and_split()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "1cbe46ce-0c1e-4e1b-85fd-05c6051a1d22",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[Document(metadata={'source': '/tmp/rocket.pdf', 'page': 0}, page_content='当社が開発したロケットエンジンである OraBooster は、次世代の宇宙探査を支える先進的な推進技術の\\n象徴です。その独自の設計は、高性能と革新性を融合させ、人類の宇宙進出を加速させるための革命的\\nな一歩となります。\\nこのエンジンの核となるのは、量子ダイナミックス・プラズマ・ブースターです。このブースターは、\\n量子力学の原理に基づいてプラズマを生成し、超高速で加速させます。その結果、従来の化学反応より\\nもはるかに高い推力を発生し、遠く離れた惑星や星系への探査を可能にします。\\nさらに、エンジンの外殻にはナノファイバー製の超軽量かつ超強度の素材が使用されています。この素\\n材は、宇宙空間の過酷な環境に耐え、高速での飛行中に生じる熱や衝撃からロケットを守ります。\\nまた、ハイパーフォトン・ジャイロスコープが搭載されており、極めて高い精度でロケットの姿勢を維\\n持し、目標を追跡します。これにより、長時間にわたる宇宙飛行中でも安定した飛行軌道を維持し、\\nミッションの成功を確保します。\\nさらに、バイオニック・リアクション・レスポンダーが統合されています。このシステムは、人工知能\\nと生体認識技術を組み合わせ、ロケットの異常な振動や動きを検知し、自己修復機能を活性化します。\\n総じて、この新開発のロケットエンジンは、革新的な技術と未来志向の設計によって、宇宙探査の新た\\nな時代を切り開くことでしょう。その高い性能と信頼性は、人類の夢を実現するための力強い支援とな\\nることでしょう。')]"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "documents "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "7ab22a95-ae16-41c3-b77a-70fcf3459783",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[Document(metadata={'source': '/tmp/rocket.pdf', 'page': 0}, page_content='当社が開発したロケットエンジンである OraBooster は、次世代の宇宙探査を支える先進的な推進技術の\\n象徴です'), Document(metadata={'source': '/tmp/rocket.pdf', 'page': 0}, page_content='その独自の設計は、高性能と革新性を融合させ、人類の宇宙進出を加速させるための革命的\\nな一歩となります。\\nこのエンジンの核となるのは、量子ダイナミックス・プラズマ・ブースターです'), Document(metadata={'source': '/tmp/rocket.pdf', 'page': 0}, page_content='このブースターは、\\n量子力学の原理に基づいてプラズマを生成し、超高速で加速させます。その結果、従来の化学反応より\\nもはるかに高い推力を発生し、遠く離れた惑星や星系への探査を可能にします'), Document(metadata={'source': '/tmp/rocket.pdf', 'page': 0}, page_content='さらに、エンジンの外殻にはナノファイバー製の超軽量かつ超強度の素材が使用されています。この素\\n材は、宇宙空間の過酷な環境に耐え、高速での飛行中に生じる熱や衝撃からロケットを守ります'), Document(metadata={'source': '/tmp/rocket.pdf', 'page': 0}, page_content='また、ハイパーフォトン・ジャイロスコープが搭載されており、極めて高い精度でロケットの姿勢を維\\n持し、目標を追跡します'), Document(metadata={'source': '/tmp/rocket.pdf', 'page': 0}, page_content='これにより、長時間にわたる宇宙飛行中でも安定した飛行軌道を維持し、\\nミッションの成功を確保します。\\nさらに、バイオニック・リアクション・レスポンダーが統合されています'), Document(metadata={'source': '/tmp/rocket.pdf', 'page': 0}, page_content='このシステムは、人工知能\\nと生体認識技術を組み合わせ、ロケットの異常な振動や動きを検知し、自己修復機能を活性化します'), Document(metadata={'source': '/tmp/rocket.pdf', 'page': 0}, page_content='総じて、この新開発のロケットエンジンは、革新的な技術と未来志向の設計によって、宇宙探査の新た\\nな時代を切り開くことでしょう'), Document(metadata={'source': '/tmp/rocket.pdf', 'page': 0}, page_content='その高い性能と信頼性は、人類の夢を実現するための力強い支援とな\\nることでしょう')]\n"
     ]
    }
   ],
   "source": [
    "from langchain.text_splitter import CharacterTextSplitter\n",
    "\n",
    "text_splitter = CharacterTextSplitter(separator=\"。\", chunk_size=100, chunk_overlap=10)\n",
    "docs = text_splitter.split_documents(documents)\n",
    "print(docs)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "b2cbca6b-3d1b-4831-ad48-06c08390102b",
   "metadata": {},
   "outputs": [],
   "source": [
    "from langchain_community.vectorstores.oraclevs import OracleVS\n",
    "from langchain_community.vectorstores.utils import DistanceStrategy\n",
    "from langchain_community.embeddings import OCIGenAIEmbeddings"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "348d00bd-e2fb-4310-accf-53d5058c1f55",
   "metadata": {},
   "outputs": [],
   "source": [
    "embeddings = OCIGenAIEmbeddings(\n",
    "    model_id=\"cohere.embed-multilingual-v3.0\",\n",
    "    service_endpoint=\"https://inference.generativeai.us-chicago-1.oci.oraclecloud.com\",\n",
    "    compartment_id=\"<compartmentのOCID>\",\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "b1b08589-10df-4e0b-8d84-a310c806d39c",
   "metadata": {},
   "outputs": [],
   "source": [
    "vector_store_dot = OracleVS.from_documents(\n",
    "    docs,\n",
    "    embeddings,\n",
    "    client=connection,\n",
    "    table_name=\"doc_table\",\n",
    "    distance_strategy=DistanceStrategy.DOT_PRODUCT,\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9f4e82d5-e8ad-4087-bb71-5f0c2dc155a0",
   "metadata": {},
   "source": [
    "### PostgreSQL\n",
    "データベースに接続、会話履歴をロードする表の作成"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "97263a47-662e-44ce-82d1-d2e888e863a6",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Connection successful!\n"
     ]
    }
   ],
   "source": [
    "import psycopg\n",
    "\n",
    "conn_info = (\n",
    "    \"postgresql://<user>:<passwd>@<id adress>/<database name>\"\n",
    "    \"?sslmode=require\"\n",
    "    \"&sslrootcert=/home/opc/postgre/CaCertificate-postgresql.pub\"\n",
    ")\n",
    "\n",
    "# PostgreSQLに接続\n",
    "try:\n",
    "    sync_connection = psycopg.connect(conn_info)\n",
    "    print(\"Connection successful!\")\n",
    "except Exception as e:\n",
    "    print(f\"Connection failed!: {e}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "6f414150-c5c4-4310-b6a7-c1a880d62f66",
   "metadata": {},
   "outputs": [],
   "source": [
    "from langchain_postgres import PostgresChatMessageHistory\n",
    "\n",
    "# 履歴を保存する表を作成\n",
    "table_name = \"message_store\"\n",
    "PostgresChatMessageHistory.create_tables(sync_connection, table_name)"
   ]
  },
  {
   "cell_type": "raw",
   "id": "ad6159f9-b862-48ee-b729-96f0fffea489",
   "metadata": {},
   "source": [
    "PostgreSQLに接続し、クエリを実行\n",
    "\n",
    "(base) [opc@ol9 ~]$ psql -h 10.0.1.254 -p 5432 -U ksonoda -d postgres\n",
    "Password for user ksonoda: \n",
    "psql (16.1, server 14.11)\n",
    "SSL connection (protocol: TLSv1.2, cipher: ECDHE-RSA-AES256-GCM-SHA384, compression: off)\n",
    "Type \"help\" for help.\n",
    "\n",
    "postgres=> \n",
    "\n",
    "作成した表(message_store)を確認\n",
    "\n",
    "postgres=> \\dt;\n",
    "            List of relations\n",
    " Schema |     Name      | Type  |  Owner  \n",
    "--------+---------------+-------+---------\n",
    " public | message_store | table | ksonoda\n",
    "(1 row)\n",
    "\n",
    "クエリを実行(現時点では0件)\n",
    "\n",
    "postgres=> select * from message_store;\n",
    " id | session_id | message | created_at \n",
    "----+------------+---------+------------\n",
    "(0 rows)\n",
    "\n",
    "postgres=> "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "49594f70-b059-40bf-936b-ec0eb482bd9f",
   "metadata": {},
   "source": [
    "### RAGの実装"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "a0d71d2e-b576-4eb1-abb9-1ce9cd7ee11e",
   "metadata": {},
   "outputs": [],
   "source": [
    "from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI\n",
    "\n",
    "llm = ChatOCIGenAI(\n",
    "    #model_id=\"cohere.command-r-16k\",\n",
    "    model_id=\"cohere.command-r-plus\",\n",
    "    service_endpoint=\"https://inference.generativeai.us-chicago-1.oci.oraclecloud.com\",\n",
    "    compartment_id=\"<compartmentのOCID>\",\n",
    "    model_kwargs={\"temperature\": 0.7, \"max_tokens\": 500},\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "93c3236f-92ec-4ee4-a8b3-7f303fd277a6",
   "metadata": {},
   "outputs": [],
   "source": [
    "from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder\n",
    "from langchain.chains import create_history_aware_retriever\n",
    "\n",
    "# 会話コンテキストに沿ってクエリ変換用retrieverを定義\n",
    "contextualize_q_system_prompt = \"\"\"Given a chat history and the latest user question \\\n",
    "which might reference context in the chat history, formulate a standalone question \\\n",
    "which can be understood without the chat history. Do NOT answer the question, \\\n",
    "just reformulate it if needed and otherwise return it as is.\"\"\"\n",
    "\n",
    "contextualize_q_prompt = ChatPromptTemplate.from_messages(\n",
    "    [\n",
    "        (\"system\", contextualize_q_system_prompt),\n",
    "        MessagesPlaceholder(\"history\"),\n",
    "        (\"human\", \"{input}\"),\n",
    "        \n",
    "    ]\n",
    ")\n",
    "\n",
    "retriever = vector_store_dot.as_retriever(search_kwargs={\"k\": 3})\n",
    "\n",
    "history_aware_retriever = create_history_aware_retriever(\n",
    "    llm, retriever, contextualize_q_prompt\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "6c5876e1-2319-4cfc-9803-e0e91463bcc1",
   "metadata": {},
   "outputs": [],
   "source": [
    "### 質問応答のチェーン\n",
    "\n",
    "qa_system_prompt = \"\"\"You are an assistant for question-answering tasks. \\\n",
    "Use the following pieces of retrieved context to answer the question. \\\n",
    "If you don't know the answer, just say that you don't know. \\\n",
    "Use three sentences maximum and keep the answer concise.\\\n",
    "\n",
    "{context}\"\"\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "a9501358-23da-49b8-bd30-0f6a5d22f9da",
   "metadata": {},
   "outputs": [],
   "source": [
    "from langchain.chains.combine_documents import create_stuff_documents_chain\n",
    "\n",
    "qa_prompt = ChatPromptTemplate.from_messages(\n",
    "    [\n",
    "        (\"system\", qa_system_prompt),\n",
    "        MessagesPlaceholder(\"history\"),\n",
    "        (\"human\", \"{input}\"),\n",
    "    ]\n",
    ")\n",
    "\n",
    "question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "39bf41c6-7045-4992-9617-171e8e973077",
   "metadata": {},
   "outputs": [],
   "source": [
    "from langchain.chains import create_retrieval_chain\n",
    "\n",
    "# クエリ変換用と質問応答チェーン用のretrieverからチェーンを定義\n",
    "rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "bb660076-8183-4fe4-9011-71ce67548a02",
   "metadata": {},
   "outputs": [],
   "source": [
    "from langchain_core.chat_history import BaseChatMessageHistory\n",
    "import uuid\n",
    "\n",
    "# 会話セッションのIDを設定\n",
    "session_id = str(uuid.uuid4())\n",
    "\n",
    "def get_session_history(session_id: str) -> BaseChatMessageHistory:\n",
    "   return PostgresChatMessageHistory(\n",
    "        table_name,\n",
    "        session_id,\n",
    "        sync_connection=sync_connection\n",
    "   )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "11f9db5f-295c-43eb-af89-1c837e3440ad",
   "metadata": {},
   "outputs": [],
   "source": [
    "from langchain_core.runnables.history import RunnableWithMessageHistory\n",
    "\n",
    "conversational_rag_chain = RunnableWithMessageHistory(\n",
    "    rag_chain,\n",
    "    get_session_history,\n",
    "    input_messages_key=\"input\",\n",
    "    history_messages_key=\"history\",\n",
    "    output_messages_key=\"answer\",\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "da534327-336a-4e84-9a91-07728df62663",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'input': 'OraBoosterとは何ですか？',\n",
       " 'history': [],\n",
       " 'context': [Document(metadata={'source': '/tmp/rocket.pdf', 'page': 0}, page_content='当社が開発したロケットエンジンである OraBooster は、次世代の宇宙探査を支える先進的な推進技術の\\n象徴です'),\n",
       "  Document(metadata={'source': '/tmp/rocket.pdf', 'page': 0}, page_content='その高い性能と信頼性は、人類の夢を実現するための力強い支援とな\\nることでしょう'),\n",
       "  Document(metadata={'source': '/tmp/rocket.pdf', 'page': 0}, page_content='また、ハイパーフォトン・ジャイロスコープが搭載されており、極めて高い精度でロケットの姿勢を維\\n持し、目標を追跡します')],\n",
       " 'answer': 'OraBooster は、次世代の宇宙探査を支えるために開発された先進的なロケットエンジンです。高い性能と信頼性、そしてハイパーフォトン・ジャイロスコープによる高精度の姿勢制御を特長とします。'}"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "response = conversational_rag_chain.invoke(\n",
    "   {\"input\": \"OraBoosterとは何ですか？\"},\n",
    "   config={\"configurable\": {\"session_id\": session_id}},\n",
    ")\n",
    "\n",
    "response"
   ]
  },
  {
   "cell_type": "raw",
   "id": "efb8aa84-277b-4360-bf65-20b9971d2f5b",
   "metadata": {},
   "source": [
    "PostgreSQLでクエリ実行(inputとanswerの2件が追加されていることを確認)\n",
    "\n",
    "\n",
    "postgres=> select * from message_store;\n",
    " id |              session_id              |                                                                                                                             \n",
    "                                                                         message                                                                                         \n",
    "                                                                                                              |          created_at           \n",
    "----+--------------------------------------+-----------------------------------------------------------------------------------------------------------------------------\n",
    "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n",
    "--------------------------------------------------------------------------------------------------------------+-------------------------------\n",
    "  1 | 5e2a952c-59a4-40e1-9355-f688a3dbf27e | {\"data\": {\"id\": null, \"name\": null, \"type\": \"human\", \"content\": \"OraBoosterとは何ですか？\", \"example\": false, \"additional_kw\n",
    "args\": {}, \"response_metadata\": {}}, \"type\": \"human\"}                                                                                                                    \n",
    "                                                                                                              | 2024-07-24 07:01:39.320933+00\n",
    "  2 | 5e2a952c-59a4-40e1-9355-f688a3dbf27e | {\"data\": {\"id\": null, \"name\": null, \"type\": \"ai\", \"content\": \"OraBooster は、次世代の宇宙探査を支える先進的な推進技術を備え\n",
    "ロケットエンジンです。高い性能と信頼性、そしてハイパーフォトン・ジャイロスコープによる極めて高い姿勢制御精度を特長とします。\", \"example\": false, \"tool_calls\": [], \"usage\n",
    "_metadata\": null, \"additional_kwargs\": {}, \"response_metadata\": {}, \"invalid_tool_calls\": []}, \"type\": \"ai\"} | 2024-07-24 07:01:39.320933+00\n",
    "(2 rows)\n",
    "\n",
    "postgres=> "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "5a267961-05fa-4cfb-981f-a8f4ffe346ab",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'input': 'それは実在するものですか？',\n",
       " 'history': [HumanMessage(content='OraBoosterとは何ですか？'),\n",
       "  AIMessage(content='OraBooster は、次世代の宇宙探査を支えるために開発された先進的なロケットエンジンです。高い性能と信頼性、そしてハイパーフォトン・ジャイロスコープによる高精度の姿勢制御を特長とします。')],\n",
       " 'context': [Document(metadata={'source': '/tmp/rocket.pdf', 'page': 0}, page_content='その高い性能と信頼性は、人類の夢を実現するための力強い支援とな\\nることでしょう'),\n",
       "  Document(metadata={'source': '/tmp/rocket.pdf', 'page': 0}, page_content='また、ハイパーフォトン・ジャイロスコープが搭載されており、極めて高い精度でロケットの姿勢を維\\n持し、目標を追跡します'),\n",
       "  Document(metadata={'source': '/tmp/rocket.pdf', 'page': 0}, page_content='これにより、長時間にわたる宇宙飛行中でも安定した飛行軌道を維持し、\\nミッションの成功を確保します。\\nさらに、バイオニック・リアクション・レスポンダーが統合されています')],\n",
       " 'answer': 'いいえ、OraBooster は架空のロケットエンジンです。'}"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "response  = conversational_rag_chain.invoke(\n",
    "   {\"input\": \"それは実在するものですか？\"},\n",
    "   config={\"configurable\": {\"session_id\": session_id}},\n",
    ")\n",
    "\n",
    "response"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "afd970b8-045b-44d5-8761-e3a0185dd50b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'input': 'それはいつ開発されましたか',\n",
       " 'history': [HumanMessage(content='OraBoosterとは何ですか？'),\n",
       "  AIMessage(content='OraBooster は、次世代の宇宙探査を支えるために開発された先進的なロケットエンジンです。高い性能と信頼性、そしてハイパーフォトン・ジャイロスコープによる高精度の姿勢制御を特長とします。'),\n",
       "  HumanMessage(content='それは実在するものですか？'),\n",
       "  AIMessage(content='いいえ、OraBooster は架空のロケットエンジンです。')],\n",
       " 'context': [Document(metadata={'source': '/tmp/rocket.pdf', 'page': 0}, page_content='その高い性能と信頼性は、人類の夢を実現するための力強い支援とな\\nることでしょう'),\n",
       "  Document(metadata={'source': '/tmp/rocket.pdf', 'page': 0}, page_content='また、ハイパーフォトン・ジャイロスコープが搭載されており、極めて高い精度でロケットの姿勢を維\\n持し、目標を追跡します'),\n",
       "  Document(metadata={'source': '/tmp/rocket.pdf', 'page': 0}, page_content='総じて、この新開発のロケットエンジンは、革新的な技術と未来志向の設計によって、宇宙探査の新た\\nな時代を切り開くことでしょう')],\n",
       " 'answer': '開発時期は不明です。'}"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "response  = conversational_rag_chain.invoke(\n",
    "   {\"input\": \"それはいつ開発されましたか\"},\n",
    "   config={\"configurable\": {\"session_id\": session_id}},\n",
    ")\n",
    "\n",
    "response"
   ]
  },
  {
   "cell_type": "raw",
   "id": "3da86d4f-a54e-47b9-be19-e0b496122f29",
   "metadata": {},
   "source": [
    "PostgreSQLでクエリ実行(追加のinputとanswerが追加されていることを確認)\n",
    "\n",
    "postgres=> select * from message_store;\n",
    " id |              session_id              |                                                                                                                             \n",
    "                                                                         message                                                                                         \n",
    "                                                                                                              |          created_at           \n",
    "----+--------------------------------------+-----------------------------------------------------------------------------------------------------------------------------\n",
    "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n",
    "--------------------------------------------------------------------------------------------------------------+-------------------------------\n",
    "  1 | 5e2a952c-59a4-40e1-9355-f688a3dbf27e | {\"data\": {\"id\": null, \"name\": null, \"type\": \"human\", \"content\": \"OraBoosterとは何ですか？\", \"example\": false, \"additional_kw\n",
    "args\": {}, \"response_metadata\": {}}, \"type\": \"human\"}                                                                                                                    \n",
    "                                                                                                              | 2024-07-24 07:01:39.320933+00\n",
    "  2 | 5e2a952c-59a4-40e1-9355-f688a3dbf27e | {\"data\": {\"id\": null, \"name\": null, \"type\": \"ai\", \"content\": \"OraBooster は、次世代の宇宙探査を支える先進的な推進技術を備え\n",
    "ロケットエンジンです。高い性能と信頼性、そしてハイパーフォトン・ジャイロスコープによる極めて高い姿勢制御精度を特長とします。\", \"example\": false, \"tool_calls\": [], \"usage\n",
    "_metadata\": null, \"additional_kwargs\": {}, \"response_metadata\": {}, \"invalid_tool_calls\": []}, \"type\": \"ai\"} | 2024-07-24 07:01:39.320933+00\n",
    "(2 rows)\n",
    "\n",
    "postgres=> \n",
    "postgres=> select * from message_store;\n",
    " id |              session_id              |                                                                                                                             \n",
    "                                                                         message                                                                                         \n",
    "                                                                                                              |          created_at           \n",
    "----+--------------------------------------+-----------------------------------------------------------------------------------------------------------------------------\n",
    "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n",
    "--------------------------------------------------------------------------------------------------------------+-------------------------------\n",
    "  1 | 5e2a952c-59a4-40e1-9355-f688a3dbf27e | {\"data\": {\"id\": null, \"name\": null, \"type\": \"human\", \"content\": \"OraBoosterとは何ですか？\", \"example\": false, \"additional_kw\n",
    "args\": {}, \"response_metadata\": {}}, \"type\": \"human\"}                                                                                                                    \n",
    "                                                                                                              | 2024-07-24 07:01:39.320933+00\n",
    "  2 | 5e2a952c-59a4-40e1-9355-f688a3dbf27e | {\"data\": {\"id\": null, \"name\": null, \"type\": \"ai\", \"content\": \"OraBooster は、次世代の宇宙探査を支える先進的な推進技術を備え\n",
    "ロケットエンジンです。高い性能と信頼性、そしてハイパーフォトン・ジャイロスコープによる極めて高い姿勢制御精度を特長とします。\", \"example\": false, \"tool_calls\": [], \"usage\n",
    "_metadata\": null, \"additional_kwargs\": {}, \"response_metadata\": {}, \"invalid_tool_calls\": []}, \"type\": \"ai\"} | 2024-07-24 07:01:39.320933+00\n",
    "  3 | 5e2a952c-59a4-40e1-9355-f688a3dbf27e | {\"data\": {\"id\": null, \"name\": null, \"type\": \"human\", \"content\": \"それは実在するものですか？\", \"example\": false, \"additional_\n",
    "kwargs\": {}, \"response_metadata\": {}}, \"type\": \"human\"}                                                                                                                  \n",
    "                                                                                                              | 2024-07-24 07:03:40.984928+00\n",
    "  4 | 5e2a952c-59a4-40e1-9355-f688a3dbf27e | {\"data\": {\"id\": null, \"name\": null, \"type\": \"ai\", \"content\": \"いいえ、OraBooster は架空のロケットエンジンです。\", \"example\":\n",
    " false, \"tool_calls\": [], \"usage_metadata\": null, \"additional_kwargs\": {}, \"response_metadata\": {}, \"invalid_tool_calls\": []}, \"type\": \"ai\"}                             \n",
    "                                                                                                              | 2024-07-24 07:03:40.984928+00\n",
    "(4 rows)\n",
    "\n",
    "postgres=> "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "4b0c314e-538e-4700-ab6f-844a738879ff",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'input': 'いつ使われる予定ですか？',\n",
       " 'history': [HumanMessage(content='OraBoosterとは何ですか？'),\n",
       "  AIMessage(content='OraBooster は、次世代の宇宙探査を支えるために開発された先進的なロケットエンジンです。高い性能と信頼性、そしてハイパーフォトン・ジャイロスコープによる高精度の姿勢制御を特長とします。'),\n",
       "  HumanMessage(content='それは実在するものですか？'),\n",
       "  AIMessage(content='いいえ、OraBooster は架空のロケットエンジンです。'),\n",
       "  HumanMessage(content='それはいつ開発されましたか'),\n",
       "  AIMessage(content='開発時期は不明です。')],\n",
       " 'context': [Document(metadata={'source': '/tmp/rocket.pdf', 'page': 0}, page_content='その高い性能と信頼性は、人類の夢を実現するための力強い支援とな\\nることでしょう'),\n",
       "  Document(metadata={'source': '/tmp/rocket.pdf', 'page': 0}, page_content='また、ハイパーフォトン・ジャイロスコープが搭載されており、極めて高い精度でロケットの姿勢を維\\n持し、目標を追跡します'),\n",
       "  Document(metadata={'source': '/tmp/rocket.pdf', 'page': 0}, page_content='総じて、この新開発のロケットエンジンは、革新的な技術と未来志向の設計によって、宇宙探査の新た\\nな時代を切り開くことでしょう')],\n",
       " 'answer': '私は使用予定について知りません。'}"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "response  = conversational_rag_chain.invoke(\n",
    "   {\"input\": \"いつ使われる予定ですか？\"},\n",
    "   config={\"configurable\": {\"session_id\": session_id}},\n",
    ")\n",
    "\n",
    "response"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "0b8f79c5-7a5a-4d75-8818-666996a21058",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'input': '重量はどれくらいですか？',\n",
       " 'history': [HumanMessage(content='OraBoosterとは何ですか？'),\n",
       "  AIMessage(content='OraBooster は、次世代の宇宙探査を支えるために開発された先進的なロケットエンジンです。高い性能と信頼性、そしてハイパーフォトン・ジャイロスコープによる高精度の姿勢制御を特長とします。'),\n",
       "  HumanMessage(content='それは実在するものですか？'),\n",
       "  AIMessage(content='いいえ、OraBooster は架空のロケットエンジンです。'),\n",
       "  HumanMessage(content='それはいつ開発されましたか'),\n",
       "  AIMessage(content='開発時期は不明です。'),\n",
       "  HumanMessage(content='いつ使われる予定ですか？'),\n",
       "  AIMessage(content='私は使用予定について知りません。')],\n",
       " 'context': [Document(metadata={'source': '/tmp/rocket.pdf', 'page': 0}, page_content='その高い性能と信頼性は、人類の夢を実現するための力強い支援とな\\nることでしょう'),\n",
       "  Document(metadata={'source': '/tmp/rocket.pdf', 'page': 0}, page_content='また、ハイパーフォトン・ジャイロスコープが搭載されており、極めて高い精度でロケットの姿勢を維\\n持し、目標を追跡します'),\n",
       "  Document(metadata={'source': '/tmp/rocket.pdf', 'page': 0}, page_content='これにより、長時間にわたる宇宙飛行中でも安定した飛行軌道を維持し、\\nミッションの成功を確保します。\\nさらに、バイオニック・リアクション・レスポンダーが統合されています')],\n",
       " 'answer': '私は重量の情報にアクセスできません。'}"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "response  = conversational_rag_chain.invoke(\n",
    "   {\"input\": \"重量はどれくらいですか？\"},\n",
    "   config={\"configurable\": {\"session_id\": session_id}},\n",
    ")\n",
    "\n",
    "response"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "dd13612e-fced-4bb1-a9a6-7f1e5bd5fc19",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'input': '姿勢制御に使われている技術は何ですか？',\n",
       " 'history': [HumanMessage(content='OraBoosterとは何ですか？'),\n",
       "  AIMessage(content='OraBooster は、次世代の宇宙探査を支えるために開発された先進的なロケットエンジンです。高い性能と信頼性、そしてハイパーフォトン・ジャイロスコープによる高精度の姿勢制御を特長とします。'),\n",
       "  HumanMessage(content='それは実在するものですか？'),\n",
       "  AIMessage(content='いいえ、OraBooster は架空のロケットエンジンです。'),\n",
       "  HumanMessage(content='それはいつ開発されましたか'),\n",
       "  AIMessage(content='開発時期は不明です。'),\n",
       "  HumanMessage(content='いつ使われる予定ですか？'),\n",
       "  AIMessage(content='私は使用予定について知りません。'),\n",
       "  HumanMessage(content='重量はどれくらいですか？'),\n",
       "  AIMessage(content='私は重量の情報にアクセスできません。')],\n",
       " 'context': [Document(metadata={'source': '/tmp/rocket.pdf', 'page': 0}, page_content='また、ハイパーフォトン・ジャイロスコープが搭載されており、極めて高い精度でロケットの姿勢を維\\n持し、目標を追跡します'),\n",
       "  Document(metadata={'source': '/tmp/rocket.pdf', 'page': 0}, page_content='これにより、長時間にわたる宇宙飛行中でも安定した飛行軌道を維持し、\\nミッションの成功を確保します。\\nさらに、バイオニック・リアクション・レスポンダーが統合されています'),\n",
       "  Document(metadata={'source': '/tmp/rocket.pdf', 'page': 0}, page_content='その高い性能と信頼性は、人類の夢を実現するための力強い支援とな\\nることでしょう')],\n",
       " 'answer': '姿勢制御には、ハイパーフォトン・ジャイロスコープ技術が使用されています。'}"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "response  = conversational_rag_chain.invoke(\n",
    "   {\"input\": \"姿勢制御に使われている技術は何ですか？\"},\n",
    "   config={\"configurable\": {\"session_id\": session_id}},\n",
    ")\n",
    "\n",
    "response"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "fd965f33-b58d-4aac-bcd3-f3992bddf1a2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'input': 'それは重要なものですか？',\n",
       " 'history': [HumanMessage(content='OraBoosterとは何ですか？'),\n",
       "  AIMessage(content='OraBooster は、次世代の宇宙探査を支えるために開発された先進的なロケットエンジンです。高い性能と信頼性、そしてハイパーフォトン・ジャイロスコープによる高精度の姿勢制御を特長とします。'),\n",
       "  HumanMessage(content='それは実在するものですか？'),\n",
       "  AIMessage(content='いいえ、OraBooster は架空のロケットエンジンです。'),\n",
       "  HumanMessage(content='それはいつ開発されましたか'),\n",
       "  AIMessage(content='開発時期は不明です。'),\n",
       "  HumanMessage(content='いつ使われる予定ですか？'),\n",
       "  AIMessage(content='私は使用予定について知りません。'),\n",
       "  HumanMessage(content='重量はどれくらいですか？'),\n",
       "  AIMessage(content='私は重量の情報にアクセスできません。'),\n",
       "  HumanMessage(content='姿勢制御に使われている技術は何ですか？'),\n",
       "  AIMessage(content='姿勢制御には、ハイパーフォトン・ジャイロスコープ技術が使用されています。')],\n",
       " 'context': [Document(metadata={'source': '/tmp/rocket.pdf', 'page': 0}, page_content='その高い性能と信頼性は、人類の夢を実現するための力強い支援とな\\nることでしょう'),\n",
       "  Document(metadata={'source': '/tmp/rocket.pdf', 'page': 0}, page_content='また、ハイパーフォトン・ジャイロスコープが搭載されており、極めて高い精度でロケットの姿勢を維\\n持し、目標を追跡します'),\n",
       "  Document(metadata={'source': '/tmp/rocket.pdf', 'page': 0}, page_content='これにより、長時間にわたる宇宙飛行中でも安定した飛行軌道を維持し、\\nミッションの成功を確保します。\\nさらに、バイオニック・リアクション・レスポンダーが統合されています')],\n",
       " 'answer': 'はい、重要です。'}"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "response  = conversational_rag_chain.invoke(\n",
    "   {\"input\": \"それは重要なものですか？\"},\n",
    "   config={\"configurable\": {\"session_id\": session_id}},\n",
    ")\n",
    "\n",
    "response"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "72a27c9e-5bc9-4d7d-bdb4-61caf51ec214",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'input': 'それがないとどうなりますか？',\n",
       " 'history': [HumanMessage(content='OraBoosterとは何ですか？'),\n",
       "  AIMessage(content='OraBooster は、次世代の宇宙探査を支えるために開発された先進的なロケットエンジンです。高い性能と信頼性、そしてハイパーフォトン・ジャイロスコープによる高精度の姿勢制御を特長とします。'),\n",
       "  HumanMessage(content='それは実在するものですか？'),\n",
       "  AIMessage(content='いいえ、OraBooster は架空のロケットエンジンです。'),\n",
       "  HumanMessage(content='それはいつ開発されましたか'),\n",
       "  AIMessage(content='開発時期は不明です。'),\n",
       "  HumanMessage(content='いつ使われる予定ですか？'),\n",
       "  AIMessage(content='私は使用予定について知りません。'),\n",
       "  HumanMessage(content='重量はどれくらいですか？'),\n",
       "  AIMessage(content='私は重量の情報にアクセスできません。'),\n",
       "  HumanMessage(content='姿勢制御に使われている技術は何ですか？'),\n",
       "  AIMessage(content='姿勢制御には、ハイパーフォトン・ジャイロスコープ技術が使用されています。'),\n",
       "  HumanMessage(content='それは重要なものですか？'),\n",
       "  AIMessage(content='はい、重要です。')],\n",
       " 'context': [Document(metadata={'source': '/tmp/rocket.pdf', 'page': 0}, page_content='その高い性能と信頼性は、人類の夢を実現するための力強い支援とな\\nることでしょう'),\n",
       "  Document(metadata={'source': '/tmp/rocket.pdf', 'page': 0}, page_content='これにより、長時間にわたる宇宙飛行中でも安定した飛行軌道を維持し、\\nミッションの成功を確保します。\\nさらに、バイオニック・リアクション・レスポンダーが統合されています'),\n",
       "  Document(metadata={'source': '/tmp/rocket.pdf', 'page': 0}, page_content='また、ハイパーフォトン・ジャイロスコープが搭載されており、極めて高い精度でロケットの姿勢を維\\n持し、目標を追跡します')],\n",
       " 'answer': 'ミッションの成功確保が難しくなります。'}"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "response  = conversational_rag_chain.invoke(\n",
    "   {\"input\": \"それがないとどうなりますか？\"},\n",
    "   config={\"configurable\": {\"session_id\": session_id}},\n",
    ")\n",
    "\n",
    "response"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "95161528-8602-46f7-b93a-15a7b173e542",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "test3",
   "language": "python",
   "name": "test3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
