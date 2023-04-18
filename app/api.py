from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datasets import load_dataset
from haystack import Document
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import BM25Retriever
from haystack.nodes import FARMReader
from haystack.pipelines import ExtractiveQAPipeline

app = FastAPI(title="Question Answer Generation System", description="API for rating Question Answer", version="0.1.0")

origins = [
    "http://localhost:3000",
    "localhost:3000"
]

app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
)

document_store = InMemoryDocumentStore(use_bm25=True)

# load the squad dataset
data = load_dataset("squad_v2", split="train")
# convert to a pandas dataframe
df = data.to_pandas()
# select only title and context column
df = df[["title", "context"]]
# drop rows containing duplicate context passages
df = df.drop_duplicates(subset="context")
df.head()

#df.to_csv('squad_Data.csv')

docs = []
for d in df.iterrows():
    d = d[1]
    # create haystack document object with text content and doc metadata
    doc = Document(
        content=d["context"],
        meta={
            "title": d["title"],
            'context': d['context']
        }
    )
    docs.append(doc)

document_store.write_documents(docs)

retriever = BM25Retriever(document_store=document_store)

reader = FARMReader(
    #model_name_or_path='deepset/deberta-v3-large-squad2', 
    model_name_or_path='deepset/roberta-base-squad2',
    use_gpu=False
)

pipe = ExtractiveQAPipeline(reader, retriever)


@app.post("/question-answer", tags=["Evaluate"])
async def question_answer_generation(query:str):
    prediction=pipe.run(query=query, params={"Retriever": {"top_k": 10}})
    document_dicts = [doc.to_dict() for doc in prediction["answers"]][0]
    #print([doc.to_dict() for doc in prediction["score"]][0])
    data=document_dicts['answer'] #[d['answer'] for d in answer['answers']]
    score=document_dicts['score']
    if score>=0.40:
        answer=data
    else:
        answer="I'm sorry, but I don't have enough information to provide a meaningful response to your question."
    return {"answer": answer,"score":score}

