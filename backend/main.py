import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil

# Imports LangChain MIS À JOUR
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.llms import Ollama
from langchain_classic.chains import RetrievalQA



# --- Configuration de l'Application FastAPI ---
app = FastAPI(
    title="API pour Q&A sur Documents",
    description="Uploadez un PDF et posez des questions dessus en utilisant OpenAI ou Ollama.",
)

# Configuration CORS pour autoriser les requêtes du frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Attention : En production, restreindre à l'URL de votre frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Variables Globales pour la Logique RAG ---
# Ces variables conserveront l'état entre les appels API (stockage en mémoire)
vectorstore = None
DOC_PATH = "temp_doc.pdf"

# --- Modèles d'API (Pydantic) ---
class AskRequest(BaseModel):
    question: str
    llm_choice: str # "openai" ou "ollama"

# --- Endpoints de l'API ---

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Endpoint pour uploader un fichier PDF. Le fichier est ensuite traité pour la RAG.
    """
    global vectorstore

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Seuls les fichiers PDF sont autorisés.")

    # 1. Sauvegarder le fichier PDF temporairement sur le serveur
    try:
        with open(DOC_PATH, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la sauvegarde du fichier: {e}")
    finally:
        file.file.close()

    try:
        # 2. Ingestion du document avec LangChain
        loader = PyPDFLoader(DOC_PATH)
        documents = loader.load()
        
        if not documents:
            raise HTTPException(status_code=500, detail="Impossible d'extraire le contenu du PDF.")

        # 3. Découpage en chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)

        # 4. Génération des embeddings et stockage dans ChromaDB
        # On utilise OpenAI pour les embeddings comme demandé, car ils sont performants
        if not os.getenv("OPENAI_API_KEY"):
            raise HTTPException(status_code=500, detail="La clé API OpenAI n'est pas configurée.")
        
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vectorstore = Chroma.from_documents(documents=texts, embedding=embeddings)

        return {"message": f"Fichier '{file.filename}' uploadé et traité avec succès."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du traitement du document: {e}")


@app.post("/ask")
async def ask_question(request: AskRequest):
    """
    Endpoint pour poser une question. Utilise le document déjà uploadé.
    """
    global vectorstore

    if vectorstore is None:
        raise HTTPException(status_code=400, detail="Veuillez d'abord uploader un document.")

    # 1. Sélection du LLM basé sur le choix de l'utilisateur
    if request.llm_choice == "openai":
        llm = ChatOpenAI(temperature=0, model_name="gpt-4")
    elif request.llm_choice == "ollama":
        llm = Ollama(model="llama2") # Assurez-vous que le modèle "llama2" est disponible
    else:
        raise HTTPException(status_code=400, detail="Choix de LLM invalide. Choisissez 'openai' ou 'ollama'.")

    # 2. Création de la chaîne de Question-Answering
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=False, # On ne retourne que la réponse
    )

    # 3. Exécution de la chaîne et renvoi de la réponse
    try:
        response = qa_chain.invoke({"query": request.question})
        return {"answer": response.get("result", "Aucune réponse trouvée.")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'interrogation du LLM: {e}")

# Point d'entrée pour lancer le serveur avec Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
