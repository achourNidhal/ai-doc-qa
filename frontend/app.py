import streamlit as st
import os
from dotenv import load_dotenv

# Imports LangChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.llms import Ollama
from langchain_classic.chains import RetrievalQA

# Charger les variables d'environnement (utile pour la clé API OpenAI)
load_dotenv()

# --- Fonctions de la Logique RAG ---

def process_document(temp_file_path):
    """
    Charge, découpe et crée une base de données vectorielle pour un fichier PDF.
    Retourne le vectorstore.
    """
    # 1. Charger le document
    loader = PyPDFLoader(temp_file_path)
    documents = loader.load()

    # 2. Découper le document
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    # 3. Créer les embeddings et le vectorstore
    # Assurez-vous que la clé API OpenAI est disponible
    if not os.getenv("OPENAI_API_KEY"):
        st.error("La clé API OpenAI n'est pas définie. Veuillez la définir dans vos variables d'environnement.")
        return None
        
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vectorstore = Chroma.from_documents(documents=texts, embedding=embeddings)
    
    return vectorstore

def get_qa_chain(vectorstore, llm_choice):
    """
    Crée et retourne une chaîne de Question-Answering.
    """
    # Sélection du LLM
    if llm_choice == "OpenAI (gpt-4)":
        llm = ChatOpenAI(temperature=0, model_name="gpt-4")
    elif llm_choice == "Ollama (local)":
        try:
            llm = Ollama(model="llama3")
            # Petit test pour voir si le serveur Ollama est bien lancé
            llm.invoke("test")
        except Exception as e:
            st.error("Impossible de se connecter à Ollama. Assurez-vous que le serveur Ollama est bien lancé.")
            return None
    else:
        st.error("Choix de LLM invalide.")
        return None

    # Création de la chaîne
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
    )
    return qa_chain

# --- Interface Streamlit ---

st.set_page_config(page_title="AI Document Q&A", layout="wide")
st.title("⭐ AI Document Q&A")
st.markdown("Uploadez un document PDF, et posez-lui des questions en utilisant OpenAI ou un modèle local via Ollama.")

# Initialiser l'état de la session si ce n'est pas déjà fait
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# Barre latérale pour les options
with st.sidebar:
    st.header("Configuration")
    
    # Uploader de fichier
    uploaded_file = st.file_uploader("1. Uploadez votre document PDF", type="pdf")

    # Bouton pour lancer le traitement
    if uploaded_file is not None:
        if st.button("Traiter le document"):
            with st.spinner("Traitement du document en cours... (cela peut prendre un moment)"):
                # Sauvegarder le fichier temporairement pour que PyPDFLoader puisse le lire
                temp_dir = "temp"
                if not os.path.exists(temp_dir):
                    os.makedirs(temp_dir)
                temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                # Traiter le document et stocker le vectorstore dans la session
                st.session_state.vectorstore = process_document(temp_file_path)
                
                # Nettoyer le fichier temporaire
                os.remove(temp_file_path)
                
                if st.session_state.vectorstore:
                    st.success("Document traité avec succès ! Vous pouvez maintenant poser des questions.")
                else:
                    st.error("Le traitement du document a échoué.")

# Interface principale pour la Q&A
if st.session_state.vectorstore:
    st.header("❓ Posez vos questions")

    # Sélection du modèle LLM
    llm_choice = st.selectbox(
        "2. Choisissez le modèle à utiliser",
        ("OpenAI (gpt-4)", "Ollama (local)")
    )
    
    # Champ de saisie pour la question
    question = st.text_input("3. Entrez votre question ici")

    if question:
        with st.spinner("Recherche de la réponse..."):
            qa_chain = get_qa_chain(st.session_state.vectorstore, llm_choice)
            if qa_chain:
                response = qa_chain.invoke({"query": question})
                st.markdown("### Réponse")
                st.write(response.get("result", "Aucune réponse n'a pu être générée."))
else:
    st.info("Veuillez uploader et traiter un document pour commencer.")
