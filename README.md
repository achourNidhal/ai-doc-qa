# ⭐ AI Document Q&A (RAG) - Solution Multi-Modèles

[![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-LangChain-green)](https://www.langchain.com/)
[![Frontend](https://img.shields.io/badge/Frontend-Streamlit-red)](https://streamlit.io/)

Ce projet est une application de **Génération Augmentée par Récupération (RAG)** permettant de discuter avec des documents PDF. Il offre une flexibilité unique en permettant de basculer entre la puissance du cloud (**OpenAI GPT-4**) et la confidentialité du local (**Ollama/Llama 3**).

---

## 🏗️ Architecture du Système

L'application repose sur une architecture modulaire qui sépare le stockage des données de la logique de raisonnement.

### Schéma de Fonctionnement (Workflow)

1. **Ingestion** : Lecture du PDF et découpage en segments (chunks).
2. **Indexation** : Conversion des segments en vecteurs (Embeddings) via OpenAI.
3. **Stockage** : Sauvegarde dans une base de données vectorielle **ChromaDB**.
4. **Récupération** : Recherche sémantique des segments les plus pertinents lors d'une question.
5. **Génération** : Envoi du contexte + de la question au LLM choisi (OpenAI ou Ollama).

---

## 🛠️ Pile Technologique (Stack)

| Composant                       | Technologie                                |
| ------------------------------- | ------------------------------------------ |
| **Orchestration IA**            | [LangChain](https://python.langchain.com/) |
| **Interface Utilisateur**       | [Streamlit](https://streamlit.io/)         |
| **Base de Données Vectorielle** | [ChromaDB](https://www.trychroma.com/)     |
| **Embeddings**                  | OpenAI (text-embedding-3-large)            |
| **LLM Cloud**                   | OpenAI (GPT-4)                             |
| **LLM Local**                   | Ollama (Llama 2 / Mistral)                 |

---

## 🧠 Comment le RAG fonctionne-t-il ici ?

Contrairement à un chatbot classique qui "devine" les réponses, cette application utilise un processus en deux temps :

1. **La Phase de Recherche (Retrieval)** : Quand vous posez une question, le système ne l'envoie pas tout de suite à l'IA. Il cherche d'abord dans le document les passages qui parlent du sujet.
2. **La Phase de Réponse (Augmentation/Generation)** : Le système donne ces passages à l'IA et lui dit : _"Utilise ces extraits pour répondre précisément à l'utilisateur"_.
   - **Avantage** : Cela élimine les hallucinations et permet de citer des documents privés non connus par l'IA lors de son entraînement.

---

## ⚙️ Installation et Configuration

### 1. Prérequis

- **Python 3.11 ou 3.12** (Important : Ne pas utiliser la version 3.14 qui est expérimentale).
- **Ollama** installé et actif ([Télécharger Ollama](https://ollama.com/)).
- Une clé API OpenAI valide.

### 2. Installation pas à pas

```bash
# Cloner le dépôt
git clone <votre-url-repo>
cd ai-doc-qa-rag

# Créer l'environnement virtuel
python -m venv venv

# Activer l'environnement
# Sur Windows :
.\venv\Scripts\activate
# Sur Mac/Linux :
source venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt
```

### 3. Configuration des variables d'environnement

Créez un fichier .env à la racine du projet et ajoutez votre clé :

```env
OPENAI_API_KEY=votre_cle_api_openai_ici
```

## 🚀 Lancement de l'Application

Pour démarrer l'interface utilisateur :

```bash
streamlit run app.py
```

Upload : Glissez un PDF dans la barre latérale.

Process : Cliquez sur "Traiter le document" pour créer l'index vectoriel.

Chat : Posez vos questions et changez de modèle (OpenAI vs Ollama) à la volée.

📂 Structure du Projet

```text
ai-doc-qa-rag/
├── app.py # Application principale (Frontend + Logique RAG)
├── requirements.txt # Dépendances Python
├── .env # Variables d'environnement (à créer)
└── README.md # Ce fichier
```

## 🔮 Évolutions Futures (Roadmap)

Pour démontrer la scalabilité du projet, voici les prochaines fonctionnalités envisagées :

Support Multi-Documents : Permettre l'upload et l'interrogation simultanée de plusieurs PDF, TXT ou DOCX.

Historique de Conversation (Memory) : Implémenter ConversationBufferMemory de LangChain pour que l'IA se souvienne du contexte des questions précédentes.

Embeddings 100% Locaux : Remplacer les embeddings OpenAI par un modèle local (ex: HuggingFace all-MiniLM-L6-v2) pour une solution totalement déconnectée d'internet (Air-gapped RAG).

Affichage des Sources : Indiquer à l'utilisateur depuis quelle page ou paragraphe précis du document la réponse a été extraite.

## ⚠️ Notes & Dépannage

Erreur d'importation (ModuleNotFoundError) : Vérifiez que votre environnement virtuel (venv) est bien activé avant de lancer l'application ou d'installer les dépendances.

Modèle Ollama non trouvé : Assurez-vous d'avoir téléchargé le modèle local avec ollama pull llama3 avant de le sélectionner dans l'interface.
