import streamlit as st
import pdfplumber
import spacy
import pandas as pd
from collections import defaultdict, Counter
import re
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from datetime import datetime
import json
import requests
import time

# Configuration de la page
st.set_page_config(
    page_title="Extraction de Connaissances PDF avec Chat IA",
    page_icon="🧠",
    layout="wide"
)

# Chargement du modèle spaCy
@st.cache_resource
def load_nlp_model():
    try:
        nlp = spacy.load("fr_core_news_sm")
        return nlp
    except OSError:
        st.error("Modèle spaCy français non trouvé. Installez-le avec: python -m spacy download fr_core_news_sm")
        return None

class MistralChatBot:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.mistral.ai/v1/chat/completions"
        self.document_context = ""
        self.chat_history = []
        
    def set_document_context(self, text):
        """Définir le contexte du document pour les réponses"""
        # Limiter le contexte pour éviter de dépasser les limites de tokens
        max_context_length = 8000  # Ajustez selon vos besoins
        if len(text) > max_context_length:
            self.document_context = text[:max_context_length] + "..."
        else:
            self.document_context = text
    
    def ask_question(self, question, model="mistral-small-latest"):
        """Poser une question sur le document"""
        if not self.document_context:
            return "❌ Aucun document chargé. Veuillez d'abord télécharger un PDF."
        
        # Construire le prompt avec le contexte
        system_prompt = f"""Tu es un assistant IA spécialisé dans l'analyse de documents. Tu as accès au contenu d'un document PDF et tu dois répondre aux questions en français en te basant uniquement sur ce contenu.

CONTEXTE DU DOCUMENT:
{self.document_context}

INSTRUCTIONS:
- Réponds uniquement en français
- Base tes réponses sur le contenu du document fourni
- Si l'information n'est pas dans le document, dis-le clairement
- Sois précis et factuel
- Cite des extraits du document quand c'est pertinent"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
        
        # Ajouter l'historique des conversations récentes (dernières 5 interactions)
        for exchange in self.chat_history[-5:]:
            messages.extend([
                {"role": "user", "content": exchange["question"]},
                {"role": "assistant", "content": exchange["answer"]}
            ])
        
        messages.append({"role": "user", "content": question})
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": model,
                "messages": messages,
                "temperature": 0.3,
                "max_tokens": 1000
            }
            
            response = requests.post(self.base_url, headers=headers, json=data,verify=False, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                answer = result["choices"][0]["message"]["content"]
                
                # Sauvegarder dans l'historique
                self.chat_history.append({
                    "question": question,
                    "answer": answer,
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })
                
                return answer
            else:
                return f"❌ Erreur API Mistral: {response.status_code} - {response.text}"
                
        except requests.exceptions.Timeout:
            return "❌ Timeout: La requête a pris trop de temps. Essayez une question plus courte."
        except requests.exceptions.RequestException as e:
            return f"❌ Erreur de connexion: {str(e)}"
        except Exception as e:
            return f"❌ Erreur inattendue: {str(e)}"
    
    def clear_history(self):
        """Effacer l'historique de conversation"""
        self.chat_history = []
    
    def get_history(self):
        """Obtenir l'historique de conversation"""
        return self.chat_history

class KnowledgeExtractor:
    def __init__(self, nlp_model):
        self.nlp = nlp_model
        self.entities = defaultdict(set)
        self.relations = []
        self.concepts = Counter()
        self.temporal_info = []
        self.locations = set()
        self.persons = set()
        self.organizations = set()
        
    def extract_entities(self, text):
        """Extraction d'entités nommées"""
        doc = self.nlp(text)
        
        entities_found = {
            'PERSON': [],
            'ORG': [],
            'LOC': [],
            'DATE': [],
            'MONEY': [],
            'MISC': []
        }
        
        for ent in doc.ents:
            entity_text = ent.text.strip()
            if len(entity_text) > 2:  # Filtrer les entités trop courtes
                if ent.label_ == "PER":
                    entities_found['PERSON'].append(entity_text)
                    self.persons.add(entity_text)
                elif ent.label_ == "ORG":
                    entities_found['ORG'].append(entity_text)
                    self.organizations.add(entity_text)
                elif ent.label_ in ["LOC", "GPE"]:
                    entities_found['LOC'].append(entity_text)
                    self.locations.add(entity_text)
                elif ent.label_ in ["DATE", "TIME"]:
                    entities_found['DATE'].append(entity_text)
                    self.temporal_info.append(entity_text)
                elif ent.label_ == "MONEY":
                    entities_found['MONEY'].append(entity_text)
                else:
                    entities_found['MISC'].append(entity_text)
        
        return entities_found
    
    def extract_key_concepts(self, text):
        """Extraction de concepts clés basée sur les mots-clés importants"""
        doc = self.nlp(text)
        
        # Filtrer les tokens importants
        key_words = []
        for token in doc:
            if (not token.is_stop and 
                not token.is_punct and 
                len(token.text) > 3 and
                token.pos_ in ['NOUN', 'ADJ', 'VERB']):
                key_words.append(token.lemma_.lower())
        
        # Compter les occurrences
        word_freq = Counter(key_words)
        self.concepts.update(word_freq)
        
        return word_freq.most_common(20)
    
    def extract_skills_and_competencies(self, text):
        """Extraction spécifique de compétences (adaptable selon le domaine)"""
        skills_patterns = [
            r'compétence[s]?\s+(?:en\s+)?([^.,:;]+)',
            r'maîtrise\s+(?:de\s+)?([^.,:;]+)',
            r'expérience\s+(?:en\s+|avec\s+)?([^.,:;]+)',
            r'spécialisé\s+en\s+([^.,:;]+)',
            r'expert\s+en\s+([^.,:;]+)',
            r'formation\s+en\s+([^.,:;]+)'
        ]
        
        skills = set()
        for pattern in skills_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                skill = match.group(1).strip()
                if len(skill) > 2:
                    skills.add(skill)
        
        return list(skills)
    
    def find_relationships(self, text):
        """Identification de relations entre entités"""
        doc = self.nlp(text)
        relationships = []
        
        # Relations simples basées sur la proximité
        entities = [(ent.text, ent.label_, ent.start, ent.end) for ent in doc.ents]
        
        for i, (ent1, label1, start1, end1) in enumerate(entities):
            for j, (ent2, label2, start2, end2) in enumerate(entities[i+1:], i+1):
                # Si les entités sont proches (moins de 50 tokens)
                if abs(start1 - start2) < 50:
                    # Extraire le contexte entre les entités
                    context_start = min(start1, start2)
                    context_end = max(end1, end2)
                    context = doc[context_start:context_end].text
                    
                    relationship = {
                        'entity1': ent1,
                        'entity2': ent2,
                        'type1': label1,
                        'type2': label2,
                        'context': context,
                        'relation_strength': 1 / (abs(start1 - start2) + 1)
                    }
                    relationships.append(relationship)
        
        return relationships
    
    def extract_temporal_events(self, text):
        """Extraction d'événements temporels"""
        doc = self.nlp(text)
        events = []
        
        # Patterns pour identifier des événements
        event_patterns = [
            r'(?:depuis|à partir de|en)\s+(\d{4})',
            r'(?:le|du)\s+(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})',
            r'(?:pendant|durant)\s+([^.,:;]+)',
        ]
        
        for pattern in event_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                events.append({
                    'time_expression': match.group(0),
                    'extracted_time': match.group(1),
                    'context': text[max(0, match.start()-50):match.end()+50]
                })
        
        return events

class TechnicalDocumentExtractor:
    """Extracteur spécialisé pour documents techniques"""
    
    def __init__(self, nlp_model):
        self.nlp = nlp_model
        
        # Patterns de reconnaissance technique
        self.tech_patterns = {
            'authors': [
                r'(?:auteur|author|écrit par|written by|développé par|developed by)[:\s]+([^\n\r]+)',
                r'(?:rédigé par|edited by|created by)[:\s]+([^\n\r]+)'
            ],
            'dates': [
                r'(?:date|créé le|created on|modifié le|modified on|version du)[:\s]*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})',
                r'(?:janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s+\d{4}',
                r'\d{4}[-\/]\d{2}[-\/]\d{2}',
                r'\d{2}[-\/]\d{2}[-\/]\d{4}'
            ]
        }
    
    def extract_metadata_from_pdf(self, pdf_file):
        """Extraction des métadonnées du PDF"""
        metadata = {}
        try:
            with pdfplumber.open(pdf_file) as pdf:
                if pdf.metadata:
                    metadata['pdf_title'] = pdf.metadata.get('Title', '')
                    metadata['pdf_author'] = pdf.metadata.get('Author', '')
                    metadata['pdf_creator'] = pdf.metadata.get('Creator', '')
                    metadata['pdf_creation_date'] = pdf.metadata.get('CreationDate', '')
                    metadata['pdf_modification_date'] = pdf.metadata.get('ModDate', '')
                    metadata['pdf_subject'] = pdf.metadata.get('Subject', '')
        except Exception as e:
            st.warning(f"Erreur lors de l'extraction des métadonnées: {e}")
        return metadata
    
    def extract_authors(self, text):
        """Extraction des auteurs"""
        authors = set()
        for pattern in self.tech_patterns['authors']:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                author = match.group(1).strip()
                if len(author) > 3 and len(author) < 100:
                    author = re.sub(r'[^\w\s\-\.]', '', author)
                    authors.add(author)
        return list(authors)
    
    def extract_dates(self, text):
        """Extraction des dates"""
        dates = set()
        for pattern in self.tech_patterns['dates']:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                date_str = match.group(0).strip()
                if len(date_str) > 4:
                    dates.add(date_str)
        return list(dates)
    
    def extract_title(self, text, metadata=None):
        """Extraction du titre du document"""
        potential_titles = []
        
        # D'abord essayer les métadonnées PDF
        if metadata and metadata.get('pdf_title'):
            potential_titles.append(metadata['pdf_title'])
        
        # Ensuite les premières lignes du document
        lines = text.split('\n')[:10]
        for line in lines:
            line = line.strip()
            if len(line) > 10 and len(line) < 200:
                if not re.search(r'[{}()\[\];=<>]', line):
                    potential_titles.append(line)
        
        return potential_titles[0] if potential_titles else "Titre non identifié"
    
    def extract_document_type(self, text, title=""):
        """Identification du type de document technique"""
        doc_types = {
            'Architecture': ['architecture', 'design', 'conception', 'structure'],
            'Spécification': ['spécification', 'specification', 'spec', 'requirements'],
            'Manuel': ['manuel', 'guide', 'documentation', 'tutorial'],
            'API': ['api', 'endpoint', 'rest', 'graphql', 'webservice'],
            'Sécurité': ['sécurité', 'security', 'audit', 'vulnérabilité'],
            'Installation': ['installation', 'deployment', 'setup', 'configuration'],
            'Test': ['test', 'testing', 'qa', 'validation'],
            'Protocole': ['protocole', 'protocol', 'standard', 'rfc'],
            'Rapport': ['rapport', 'report', 'analyse', 'étude']
        }
        
        text_lower = (text + " " + title).lower()
        type_scores = {}
        
        for doc_type, keywords in doc_types.items():
            score = sum(text_lower.count(keyword) for keyword in keywords)
            if score > 0:
                type_scores[doc_type] = score
        
        return max(type_scores, key=type_scores.get) if type_scores else "Document technique générique"
    
    def extract_technologies(self, text):
        """Extraction des technologies mentionnées"""
        technologies = {
            'languages': set(),
            'frameworks': set(),
            'databases': set(),
            'tools': set(),
            'cloud': set(),
            'projet':set()
        }
        
        tech_categories = {
            'languages': [
                r'\b(?:Python|Java|JavaScript|TypeScript|C\+\+|C#|PHP|Ruby|Go|Rust|Swift|Kotlin|Scala)\b',
            ],
            'frameworks': [
                r'\b(?:React|Angular|Spring\s*(?:MVC|Security|Boot)?|Nats|Vue\.?js|Django|Flask|Express\.?js|Laravel|Rails|\.NET|Node\.?js)\b',
            ],
            'databases': [
                r'\b(?:MySQL|Microsoft\s*SQL\s*Server|PostgreSQL|MongoDB|Redis|Elasticsearch|Oracle|SQL\s*Server|SQLite|Cassandra|Neo4j)\b',
            ],
            'tools': [
                r'\b(?:Git|SVN|Maven|Gradle|npm|pip|Docker|Kubernetes|Jenkins|GitLab|GitHub)\b',
            ],
            'cloud': [
                r'\b(?:AWS|Azure|Google\s*Cloud|GCP|Heroku|DigitalOcean|Terraform|Ansible)\b',
            ],
            'projet': [
                r'\b(?:P138|P170|P190|P18)\b',
            ]
        }
        
        for category, patterns in tech_categories.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    technologies[category].add(match.group(0))
        
        return {k: list(v) for k, v in technologies.items()}
    
    def extract_security_info(self, text):
        """Extraction des informations de sécurité et chiffrement"""
        security_info = {
            'encryption_algorithms': set(),
            'protocols': set(),
            'security_measures': set()
        }
        
        patterns = {
            'encryption_algorithms': [
                r'\b(?:AES|RSA|SHA|MD5|DES|3DES|Blowfish|ChaCha20|SHA-?(?:1|256|384|512)|HMAC|bcrypt)\b'
            ],
            'protocols': [
                r'\b(?:SSL|TLS|HTTPS|VPN|IPSec|OAuth|JWT|SAML|OpenID)\b'
            ],
            'security_measures': [
                r'\b(?:firewall|antivirus|2FA|MFA|authentification|chiffrement|encryption)\b'
            ]
        }
        
        for category, pattern_list in patterns.items():
            for pattern in pattern_list:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    security_info[category].add(match.group(0))
        
        return {k: list(v) for k, v in security_info.items()}
    
    def extract_versions_and_standards(self, text):
        """Extraction des versions et standards"""
        versions = set()
        standards = set()
        
        # Versions
        version_patterns = [r'(?:Version|v\.?)\s*(\d+(?:\.\d+)*)', r'\bv(\d+(?:\.\d+)+)\b']
        for pattern in version_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                versions.add(match.group(0))
        
        # Standards
        standard_patterns = [r'\b(?:ISO\s*\d+|IEEE\s*\d+|RFC\s*\d+|W3C|OWASP|NIST|GDPR|HIPAA)\b']
        for pattern in standard_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                standards.add(match.group(0))
        
        return list(versions), list(standards)
    
    def analyze_technical_document(self, text, pdf_file=None):
        """Analyse complète d'un document technique"""
        pdf_metadata = self.extract_metadata_from_pdf(pdf_file) if pdf_file else {}
        
        results = {
            'metadata': pdf_metadata,
            'authors': self.extract_authors(text),
            'dates': self.extract_dates(text),
            'title': self.extract_title(text, pdf_metadata),
            'document_type': self.extract_document_type(text),
            'technologies': self.extract_technologies(text),
            'security': self.extract_security_info(text),
            'versions': [],
            'standards': []
        }
        
        versions, standards = self.extract_versions_and_standards(text)
        results['versions'] = versions
        results['standards'] = standards
        
        return results

@st.cache_data
def extract_text_from_pdf(pdf_file):
    """Extraction du texte du PDF"""
    all_text = ""
    metadata = {"pages": 0}
    
    with pdfplumber.open(pdf_file) as pdf:
        metadata["pages"] = len(pdf.pages)
        
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                all_text += f"\n{text}"
    
    return all_text, metadata

def create_knowledge_graph(relationships):
    """Création d'un graphe de connaissances interactif"""
    if not relationships or len(relationships) == 0:
        return None
        
    G = nx.Graph()
    
    # Ajouter les nœuds et arêtes
    for rel in relationships:
        G.add_edge(rel['entity1'], rel['entity2'], weight=rel['relation_strength'])
    
    if len(G.nodes()) == 0:
        return None
    
    # Positionnement des nœuds
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Préparer les données pour Plotly
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, 
        y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    node_x = []
    node_y = []
    node_text = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
    
    node_trace = go.Scatter(
        x=node_x, 
        y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="middle center",
        marker=dict(
            size=20,
            color='lightblue',
            line=dict(width=2, color='black')
    ))
    
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(text='Graphe de Connaissances', font=dict(size=16)),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[dict(
                text="Graphe des relations entre entités",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor="left", yanchor="bottom",
                font=dict(color="#888", size=12)
        )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
    )
    
    return fig

def create_wordcloud(concepts):
    """Création d'un nuage de mots"""
    if not concepts:
        return None
    
    # Préparer les données pour le nuage de mots
    word_freq = dict(concepts)
    
    if not word_freq:
        return None
    
    # Générer le nuage de mots
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        colormap='viridis',
        max_words=100
    ).generate_from_frequencies(word_freq)
    
    # Créer la figure matplotlib
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    
    return fig

def main():
    st.title("🧠 Extracteur de Connaissances PDF avec SPACY + MISTRAL AI")
    st.markdown("**Système d'extraction de connaissances textes (Spacy) et chat intelligent ( Mistral AI)**")
    
    # Charger le modèle NLP
    nlp = load_nlp_model()
    if not nlp:
        return
    
    # Configuration Mistral AI dans la sidebar
    with st.sidebar:
        st.header("🔑 Configuration Mistral AI")
        mistral_api_key = st.text_input(
            "Clé API Mistral AI",
            value="by35iBAjcIbTagHJUHVvjCLLArbV35B8",
            type="password",
            help="Obtenez votre clé API sur https://console.mistral.ai/"
        )
        
        if mistral_api_key:
            st.success("✅ Clé API configurée")
        else:
            st.warning("⚠️ Clé API Mistral requise pour le chat")
        
        st.markdown("---")
        
        st.header("⚙️ Configuration d'extraction")
        
        # Choix du mode d'extraction
        extraction_mode = st.radio(
            "Mode d'extraction :",
            ["Document Technique"],
            help="Choisissez le type d'analyse adapté à votre document"
        )
        
        if extraction_mode == "📄 Général":
            extract_options = st.multiselect(
                "Types d'extraction :",
                ["Entités nommées", "Concepts clés", "Compétences", "Relations", "Événements temporels"],
                default=["Entités nommées", "Concepts clés", "Relations"]
            )
            
            min_frequency = st.slider("Fréquence minimale des concepts", 1, 10, 2)
            max_entities = st.slider("Nombre max d'entités à afficher", 10, 100, 50)
        
        else:  # Mode technique
            st.markdown("**🔧 Extraction spécialisée pour documents techniques**")
            tech_extract_options = st.multiselect(
                "Éléments à extraire :",
                [
                    "📝 Métadonnées (auteur, titre, dates)",
                    "💻 Technologies utilisées", 
                    "🔒 Sécurité et chiffrement",
                    "📋 Standards et protocoles",
                    "📊 Versions "
                ],
                default=[
                    "📝 Métadonnées (auteur, titre, dates)",
                    "💻 Technologies utilisées", 
                    "🔒 Sécurité et chiffrement"
                ]
            )
    
    # Initialiser le chatbot Mistral
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
        st.session_state.document_loaded = False
        st.session_state.document_text = ""
        st.session_state.extraction_results = None
    
    if mistral_api_key and st.session_state.chatbot is None:
        st.session_state.chatbot = MistralChatBot(mistral_api_key)
    
    # Zone principale avec onglets
    main_tabs = st.tabs(["📄 Upload & Extraction", "💬 Chat avec le Document"])
    
    with main_tabs[0]:
        # Zone d'upload et d'extraction
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("📄 Upload de document")
            uploaded_file = st.file_uploader(
                "Téléchargez votre PDF",
                type=["pdf"]
            )
            
            if uploaded_file:
                # Extraction du texte
                with st.spinner("📖 Extraction du texte..."):
                    text, metadata = extract_text_from_pdf(uploaded_file)
                
                st.success(f"✅ Texte extrait ({len(text)} caractères)")
                st.info(f"📄 {metadata['pages']} pages")
                
                # Sauvegarder le texte pour le chat
                st.session_state.document_text = text
                st.session_state.document_loaded = True
                
                # Configurer le contexte du chatbot
                if st.session_state.chatbot:
                    st.session_state.chatbot.set_document_context(text)
                
                # Prévisualisation du texte
                with st.expander("👀 Aperçu du texte"):
                    st.text_area("Contenu", text[:1000] + "..." if len(text) > 1000 else text, height=200)
        
        with col2:
            if extraction_mode == "📄 Général":
                st.subheader("🔍 Connaissances extraites")
            else:
                st.subheader("🔧 Analyse technique")
        
        if uploaded_file and st.button("🚀 Lancer l'extraction", type="primary"):
            
            if extraction_mode == "📄 Général":
                extractor = KnowledgeExtractor(nlp)
                
                with st.spinner("🧠 Extraction des connaissances en cours..."):
                    results = {}
                    
                    if "Entités nommées" in extract_options:
                        results['entities'] = extractor.extract_entities(text)
                    
                    if "Concepts clés" in extract_options:
                        results['concepts'] = extractor.extract_key_concepts(text)
                    
                    if "Compétences" in extract_options:
                        results['skills'] = extractor.extract_skills_and_competencies(text)
                    
                    if "Relations" in extract_options:
                        results['relationships'] = extractor.find_relationships(text)
                    
                    if "Événements temporels" in extract_options:
                        results['events'] = extractor.extract
                        results['events'] = extractor.extract_temporal_events(text)
                    
                    st.session_state.extraction_results = results
                
                # Affichage des résultats d'extraction générale
                if st.session_state.extraction_results:
                    results = st.session_state.extraction_results
                    
                    # Onglets pour les différents types de résultats
                    result_tabs = st.tabs(["🏷️ Entités", "💡 Concepts", "🎯 Compétences", "🔗 Relations", "⏰ Événements", "📊 Visualisations"])
                    
                    with result_tabs[0]:  # Entités
                        if 'entities' in results:
                            st.subheader("🏷️ Entités nommées")
                            entities = results['entities']
                            
                            for entity_type, entity_list in entities.items():
                                if entity_list:
                                    st.write(f"**{entity_type}:** {', '.join(set(entity_list[:max_entities]))}")
                    
                    with result_tabs[1]:  # Concepts
                        if 'concepts' in results:
                            st.subheader("💡 Concepts clés")
                            concepts = results['concepts']
                            
                            # Filtrer par fréquence minimale
                            filtered_concepts = [(word, freq) for word, freq in concepts if freq >= min_frequency]
                            
                            if filtered_concepts:
                                concepts_df = pd.DataFrame(filtered_concepts, columns=['Concept', 'Fréquence'])
                                st.dataframe(concepts_df)
                                
                                # Graphique des concepts
                                fig_concepts = px.bar(
                                    concepts_df.head(20), 
                                    x='Fréquence', 
                                    y='Concept', 
                                    orientation='h',
                                    title="Top 20 des concepts les plus fréquents"
                                )
                                st.plotly_chart(fig_concepts, use_container_width=True)
                    
                    with result_tabs[2]:  # Compétences
                        if 'skills' in results:
                            st.subheader("🎯 Compétences identifiées")
                            skills = results['skills']
                            
                            if skills:
                                for skill in skills[:20]:
                                    st.write(f"• {skill}")
                            else:
                                st.info("Aucune compétence spécifique identifiée avec les patterns utilisés.")
                    
                    with result_tabs[3]:  # Relations
                        if 'relationships' in results:
                            st.subheader("🔗 Relations entre entités")
                            relationships = results['relationships']
                            
                            if relationships:
                                st.write(f"**{len(relationships)} relations trouvées**")
                                
                                # Tableau des relations
                                rel_data = []
                                for rel in relationships[:20]:
                                    rel_data.append({
                                        'Entité 1': rel['entity1'],
                                        'Entité 2': rel['entity2'],
                                        'Contexte': rel['context'][:100] + "..." if len(rel['context']) > 100 else rel['context']
                                    })
                                
                                if rel_data:
                                    rel_df = pd.DataFrame(rel_data)
                                    st.dataframe(rel_df)
                            else:
                                st.info("Aucune relation forte identifiée entre les entités.")
                    
                    with result_tabs[4]:  # Événements
                        if 'events' in results:
                            st.subheader("⏰ Événements temporels")
                            events = results['events']
                            
                            if events:
                                for event in events[:10]:
                                    st.write(f"**{event['time_expression']}**")
                                    st.write(f"Contexte: {event['context']}")
                                    st.write("---")
                            else:
                                st.info("Aucun événement temporel identifié.")
                    
                    with result_tabs[5]:  # Visualisations
                        st.subheader("📊 Visualisations")
                        
                        col_viz1, col_viz2 = st.columns(2)
                        
                        with col_viz1:
                            # Nuage de mots
                            if 'concepts' in results and results['concepts']:
                                try:
                                    fig_wordcloud = create_wordcloud(results['concepts'])
                                    if fig_wordcloud:
                                        st.pyplot(fig_wordcloud)
                                except Exception as e:
                                    st.error(f"Erreur lors de la création du nuage de mots: {e}")
                        
                        with col_viz2:
                            # Graphe de connaissances
                            if 'relationships' in results and results['relationships']:
                                try:
                                    fig_graph = create_knowledge_graph(results['relationships'])
                                    if fig_graph:
                                        st.plotly_chart(fig_graph, use_container_width=True)
                                except Exception as e:
                                    st.error(f"Erreur lors de la création du graphe: {e}")
            
            else:  # Mode technique
                tech_extractor = TechnicalDocumentExtractor(nlp)
                
                with st.spinner("🔧 Analyse technique en cours..."):
                    tech_results = tech_extractor.analyze_technical_document(text, uploaded_file)
                    st.session_state.extraction_results = tech_results
                
                # Affichage des résultats techniques
                if st.session_state.extraction_results:
                    tech_results = st.session_state.extraction_results
                    
                    # Informations générales
                    st.subheader("📋 Informations générales")
                    col_info1, col_info2 = st.columns(2)
                    
                    with col_info1:
                        st.write(f"**Titre:** {tech_results['title']}")
                        st.write(f"**Type de document:** {tech_results['document_type']}")
                        
                        if tech_results['authors']:
                            st.write(f"**Auteurs:** {', '.join(tech_results['authors'])}")
                    
                    with col_info2:
                        if tech_results['dates']:
                            st.write(f"**Dates:** {', '.join(tech_results['dates'])}")
                        
                        if tech_results['metadata']:
                            metadata = tech_results['metadata']
                            if metadata.get('pdf_creation_date'):
                                st.write(f"**Date de création PDF:** {metadata['pdf_creation_date']}")
                    
                    # Onglets techniques
                    tech_tabs = st.tabs(["💻 Technologies", "🔒 Sécurité", "📋 Standards", "📊 Métadonnées"])
                    
                    with tech_tabs[0]:  # Technologies
                        st.subheader("💻 Technologies identifiées")
                        technologies = tech_results['technologies']
                        
                        for category, tech_list in technologies.items():
                            if tech_list:
                                st.write(f"**{category.title()}:** {', '.join(tech_list)}")
                    
                    with tech_tabs[1]:  # Sécurité
                        st.subheader("🔒 Éléments de sécurité")
                        security = tech_results['security']
                        
                        for category, items in security.items():
                            if items:
                                st.write(f"**{category.replace('_', ' ').title()}:** {', '.join(items)}")
                    
                    with tech_tabs[2]:  # Standards
                        st.subheader("📋 Standards et versions")
                        
                        if tech_results['standards']:
                            st.write(f"**Standards:** {', '.join(tech_results['standards'])}")
                        
                        if tech_results['versions']:
                            st.write(f"**Versions:** {', '.join(tech_results['versions'])}")
                    
                    with tech_tabs[3]:  # Métadonnées
                        st.subheader("📊 Métadonnées PDF")
                        metadata = tech_results['metadata']
                        
                        if metadata:
                            for key, value in metadata.items():
                                if value:
                                    st.write(f"**{key.replace('pdf_', '').replace('_', ' ').title()}:** {value}")
    
    with main_tabs[1]:
        # Interface de chat
        st.subheader("💬 Chat avec votre document")
        
        if not st.session_state.document_loaded:
            st.warning("⚠️ Veuillez d'abord télécharger et analyser un document dans l'onglet 'Upload & Extraction'")
        elif not mistral_api_key:
            st.warning("⚠️ Veuillez configurer votre clé API Mistral AI dans la sidebar")
        else:
            # Interface de chat
            st.info("💡 Posez des questions sur votre document. L'IA se base uniquement sur le contenu analysé.")
            
            # Affichage de l'historique de conversation
            if st.session_state.chatbot and st.session_state.chatbot.get_history():
                st.subheader("📜 Historique de conversation")
                
                for i, exchange in enumerate(st.session_state.chatbot.get_history()):
                    col1, col2 = st.columns([1, 1])

                    with col2:
                        st.markdown(f"""
                        <div style='background-color:#e6f7ff; padding:10px; border-radius:10px; text-align:right; margin-top:5px; margin-bottom:0px;'>
                            <strong style="color:#0b5394;">Vous :</strong><br>{exchange['question']}
                        </div>
                        """, unsafe_allow_html=True)

                    with col1:
                        st.markdown(f"""
                        <div style='background-color:#f0f0f0; padding:10px; border-radius:10px; text-align:left; margin-top:40px; margin-bottom:20px;'>
                            <strong style="color:#6b6b6b;">Assistant :</strong><br>{exchange['answer']}
                        </div>
                        """, unsafe_allow_html=True)


            
            # Zone de saisie de question
            question = st.text_input(
                "Posez votre question :",
                placeholder="Ex: Quels sont les points clés de ce document ?",
                key="chat_input"
            )
            
            col_chat1, col_chat2, col_chat3 = st.columns([2, 1, 1])
            
            with col_chat1:
                ask_button = st.button("🚀 Poser la question", type="primary")
            
            with col_chat2:
                if st.button("🗑️ Effacer l'historique"):
                    if st.session_state.chatbot:
                        st.session_state.chatbot.clear_history()
                        st.success("Historique effacé !")
                        st.rerun()
            
            with col_chat3:
                # Bouton pour télécharger l'historique
                if st.session_state.chatbot and st.session_state.chatbot.get_history():
                    history_json = json.dumps(st.session_state.chatbot.get_history(), indent=2, ensure_ascii=False)
                    st.download_button(
                        "💾 Export historique",
                        data=history_json,
                        file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
            
            # Questions suggérées
            with st.expander("💡 Questions suggérées"):
                suggested_questions = [
                    "Résume-moi les points principaux de ce document",
                    "Quelles sont les technologies mentionnées ?",
                    "Qui sont les auteurs ou personnes citées ?",
                    "Quelles sont les dates importantes ?",
                    "Y a-t-il des informations sur la sécurité ?",
                    "Quels sont les concepts clés abordés ?"
                ]
                
                for suggestion in suggested_questions:
                    if st.button(f"💬 {suggestion}", key=f"suggest_{suggestion}"):
                        question = suggestion
                        ask_button = True
            
            # Traitement de la question
            if ask_button and question and st.session_state.chatbot:
                with st.spinner("🤔 L'IA analyse votre question..."):
                    response = st.session_state.chatbot.ask_question(question)
                
                # Affichage de la réponse
                st.subheader("🤖 Réponse de l'assistant")
                st.write(response)
                
                # Auto-scroll vers le bas
                st.rerun()

    # Section d'export des résultats
    if st.session_state.extraction_results:
        st.markdown("---")
        st.subheader("💾 Export des résultats")
        
        col_export1, col_export2 = st.columns(2)
        
        with col_export1:
            # Export JSON
            results_json = json.dumps(st.session_state.extraction_results, indent=2, ensure_ascii=False, default=str)
            st.download_button(
                "📄 Télécharger les résultats (JSON)",
                data=results_json,
                file_name=f"extraction_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col_export2:
            # Export CSV pour les concepts si disponibles
            if 'concepts' in st.session_state.extraction_results:
                concepts_df = pd.DataFrame(
                    st.session_state.extraction_results['concepts'], 
                    columns=['Concept', 'Fréquence']
                )
                csv = concepts_df.to_csv(index=False)
                st.download_button(
                    "📊 Télécharger les concepts (CSV)",
                    data=csv,
                    file_name=f"concepts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    main()

