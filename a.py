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

# Configuration de la page
st.set_page_config(
    page_title="Extraction de Connaissances PDF",
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
            'cloud': set()
        }
        
        tech_categories = {
            'languages': [
                r'\b(?:Python|Java|JavaScript|TypeScript|C\+\+|C#|PHP|Ruby|Go|Rust|Swift|Kotlin|Scala)\b',
            ],
            'frameworks': [
                r'\b(?:React|Angular|Spring MVC|Spring Security|Nats|Vue\.?js|Django|Flask|Spring|Express\.?js|Laravel|Rails|\.NET|Node\.?js)\b',
            ],
            'databases': [
                r'\b(?:MySQL|Microsoft SQL Server|PostgreSQL|MongoDB|Redis|Elasticsearch|Oracle|SQL\s*Server|SQLite|Cassandra|Neo4j)\b',
            ],
            'tools': [
                r'\b(?:Git|SVN|Maven|Gradle|npm|pip|Docker|Kubernetes|Jenkins|GitLab|GitHub)\b',
            ],
            'cloud': [
                r'\b(?:AWS|Azure|Google\s*Cloud|GCP|Heroku|DigitalOcean|Terraform|Ansible)\b',
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
        version_patterns = [r'(?:version|v\.?)\s*(\d+(?:\.\d+)*)', r'\bv(\d+(?:\.\d+)+)\b']
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

def main():
    st.title("🧠 Extracteur de Connaissances PDF")
    st.markdown("**Système d'extraction automatique de connaissances à partir de documents PDF**")
    
    # Charger le modèle NLP
    nlp = load_nlp_model()
    if not nlp:
        return
    
    # Sidebar pour configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Choix du mode d'extraction
        extraction_mode = st.radio(
            "Mode d'extraction :",
            ["📄 Général", "🔧 Document Technique"],
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
                    "📊 Versions et releases"
                ],
                default=[
                    "📝 Métadonnées (auteur, titre, dates)",
                    "💻 Technologies utilisées", 
                    "🔒 Sécurité et chiffrement"
                ]
            )
        
        st.markdown("---")
        if extraction_mode == "📄 Général":
            st.markdown("**💡 Types d'extraction :**")
            st.markdown("• **Entités** : Personnes, lieux, organisations")
            st.markdown("• **Concepts** : Mots-clés importants")
            st.markdown("• **Relations** : Liens entre entités")
        else:
            st.markdown("**🔧 Extraction technique :**")
            st.markdown("• **Métadonnées** : Auteur, titre, dates")
            st.markdown("• **Technologies** : Langages, frameworks")
            st.markdown("• **Sécurité** : Chiffrement, protocoles")
    
    # Zone principale
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
                    results['events'] = extractor.extract_temporal_events(text)
            
            # Affichage des résultats
            st.success("✅ Extraction terminée !")
            
            tabs = st.tabs(["📊 Résumé", "🏷️ Entités", "🔤 Concepts", "🔗 Relations", "⏱️ Événements"])
            
            with tabs[0]:  # Résumé
                st.subheader("📊 Statistiques d'extraction")
                
                col1, col2, col3 = st.columns(3)
                if "Entités nommées" in extract_options:
                    total_entities = sum(len(v) for v in results['entities'].values())
                    col1.metric("Entités trouvées", total_entities)
                
                if "Concepts clés" in extract_options:
                    col2.metric("Concepts uniques", len(results['concepts']))
                
                if "Relations" in extract_options:
                    col3.metric("Relations identifiées", len(results['relationships']))
                
                if "Compétences" in extract_options and results.get('skills'):
                    st.subheader("💼 Compétences identifiées")
                    for skill in results['skills']:
                        st.write(f"- {skill}")
            
            with tabs[1]:  # Entités
                if "Entités nommées" in extract_options:
                    st.subheader("🏷️ Entités nommées")
                    
                    entity_types = st.multiselect(
                        "Filtrer par type d'entité",
                        list(results['entities'].keys()),
                        default=list(results['entities'].keys()))
                    
                    for entity_type in entity_types:
                        if results['entities'][entity_type]:
                            st.write(f"**{entity_type}**")
                            entities = results['entities'][entity_type][:max_entities]
                            for entity in entities:
                                st.write(f"- {entity}")
            
            with tabs[2]:  # Concepts
                if "Concepts clés" in extract_options:
                    st.subheader("🔤 Concepts clés")
                    
                    # Filtrer par fréquence
                    filtered_concepts = [(word, count) for word, count in results['concepts'] if count >= min_frequency]
                    
                    if filtered_concepts:
                        df_concepts = pd.DataFrame(filtered_concepts, columns=['Concept', 'Fréquence'])
                        st.dataframe(df_concepts)
                        
                        # Nuage de mots
                        st.subheader("☁️ Nuage de mots")
                        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dict(filtered_concepts))
                        plt.figure(figsize=(10, 5))
                        plt.imshow(wordcloud, interpolation='bilinear')
                        plt.axis("off")
                        st.pyplot(plt)
                    else:
                        st.warning("Aucun concept ne correspond aux critères de fréquence")
            
            with tabs[3]:  # Relations
                if "Relations" in extract_options and results.get('relationships'):
                    st.subheader("🔗 Relations entre entités")
                    
                    # Graphe de connaissances
                    st.subheader("🧠 Graphe de connaissances")
                    fig = create_knowledge_graph(results['relationships'])
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Pas assez de relations pour créer un graphe")
                    
                    # Liste des relations
                    st.subheader("📋 Liste des relations")
                    for rel in results['relationships'][:20]:
                        st.write(f"- **{rel['entity1']}** → **{rel['entity2']}**")
                        with st.expander("Contexte"):
                            st.write(rel['context'])
            
            with tabs[4]:  # Événements
                if "Événements temporels" in extract_options and results.get('events'):
                    st.subheader("⏱️ Événements temporels")
                    
                    for event in results['events'][:20]:
                        st.write(f"- **{event['time_expression']}**")
                        st.write(f"  Extrait: {event['extracted_time']}")
                        with st.expander("Contexte"):
                            st.write(event['context'])
        
        else:  # Mode technique
            tech_extractor = TechnicalDocumentExtractor(nlp)
            
            with st.spinner("🔧 Analyse technique en cours..."):
                tech_results = tech_extractor.analyze_technical_document(text, uploaded_file)
            
            st.success("✅ Analyse technique terminée !")
            
            # Affichage des résultats techniques
            tech_tabs = st.tabs(["📋 Résumé", "📝 Métadonnées", "💻 Technologies", "🔒 Sécurité", "📊 Standards"])
            
            with tech_tabs[0]:  # Résumé
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("📄 Type de document", tech_results['document_type'])
                col_b.metric("👥 Auteurs trouvés", len(tech_results['authors']))
                col_c.metric("🔧 Technologies détectées", sum(len(v) for v in tech_results['technologies'].values()))
                
                st.subheader("📄 Informations principales")
                st.write(f"**Titre :** {tech_results['title']}")
                st.write(f"**Type :** {tech_results['document_type']}")
                if tech_results['authors']:
                    st.write(f"**Auteur(s) :** {', '.join(tech_results['authors'])}")
                if tech_results['dates']:
                    st.write(f"**Dates :** {', '.join(tech_results['dates'][:3])}")
            
            with tech_tabs[1]:  # Métadonnées
                st.subheader("📝 Métadonnées extraites")
                
                if tech_results['metadata']:
                    st.write("**Métadonnées PDF :**")
                    for key, value in tech_results['metadata'].items():
                        if value:
                            st.write(f"• **{key.replace('pdf_', '').title()} :** {value}")
                
                if tech_results['authors']:
                    st.write("**Auteurs identifiés dans le texte :**")
                    for author in tech_results['authors']:
                        st.write(f"• {author}")
                
                if tech_results['dates']:
                    st.write("**Dates trouvées :**")
                    for date in tech_results['dates']:
                        st.write(f"• {date}")
            
            with tech_tabs[2]:  # Technologies
                st.subheader("💻 Technologies identifiées")
                
                for category, techs in tech_results['technologies'].items():
                    if techs:
                        st.write(f"**{category.title()} :**")
                        for tech in techs:
                            st.write(f"• {tech}")
                        st.write("")
            
            with tech_tabs[3]:  # Sécurité
                st.subheader("🔒 Éléments de sécurité")
                
                for category, items in tech_results['security'].items():
                    if items:
                        category_name = category.replace('_', ' ').title()
                        st.write(f"**{category_name} :**")
                        for item in items:
                            st.write(f"• {item}")
                        st.write("")
            
            with tech_tabs[4]:  # Standards
                st.subheader("📊 Standards et versions")
                
                if tech_results['standards']:
                    st.write("**Standards identifiés :**")
                    for standard in tech_results['standards']:
                        st.write(f"• {standard}")
                
                if tech_results['versions']:
                    st.write("**Versions mentionnées :**")
                    for version in tech_results['versions']:
                        st.write(f"• {version}")

if __name__ == "__main__":
    main()

#test
