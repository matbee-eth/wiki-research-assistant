import spacy
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('maxent_ne_chunker')
    nltk.download('words')
except Exception as e:
    logging.error(f"Error downloading NLTK data: {str(e)}")

# Initialize spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    logging.info("Downloading spaCy model...")
    import subprocess
    subprocess.run(["/Users/mathieugosbee/miniconda3/envs/researcher/bin/python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load('en_core_web_sm')

def preprocess_text(text: str) -> str:
    """
    Preprocess text by tokenizing and removing stopwords.
    
    Args:
        text (str): Input text to preprocess
        
    Returns:
        str: Preprocessed text
    """
    try:
        # Use spaCy for preprocessing
        doc = nlp(text)
        
        # Remove stopwords and punctuation
        tokens = [token.text.lower() for token in doc if not token.is_stop and not token.is_punct]
        
        return ' '.join(tokens)
    except Exception as e:
        logging.error(f"Error in text preprocessing: {str(e)}")
        return text

def extract_entities(text: str) -> List[Tuple[str, str]]:
    """
    Extract named entities from text using spaCy.
    
    Args:
        text: Input text to process
        
    Returns:
        List of tuples containing (entity_text, entity_label)
    """
    try:
        doc = nlp(text)
        entities = []
        
        # Process each entity
        for ent in doc.ents:
            # Only include relevant entity types
            if ent.label_ in ['PERSON', 'DATE', 'GPE', 'ORG', 'NORP', 'LOC']:
                # Clean and normalize the entity text
                entity_text = ent.text.strip()
                if entity_text:  # Only add non-empty entities
                    entities.append((entity_text, ent.label_))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_entities = []
        for entity in entities:
            if entity not in seen:
                seen.add(entity)
                unique_entities.append(entity)
        
        return unique_entities
        
    except Exception as e:
        logger.error(f"Error extracting entities: {str(e)}", exc_info=True)
        return []

def perform_topic_modeling(documents: list, num_topics: int = 5) -> list:
    """
    Perform topic modeling using Latent Dirichlet Allocation.
    
    Args:
        documents (list): List of document texts
        num_topics (int): Number of topics to extract
        
    Returns:
        list: List of topics with their keywords
    """
    try:
        if not documents or not any(doc.strip() for doc in documents):
            return []
            
        # Preprocess documents
        processed_docs = [preprocess_text(doc) for doc in documents]
        if not any(doc.strip() for doc in processed_docs):
            return []
            
        # Create document-term matrix
        vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
        try:
            doc_term_matrix = vectorizer.fit_transform(processed_docs)
        except ValueError:
            return []
            
        # Apply LDA
        lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        lda.fit(doc_term_matrix)
        
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Extract topics
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-10:-1]]
            topics.append(", ".join(top_words))
            
        return topics
    except Exception as e:
        logger.error(f"Error in topic modeling: {str(e)}")
        return []

def generate_literature_review(summaries: list) -> str:
    """
    Generate a literature review from a list of summaries.
    
    Args:
        summaries (list): List of text summaries
        
    Returns:
        str: Generated literature review
    """
    # Combine summaries
    combined_text = ' '.join(summaries)
    
    # Extract key entities and topics
    entities = extract_entities(combined_text)
    topics = perform_topic_modeling([combined_text])
    
    # Create a structured review
    review = "Literature Review:\n\n"
    
    # Add key entities
    review += "Key Entities:\n"
    for entity in entities:
        review += f"- {entity[0]} ({entity[1]})\n"
    
    # Add main topics
    review += "\nMain Topics:\n"
    for topic in topics:
        review += f"- Topic: {topic}\n"
    
    return review
