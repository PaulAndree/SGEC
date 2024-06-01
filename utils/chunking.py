from pypdf import PdfReader
import re
import numpy as np
import ollama
from langchain_community.embeddings import OllamaEmbeddings
from sklearn.metrics.pairwise import cosine_similarity


def pdf_reader(pdf_path):
    reader = PdfReader(pdf_path)
    all_sentences = [] # Go through every page and extract sentences
    
    for i in range(len(reader.pages)):
        page = reader.pages[i].extract_text()
        single_sentence = re.split(r'(?<=[.?!])\s+', page)
        all_sentences.extend(single_sentence)
    
    #store the sentences as a list of dictionaries in order to keep an index and remove the whitelines
    sentences = [{'sentence': x.strip().replace('\n', ''), 'index' : i} for i, x in enumerate(all_sentences)]
    return sentences
    
def combine_sentences(sentences, buffer_size=2):
    for i in range(len(sentences)): # Go through each sentence dict
        combined_sentence = ''
        
        for j in range(i - buffer_size, i): # Add the sentence before  the current one
            if j >= 0: #to avoid index out of range like on the first one, j cant be negative
                combined_sentence += sentences[j]['sentence'] + ' '

        combined_sentence += sentences[i]['sentence'] # Add the current sentence

        for j in range(i + 1, i + 1 + buffer_size):  # Add the sentence after the current one
            if j < len(sentences): # Check if the index j is within the range of the sentences list
                combined_sentence += ' ' + sentences[j]['sentence']
                
        sentences[i]['combined_sentence'] = combined_sentence # Store the combined sentence
    return sentences

#Embedings
def get_embeddings(sentences):
    ollama_emb = OllamaEmbeddings(model="llama3")
    embeddings = ollama_emb.embed_documents([x['combined_sentence'] for x in sentences])
    
    for i, sentence in enumerate(sentences):
        sentence['combined_sentence_embedding'] = embeddings[i]
    return sentences

def calculate_cosine_distances(sentences):
    distances = []
    for i in range(len(sentences) - 1):
        embedding_current = sentences[i]['combined_sentence_embedding']
        embedding_next = sentences[i + 1]['combined_sentence_embedding']
        
        similarity = cosine_similarity([embedding_current], [embedding_next])[0][0] # Calculate cosine similarity
        
        distance = 1 - similarity # Convert to cosine distance
        distances.append(distance)

        sentences[i]['distance_to_next'] = distance # Store distance in the dictionary
    return distances, sentences

def get_chunks(distances, sentences, user_questions):
    # get the distance threshold
    breakpoint_percentile_threshold = 90
    breakpoint_distance_threshold = np.percentile(distances, breakpoint_percentile_threshold)
    num_distances_above_theshold = len([x for x in distances if x > breakpoint_distance_threshold]) # The amount of distances above the threshold
    
    if num_distances_above_theshold <= user_questions: 
        breakpoint_percentile_threshold = 85
        breakpoint_distance_threshold = np.percentile(distances, breakpoint_percentile_threshold)
        num_distances_above_theshold = len([x for x in distances if x > breakpoint_distance_threshold])
    
    if num_distances_above_theshold <= user_questions:
        breakpoint_percentile_threshold = 75
        breakpoint_distance_threshold = np.percentile(distances, breakpoint_percentile_threshold)
        num_distances_above_theshold = len([x for x in distances if x > breakpoint_distance_threshold])
    
        
    # get the index of the distances that are above the threshold.
    indices_above_thresh = [i for i, x in enumerate(distances) if x > breakpoint_distance_threshold] 
    
    for i, breakpoint_index in enumerate(indices_above_thresh):
        start_index = 0 if i == 0 else indices_above_thresh[i - 1]
        end_index = breakpoint_index if i < len(indices_above_thresh) - 1 else len(distances)
        
    #Get the sentences 
    start_index = 0
    chunks = []
    
    for index in indices_above_thresh:
        end_index = index
        group = sentences[start_index:end_index + 1] # Slice the sentence_dicts from the current start index to the end index
        combined_text = ' '.join([d['sentence'] for d in group])
        chunks.append(combined_text)
        start_index = index + 1
    
    if start_index < len(sentences): # The last group, if any sentences remain
        combined_text = ' '.join([d['sentence'] for d in sentences[start_index:]])
        chunks.append(combined_text)
    
    return chunks
