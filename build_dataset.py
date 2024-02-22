#standard libraries
import time
from typing import List

from rich import print
from torch import cuda
from tqdm.notebook import tqdm

#external files
from preprocessing import FileIO
from llama_index.text_splitter import SentenceSplitter #one of the best on the market
from sentence_transformers import SentenceTransformer
from class_templates import impact_theory_class_properties
from weaviate_interface import WeaviateClient, WeaviateIndexer

def chunk_data(data, text_splitter, content_field='content'):
    return [text_splitter.split_text(d[content_field]) for d in tqdm(data, 'CHUNKING')]

def create_vectors(content_splits: List[List[str]], model: SentenceTransformer):
    text_vector_tuples = []
    for chunk in tqdm(content_splits, 'VECTORS'):
        vectors = model.encode(chunk, show_progress_bar=False, device='cuda:0')
        text_vector_tuples.append(list(zip(chunk, vectors)))
    return text_vector_tuples

def join(corpus, tuples):
    docs = []
    for i, d in enumerate(corpus):
        for j, episode in enumerate(tuples[i]):
            doc = {k:v for k,v in d.items() if k != 'content'}
            video_id = doc['video_id']
            doc['doc_id'] = f'{video_id}_{j}'
            doc['content'] = episode[0]
            doc['content_embedding'] = episode[1].tolist()
            docs.append(doc)
    return docs
    
def create_dataset(corpus: List[dict],
                   embedding_model: SentenceTransformer,
                   text_splitter: SentenceSplitter,
                   file_outpath_prefix: str='./impact-theory-minilmL6',
                   content_field: str='content',
                   embedding_field: str='content_embedding',
                   device: str='cuda:0' if cuda.is_available() else 'cpu'
                   ) -> None:
    '''
    Given a raw corpus of data, this function creates a new dataset where each dataset 
    doc contains episode metadata and it's associated text chunk and vector representation. 
    Output is directly saved to disk. 
    '''
    
    io = FileIO()

    chunk_size = text_splitter.chunk_size
    print(f'Creating dataset using chunk_size: {chunk_size}')
    start = time.perf_counter()
    ########################
    # START YOUR CODE HERE #
    ########################
    content_splits = chunk_data(corpus, text_splitter)
    text_vector_tuples = create_vectors(content_splits, embedding_model)
    joined_docs = join(corpus, text_vector_tuples)
    ########################
    # END YOUR CODE HERE #
    ########################
    file_path = f'{file_outpath_prefix}-{chunk_size}.parquet'
    io.save_as_parquet(file_path=file_path, data=joined_docs, overwrite=False)
    end = time.perf_counter() - start
    print(f'Total Time to process dataset of chunk_size ({chunk_size}): {round(end/60, 2)} minutes')
    return joined_docs

def create_index(data: List[dict], class_name: str, client: WeaviateClient) -> None:

    class_config = {'classes': [

                        {"class": class_name,        
                        
                        "description": "Episodes of Impact Theory up to Nov 2023", 
                        
                        "vectorIndexType": "hnsw", 
                        
                        # Vector index specific settings
                        "vectorIndexConfig": {                   
                            
                            "ef": 400,
                            "efConstruction": 128, 
                            "maxConnections": 64,  
                            "dynamicEfMin": 100,   
                            "dynamicEfMax": 400,    
                                                },
                        
                        "vectorizer": "none",            
                        
                        # pre-defined property mappings
                        "properties": impact_theory_class_properties }         
                        ]
                }
    client.schema.create(class_config)
    indexer = WeaviateIndexer(client, batch_size=200, num_workers=2)
    indexer.batch_index_data(data, class_name)