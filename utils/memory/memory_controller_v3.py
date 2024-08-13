import numpy as np
from tqdm.auto import tqdm

import Pyro4

class MemoryControllerForRelations():
    def __init__(
            self,
            pyro_uri="localhost",
            dim=768,
            rel_type_dim=768,
            dist_thr=0.1,
            dist_rel_thr=0.1,
            embedding_function=None,
            embedding_function_rel=None,
            rel_embedding_template="[[RELATION_TYPE]]",
            caching_strategy="all"
        ) -> None:

        
        self.RelMem = Pyro4.Proxy(pyro_uri)
        
        self.dim = dim
        self.rel_type_dim = rel_type_dim
        self.dist_thr = dist_thr
        self.dist_rel_thr = dist_rel_thr
        
        self.emb_func = embedding_function
        self.emb_func_rel_type = embedding_function_rel if embedding_function_rel is not None else embedding_function

        self.cache_strategy = caching_strategy # "all": caches everything in the object, "func": only caches in a single function call
        self.embedding_cache = {}
        self.embedding_cache_rel_type = {}
        self.entity_cache = {}
        self.entity_cache_reversed = {}
        self.relation_type_cache = {}
        self.relation_type_cache_reversed = {}

        self.rel_embedding_template = rel_embedding_template

        self.ef_runtime = 200
        self.search_eps = 0.8

    
    def reset_cache(self):
        self.embedding_cache = {}
        self.entity_cache = {}
        self.relation_cache = {}
        self.entity_cache_reversed = {}
        self.relation_type_cache_reversed = {}
        self.embedding_cache_rel_type = {}


    def index_entities(self, embedding_hnsw_m=20, embedding_hnsw_ef=200, embedding_hnsw_ef_runtime=10, max_elements=100000000):
        self.RelMem.index_entities(dim=self.rel_type_dim, dist="cosine", hnsw_M=embedding_hnsw_m, hnsw_ef_construction=embedding_hnsw_ef, max_elements=max_elements)
        self.RelMem.set_index_entities_ef(embedding_hnsw_ef_runtime)

    def index_relation_types(self, embedding_hnsw_m=20, embedding_hnsw_ef=200, embedding_hnsw_ef_runtime=10, max_elements=999):
        self.RelMem.index_relation_types(dim=self.rel_type_dim, dist="cosine", hnsw_M=embedding_hnsw_m, hnsw_ef_construction=embedding_hnsw_ef, max_elements=max_elements)
        self.RelMem.set_index_relation_types(embedding_hnsw_ef_runtime)
    

    def add_embeddings(self, strings, pbar=False):
        new_strings = [str(s) for s in strings if str(s) not in self.embedding_cache]
        new_embs = self.emb_func(new_strings, pbar=pbar)
        for i, s in enumerate(new_strings):
            self.embedding_cache[s] = new_embs[i]


    def add_rel_embeddings(self, strings, pbar=False):
        new_strings = [str(s) for s in strings if str(s) not in self.embedding_cache_rel_type]
        new_strings_w_templates = [self.rel_embedding_template.replace("[[RELATION_TYPE]]", s) for s in new_strings]
        new_embs = self.emb_func(new_strings_w_templates, pbar=pbar)
        for i, s in enumerate(new_strings):
            self.embedding_cache_rel_type[s] = new_embs[i]
    
    
    def store_relationship_batched_v2(self, relationship_tuples, verbose=True):
        all_entity_phrases = list(set([str(relationship_tuple[0]) for relationship_tuple in relationship_tuples] + [str(relationship_tuple[2]) for relationship_tuple in relationship_tuples]))
        all_rel_phrases = list(set([str(relationship_tuple[1]) for relationship_tuple in relationship_tuples]))

        result = self.RelMem.get_entities(all_entity_phrases)
        new_entity_phrases = [all_entity_phrases[i] for i in range(len(all_entity_phrases)) if result[i] is None]

        rt_result = self.RelMem.get_relation_types(all_rel_phrases)
        new_rel_phrases = [all_rel_phrases[i] for i in range(len(all_rel_phrases)) if rt_result[i] is None]

        self.add_embeddings(new_entity_phrases, pbar=verbose)
        if len(new_entity_phrases) > 0:
            entity_ids = self.RelMem.store_entities(new_entity_phrases, [self.embedding_cache[e].astype(np.float32).tobytes() for e in new_entity_phrases])
            inv_entity_dict = {k: v for k,v in zip(new_entity_phrases, entity_ids)}
        else:
            inv_entity_dict = {}
        for r in result:
            if r:
                inv_entity_dict[r[1]] = r[0]

        self.add_rel_embeddings(new_rel_phrases, pbar=verbose)
        relation_type_ids = self.RelMem.store_relation_types(new_rel_phrases, [self.embedding_cache_rel_type[rt].astype(np.float32).tobytes() for rt in new_rel_phrases])
        inv_relation_type_dict = {k: v for k,v in zip(new_rel_phrases, relation_type_ids)}
        for r in rt_result:
            if r:
                inv_relation_type_dict[r[1]] = r[0]

        if verbose:
            print("Embeddings uploaded")
        
        relationship_tags = []
        for relationship_tuple in relationship_tuples:
            entity_1, rel, entity_2 = relationship_tuple
            entity_1 = str(entity_1)
            entity_2 = str(entity_2)
            rel = str(rel)

            entity_1_id = inv_entity_dict[entity_1]
            entity_2_id = inv_entity_dict[entity_2]
            relation_type_id = inv_relation_type_dict[rel]

            relationship_tag = (entity_1_id, relation_type_id, entity_2_id)
            relationship_tags.append(relationship_tag)

        if verbose:
            print("Pushing relationships...")

        _ = self.RelMem.store_relationships(relationship_tags)


        if self.cache_strategy == "func":
            self.reset_cache()
        return None
    

    def get_entity_by_id(self, entity_id):
        output = self.RelMem.get_entities_by_idx([entity_id])
        return output[0]
    

    def get_relation_type_by_id(self, relation_type_id):
        output = self.RelMem.get_relation_types_by_idx([relation_type_id])
        return output[0]


    def get_relationship_by_id(self, relationship_id):
        entity_1_id, relation_type_id, entity_2_id = self.RelMem.get_relationships([relationship_id])[0]

        entity_1_name = self.get_entity_by_id(entity_1_id)
        entity_2_name = self.get_entity_by_id(entity_2_id)
        rel = self.get_relation_type_by_id(relation_type_id)

        return (entity_1_name, rel, entity_2_name)
    

    def find_entity(self, entity_vector):
        return self.RelMem.search_entity(entity_vector.astype(np.float32).tobytes(), dist=self.dist_thr, k=300)

    def find_relation_type(self, relation_type_vector):
        return self.RelMem.search_relation_type(relation_type_vector.astype(np.float32).tobytes(), dist=self.dist_rel_thr, k=25)

    def query_relationship(self, query_tuple, filtered_relationships=[], q_thr=0.001, Q_THR=30, prioritize_exact_match=True):
        entity_1, rel, entity_2 = query_tuple
        self.add_embeddings([entity_1, entity_2])
        self.add_rel_embeddings([rel])

        query_type = "subj" if entity_1 else "obj"
        if query_type == "subj":
            entity = entity_1
        else:
            entity = entity_2
        
        entity_candidates = self.find_entity(self.embedding_cache[entity])
        relation_type_candidates = self.find_relation_type(self.embedding_cache_rel_type[rel])

        output_relationships = []

        min_score = 1.0
        for e_c in entity_candidates:
            for rt_c in relation_type_candidates:
                score = (e_c[2] + rt_c[2]) / 2
                # print(e_c, rt_c)
                if score < q_thr:
                    query_results = self.RelMem.get_relationships_using_q(e_s_q=None if query_type != "subj" else e_c[0], r_t_q=rt_c[0], e_o_q=None if query_type == "subj" else e_c[0], replace_id_w_txt=True, k=Q_THR + 1)
                    # if len(query_results) > Q_THR:
                    #     continue
                    for result in query_results:
                        if result[0] not in filtered_relationships:
                            min_score = min(min_score, score)
                            output_relationships.append(
                            {
                                "score": score,
                                "rel_id": result[0],
                                "relationship": result[1:4]
                            })

        if prioritize_exact_match and min_score < 0.001:
            return [r for r in output_relationships if r["score"] < 0.001], None

        return sorted(output_relationships, key=lambda x: x["score"]), None

