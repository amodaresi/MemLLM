import os
import numpy as np
import pickle
from tqdm.auto import tqdm
import hnswlib
import multiprocessing as mp
import Pyro4
import base64
import argparse
import sqlite3

argparser = argparse.ArgumentParser()
argparser.add_argument('--num_cpus', type=int, default=32)
argparser.add_argument('--server_loc', type=str, default="", help='Checkpoint files')
argparser.add_argument('--server_name', type=str, default="wikidata")
argparser.add_argument('--host', type=str, default="")
argparser.add_argument('--mode', type=str, default="single")
argparser.add_argument('--timeout', type=float, default=0.0)
argparser.add_argument('--port', type=int, default=6381)
args = argparser.parse_args()

N_CPUs = args.num_cpus
SERVER_LOC = args.server_loc
SERVER_NAME = args.server_name
HOST = args.host
PORT = args.port
MODE = args.mode
TIMEOUT = args.timeout

def instance_creator(cls):
    print("(Pyro is asking for a server instance! Creating one!)")
    return cls(server_location=SERVER_LOC, n_cpus=N_CPUs)

Pyro4.config.COMMTIMEOUT = TIMEOUT

@Pyro4.expose
@Pyro4.behavior(instance_mode=MODE, instance_creator=instance_creator)
class RelationMemory:
    def __init__(self, server_location="", n_cpus=-1) -> None:
        self._db_location = os.path.join(server_location, "data.db")
        self._db = sqlite3.connect(":memory:", check_same_thread=False, isolation_level="DEFERRED")
        self._server_location = server_location
        
        self._db_cursor = self._db.cursor()
        self._db_cursor.execute("PRAGMA synchronous = OFF")
        self._db_cursor.execute('PRAGMA journal_mode = MEMORY')
        self._db_cursor.execute(f"PRAGMA mmap_size = {200 * 1024 * 1024 * 1024};")
        self._db_cursor.execute("PRAGMA page_size = 2048")
        self._db_cursor.execute(f"PRAGMA cache_size = {200 * 1024 * 1024 * 1024 // 2048};")

        self._db_cursor.execute("CREATE TABLE IF NOT EXISTS triples (id INTEGER PRIMARY KEY, ent_s_id INTEGER, rel_t_id INTEGER, ent_o_id INTEGER, UNIQUE(ent_s_id,rel_t_id,ent_o_id))")
        self._db_cursor.execute("CREATE TABLE IF NOT EXISTS entities (id INTEGER PRIMARY KEY, entity TEXT UNIQUE NOT NULL)")
        self._db_cursor.execute("CREATE TABLE IF NOT EXISTS relation_types (id INTEGER PRIMARY KEY, relation_type TEXT UNIQUE NOT NULL)")
        self._db_cursor.execute("CREATE INDEX IF NOT EXISTS idx_s_t ON triples (ent_s_id, rel_t_id)")
        self._db_cursor.execute("CREATE INDEX IF NOT EXISTS idx_o_t ON triples (ent_o_id, rel_t_id)")
        # self._db_cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_tr ON triples (ent_s_id, rel_t_id, ent_o_id)")
        # self._db_cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_tr_str ON triples (triplet_str)")
        self._db_cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_ent ON entities (entity)")
        self._db.commit()

        print("DB building complete")

        self._db_lock = mp.Lock()

        self._entity_vec_index = None
        self._relation_type_vec_index = None

        self._n_cpus = n_cpus

    def __del__(self):
        self._db.commit()
        self._db_cursor.close()
        self._db.close()
        print("Session Closed")
    
    @property
    def n_cpus(self):
        return self._n_cpus
    
    @property
    def entity_vec_index(self):
        return self._entity_vec_index
    
    @property
    def relation_type_vec_index(self):
        return self._relation_type_vec_index

    def index_entities(self, dim=128, dist="cosine", hnsw_M=16, hnsw_ef_construction=200, max_elements=999999999):
        if self._entity_vec_index is not None:
            raise Exception("Index already exists.")
        self._entity_vec_index = hnswlib.Index(space=dist, dim=dim)
        self._entity_vec_index.init_index(max_elements=max_elements, ef_construction=hnsw_ef_construction, M=hnsw_M)
        self._entity_vec_index.set_num_threads(self.n_cpus)

    def index_relation_types(self, dim=128, dist="cosine", hnsw_M=16, hnsw_ef_construction=200, max_elements=999):
        if self._relation_type_vec_index is not None:
            raise Exception("Index already exists.")
        self._relation_type_vec_index = hnswlib.Index(space=dist, dim=dim)
        self._relation_type_vec_index.init_index(max_elements=max_elements, ef_construction=hnsw_ef_construction, M=hnsw_M)
        self._relation_type_vec_index.set_num_threads(self.n_cpus)

    def set_index_entities_ef(self, ef):
        self._entity_vec_index.set_ef(ef)
        print(ef)

    def set_index_relation_types(self, ef):
        self._relation_type_vec_index.set_ef(ef)
        print(ef)

    def reindex_entities(self, dim=128, dist="cosine", hnsw_M=16, hnsw_ef_construction=200, max_elements=999999999):
        list_of_elements = self._entity_vec_index.get_ids_list()
        print(type(list_of_elements))
        print(len(list_of_elements))
        vectors = self._entity_vec_index.get_items(list_of_elements)
        print("Loaded vectors", vectors.shape)
        self._entity_vec_index = hnswlib.Index(space=dist, dim=dim)
        self._entity_vec_index.init_index(max_elements=max_elements, ef_construction=hnsw_ef_construction, M=hnsw_M)
        self._entity_vec_index.set_num_threads(self.n_cpus)
        self._entity_vec_index.add_items(vectors, np.array(list_of_elements))
        print("DONE.")

    def get_entities(self, batch_of_entites):
        with self._db_lock:
            self._db_cursor.execute('SELECT id, entity FROM entities WHERE entity IN (' + ("?,"*len(batch_of_entites))[:-1] + ")", batch_of_entites)
            results = self._db_cursor.fetchall()
        results_dict = {e: _id for (_id,e) in results}
        output = []
        for e in batch_of_entites:
            if e in results_dict:
                output.append((results_dict[e], e))
            else:
                output.append(None)
        return output
    
    def get_entities_by_idx(self, batch_of_entites_idxes):
        with self._db_lock:
            self._db_cursor.execute('SELECT id, entity FROM entities WHERE id IN (' + ("?,"*len(batch_of_entites_idxes))[:-1] + ")", batch_of_entites_idxes)
            results = self._db_cursor.fetchall()
        results_dict = {_id: e for (_id,e) in results}
        output = []
        for _id in batch_of_entites_idxes:
            if _id in results_dict:
                output.append((_id, results_dict[_id]))
            else:
                output.append(None)
        return output
    
    def get_relation_types(self, batch_of_types):
        with self._db_lock:
            self._db_cursor.execute('SELECT id, relation_type FROM relation_types WHERE relation_type IN (' + ("?,"*len(batch_of_types))[:-1] + ")", batch_of_types)
            results = self._db_cursor.fetchall()
        results_dict = {t: _id for (_id,t) in results}
        output = []
        for t in batch_of_types:
            if t in results_dict:
                output.append((results_dict[t], t))
            else:
                output.append(None)
        return output

    def get_relation_types_by_idx(self, batch_of_types_idxes):
        with self._db_lock:
            self._db_cursor.execute('SELECT id, relation_type FROM relation_types WHERE id IN (' + ("?,"*len(batch_of_types_idxes))[:-1] + ")", batch_of_types_idxes)
            results =  self._db_cursor.fetchall()
        results_dict = {_id: t for (_id,t) in results}
        output = []
        for _id in batch_of_types_idxes:
            if _id in results_dict:
                output.append((_id, results_dict[_id]))
            else:
                output.append(None)
        return output
    
    def store_entities(self, batch_of_entites, batch_of_entities_vectors):
        batch_of_entities_vectors = [np.frombuffer(base64.b64decode(v["data"]), dtype=np.float32) for v in batch_of_entities_vectors]
        if self._entity_vec_index is None:
            raise Exception("No entity index found")
        
        # print("First Read.")
        # self._db_cursor.execute('SELECT id, entity FROM entities WHERE entity IN (' + ("?,"*len(batch_of_entites))[:-1] + ")", batch_of_entites)
        # pre_result = self._db_cursor.fetchall()
        # print("First Read Done.")
        # already_existing_entities = set([x[1] for x in pre_result])

        print("Pushing to db")
        # self._db_cursor.executemany("INSERT OR IGNORE INTO entities (entity) OUTPUT INSERTED.ID, INSERTED.entity VALUES (?)", [(s,) for s in batch_of_entites])
        successful_ids = []
        for s in batch_of_entites:
            self._db_cursor.execute("INSERT OR IGNORE INTO entities (entity) VALUES (?)", (s,))
            # Check if the insert was successful
            if self._db_cursor.rowcount > 0:  # Rowcount is 1 if the insert was successful
                # Fetch the ID of the last inserted row
                last_id = self._db_cursor.lastrowid
                successful_ids.append(last_id)

        # result = self._db_cursor.fetchall()
        result = self.get_entities_by_idx(successful_ids)
        print("Pushing to db done")
        # self._db_cursor.execute('SELECT id, entity FROM entities WHERE entity IN (' + ("?,"*len(batch_of_entites))[:-1] + ")", batch_of_entites)
        # result = self._db_cursor.fetchall()
        # print("second read done")
        self._db.commit()
        print("Commited")
        result_dict = {k: v for (v,k) in result}
        
        batch_idxes_to_store = [i for i, b in enumerate(batch_of_entites) if b in result_dict]
        idxes_to_store = [result_dict[b] for b in batch_of_entites if b in result_dict]
        print(len(idxes_to_store))
        
        if len(idxes_to_store) > 0:
            self._entity_vec_index.add_items(np.array(batch_of_entities_vectors)[batch_idxes_to_store], np.array(idxes_to_store))
        return successful_ids


    def store_relation_types(self, batch_of_types, batch_of_type_vectors):
        batch_of_type_vectors = [np.frombuffer(base64.b64decode(v["data"]), dtype=np.float32) for v in batch_of_type_vectors]
        if self._relation_type_vec_index is None:
            raise Exception("No relation type index found")
        
        successful_ids = []
        for s in batch_of_types:
            self._db_cursor.execute("INSERT OR IGNORE INTO relation_types (relation_type) VALUES (?)", (s,))
            # Check if the insert was successful
            if self._db_cursor.rowcount > 0:  # Rowcount is 1 if the insert was successful
                # Fetch the ID of the last inserted row
                last_id = self._db_cursor.lastrowid
                successful_ids.append(last_id)
        
        # self._db_cursor.execute('SELECT id, relation_type FROM relation_types WHERE relation_type IN (' + ("?,"*len(batch_of_types))[:-1] + ")", batch_of_types)
        # pre_result = self._db_cursor.fetchall()
        # already_existing_types = set([x[1] for x in pre_result])

        # self._db_cursor.executemany("INSERT OR IGNORE INTO relation_types (relation_type) VALUES (?)", [(s,) for s in batch_of_types])
        # self._db_cursor.execute('SELECT id, relation_type FROM relation_types WHERE relation_type IN (' + ("?,"*len(batch_of_types))[:-1] + ")", batch_of_types)
        # result = self._db_cursor.fetchall()
        self._db.commit()
        result = self.get_relation_types_by_idx(successful_ids)
        result_dict = {k: v for (v,k) in result}
        
        batch_idxes_to_store = [i for i, b in enumerate(batch_of_types) if b in result_dict]
        idxes_to_store = [result_dict[b] for b in batch_of_types if b in result_dict]
        if len(idxes_to_store) > 0:
            self._relation_type_vec_index.add_items(np.array(batch_of_type_vectors)[batch_idxes_to_store], np.array(idxes_to_store))
        return successful_ids
    
    
    def store_relationships(self, batch_of_relationships):
        # triples_to_add = []
        # for i in range(len(batch_of_relationships)):
        #     relationship = batch_of_relationships[i]
        #     rel_str = "_".join([str(i) for i in relationship])
        #     triples_to_add.append(relationship + (rel_str,))
        
        # print("Start to push rels")
        self._db_cursor.executemany("INSERT OR IGNORE INTO triples (ent_s_id, rel_t_id, ent_o_id) VALUES (?,?,?)", batch_of_relationships)
        # print("Get rel ids")
        # # self._db_cursor.execute('SELECT id FROM triples WHERE triplet_str IN (' + ("?,"*len(triples_to_add))[:-1] + ")", [r[-1] for r in triples_to_add])
        # results = []
        # batch_of_relationships
        # for rel in batch_of_relationships:
        #     self._db_cursor.execute(f'SELECT id FROM triples WHERE ent_s_id = ? AND rel_t_id = ? AND ent_o_id = ?', rel)
        #     results.append(self._db_cursor.fetchall())
        # print("Get rel id results")
        # successful_ids = []

        # for s in batch_of_relationships:
        #     self._db_cursor.execute("INSERT OR IGNORE INTO triples (ent_s_id, rel_t_id, ent_o_id) VALUES (?,?,?)", s)
        #     # Check if the insert was successful
        #     if self._db_cursor.rowcount > 0:  # Rowcount is 1 if the insert was successful
        #         # Fetch the ID of the last inserted row
        #         last_id = self._db_cursor.lastrowid
        #         successful_ids.append(last_id)
        print("Start commit")
        self._db.commit()
        print("commited")

        return None
    
    def get_relationships_by_id(self, batch_of_relationships_idxes, replace_id_w_txt=False):
        # return_hash_str = ", triples.triplet_str " if return_hash_str else " "
        if replace_id_w_txt:
            sql = 'SELECT triples.id, entities_1.entity as entity_1, relation_types.relation_type, entities_2.entity as entity_2 FROM triples INNER JOIN entities as entities_1 ON entities_1.id=triples.ent_s_id INNER JOIN relation_types ON relation_types.id=triples.rel_t_id INNER JOIN entities as entities_2 ON entities_2.id=triples.ent_o_id'
        else:
            sql = 'SELECT triples.id, triples.ent_s_id, triples.rel_t_id, triples.ent_o_id FROM triples'
        sql += ' WHERE triples.id IN (' + ("?,"*len(batch_of_relationships_idxes))[:-1] + ")"
        with self._db_lock:
            self._db_cursor.execute(sql, batch_of_relationships_idxes)
            results = self._db_cursor.fetchall()
        results_dict = {v[0]: v for v in results}
        return [results_dict[_idx] for _idx in batch_of_relationships_idxes]
    
    def get_relationships_using_q(self, e_s_q=None, r_t_q=None, e_o_q=None, replace_id_w_txt=False, k=100):
        if replace_id_w_txt:
            sql = 'SELECT triples.id, entities_1.entity as entity_1, relation_types.relation_type, entities_2.entity as entity_2 FROM triples INNER JOIN entities as entities_1 ON entities_1.id=triples.ent_s_id INNER JOIN relation_types ON relation_types.id=triples.rel_t_id INNER JOIN entities as entities_2 ON entities_2.id=triples.ent_o_id'
        else:
            sql = 'SELECT triples.id, triples.ent_s_id, triples.rel_t_id, triples.ent_o_id FROM triples'
        if r_t_q is None:
            raise Exception("Relation type should be defined in a query.")
        if e_s_q is not None and e_o_q is not None:
            raise Exception("Only one of the entities should be defined in a query.")
        if e_s_q is not None:
            sql += f' WHERE triples.rel_t_id = {r_t_q} AND triples.ent_s_id = {e_s_q}'
        else:
            sql += f' WHERE triples.rel_t_id = {r_t_q} AND triples.ent_o_id = {e_o_q}'
        sql += f' LIMIT {k}'
        with self._db_lock:
            self._db_cursor.execute(sql)
            res = self._db_cursor.fetchall()
        return res

    def search_entity(self, entity_vector, dist=1.0, k=100):
        entity_vector = np.frombuffer(base64.b64decode(entity_vector["data"]), dtype=np.float32) 
        labels, distances = self._entity_vec_index.knn_query(entity_vector, k=min(k, self._entity_vec_index.get_current_count()))
        found_idxes = (labels[distances < dist]).tolist()
        output_distances = (distances[distances < dist]).tolist()
        return [x + (d,) for x, d in zip(self.get_entities_by_idx(found_idxes), output_distances)]
    
    def search_relation_type(self, relation_type_vector, dist=1.0, k=25):
        relation_type_vector = np.frombuffer(base64.b64decode(relation_type_vector["data"]), dtype=np.float32) 
        labels, distances = self._relation_type_vec_index.knn_query(relation_type_vector, k=min(k, self._relation_type_vec_index.get_current_count()))
        found_idxes = (labels[distances < dist]).tolist()
        output_distances = (distances[distances < dist]).tolist()

        return [x + (d,) for x, d in zip(self.get_relation_types_by_idx(found_idxes), output_distances)]
    


    def save_snapshot(self):
        os.makedirs(self._server_location, exist_ok=True)
        print("Saving relation type index...")
        self._relation_type_vec_index.save_index(os.path.join(self._server_location, f"rel_type_index.bin"))
        if os.path.isfile(os.path.join(self._server_location, f"entity_index.bin")):
            print("Found previous entities index - renaming...")
            os.rename(os.path.join(self._server_location, f"entity_index.bin"), os.path.join(self._server_location, f"entity_index.bin.bak"))
        print("Saving entities index...")
        self._entity_vec_index.save_index(os.path.join(self._server_location, f"entity_index.bin"))
        if os.path.isfile(os.path.join(self._server_location, f"entity_index.bin.bak")):
            print("Removing previous entities index...")
            os.remove(os.path.join(self._server_location, f"entity_index.bin.bak"))
        print("Storing DB")
        disk_conn = sqlite3.connect(self._db_location)
        with disk_conn:
            self._db.backup(disk_conn)
        disk_conn.close()
        print("DONE")

    def load_snapshot(self):
        if self._relation_type_vec_index is None or self._entity_vec_index is None:
            raise Exception("Create indices first...")

        print("Loading entity index...")
        self._entity_vec_index.load_index(os.path.join(self._server_location, f"entity_index.bin"))
        print(self._entity_vec_index.get_current_count())

        print("Loading relation type index...")
        self._relation_type_vec_index.load_index(os.path.join(self._server_location, f"rel_type_index.bin"))
        print(self._relation_type_vec_index.get_current_count())

        print("Loading DB...")
        disk_conn = sqlite3.connect(self._db_location)
        with self._db:
            disk_conn.backup(self._db)
        disk_conn.close()

        print("DONE")
    
Pyro4.Daemon.serveSimple(
{
    RelationMemory: f"{SERVER_NAME}.RelMem"
},
ns = False, port=PORT, host=HOST)
