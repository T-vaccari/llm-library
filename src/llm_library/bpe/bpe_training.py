from llm_library.bpe.pretokenization_example import find_chunk_boundaries
import regex as re
from multiprocessing import Pool
import os



def process_chunk(start,end,segment_delimiter,PAT,input_path):
   with open(input_path, "rb") as f:
      f.seek(start)
      chunk = f.read(end - start).decode("utf-8", errors="ignore")
   # Run pre-tokenization on your chunk and store the counts for each pre-token
   """
   The steps to perform in order are the following:
   1)Chunk safealy to obtain chunk to process in parallel(still containing delimiters)
   2)For each chunk strip out all special tokens, and obtain documents from the delimiter, we want to pre-tokenize 
      separately doc1 and doc2 where we have doc1 <|endoftext|>
   3)Run the regex pre tokenization on each doc indipendently in each worker and then I can accumulate per token count.
   
   """
   
   #Split in docs
   
   
   docs = re.split(segment_delimiter,chunk)
   
   #For each docs separately I need to tokenize 
   
   local_pre_token_count = dict()

   for doc in docs:
   
      for match in re.finditer(PAT, doc): # Process every pre-token finded by the regex
         match_encoded = match.group(0).encode("utf-8")
         tuple_of_bytes = tuple([match_encoded[i:i+1] for i in range(len(match_encoded)) ])
         local_pre_token_count[tuple_of_bytes] = local_pre_token_count.get(tuple_of_bytes, 0) + 1

   return local_pre_token_count



def _merge_local_pre_token_counts(results):
   pre_token_count = dict()
   for lptc in results:
         for pre_token in lptc.keys():
            pre_token_count[pre_token] = lptc.get(pre_token, 0) + pre_token_count.get(pre_token,0)
   return pre_token_count



def _return_top_pair(pair_count):
   return max(pair_count, key = lambda  x : (pair_count[x], x))

def _initialize_pair_statistics(pre_token_count):
   pair_count = dict()
   pair_to_key = dict()
   for pre_token in pre_token_count.keys(): #Iterate along all the keys
      for i in range(len(pre_token)-1):
         a = pre_token[i]
         b = pre_token[i+1]
         pair = (a,b)
         pair_count[pair]  = pair_count.get(pair, 0) + 1 * pre_token_count[pre_token]
         pair_to_key.setdefault(pair, set()).add(pre_token)
   return [pair_count, pair_to_key]


# Hot-Spot to optimize------
# Scenario : I have found a top pair , i have decided to merge it, now i have to update the token-count, that
# means changing the keys that contains that couple, 
# Key-Idea : For each couple maintain a set of tuple that are the key affected by that couple
# When updating the key of token count we can also update the statistic without rebuilding them from scratch every time




def _update_token_count(pre_token_count, top_pair, pair_count, pair_to_key):
   # This function is called after a new merge pair has been chose, the goal now is to do the following thing given the top pair:
   # 1)Exploit the pair to key to find the keys affected by that pair
   # 2)Remove the stats of the pre-toke from the pair count, modify the key taking in consideration the new pair, re-update the statistics and then
   # I have to update the pair to key, inserting that the new created pair have that key into the pair_to_key and remove the old key from all the old couples
   # 3) Do it for every key
   # 4) Remove the old key from pair to key

   affected_keys = list(pair_to_key[top_pair])

   for key_to_update in affected_keys: #Iterate along the key to update
      # Firs step, for each pair in the ex-key remove from the pair_to_key mapping, so i can remove the mapping pair -> key
      # Second step, remove the stats of that pair from the dict
      # Step 3 update the key, insert it into the dict taking in consideration the old appearances, iterate on it to update also the pair_to_key struct
      for i in range(len(key_to_update) - 1):
         a = key_to_update[i]
         b = key_to_update[i+1]
         pair = (a, b)
         pair_to_key[pair].discard(key_to_update) #Step 1
         pair_count[pair] -= 1 * pre_token_count[key_to_update] #Step 2

      merged_pre_token = list()
      i = 0
      pre_token = key_to_update
      while i < len(pre_token):
         if i < len(pre_token) - 1 and top_pair[0] == pre_token[i] and top_pair[1] == pre_token[i + 1]:
            merged_pre_token.append(top_pair[0] + top_pair[1])
            i+=2
         else:
            merged_pre_token.append(pre_token[i])
            i+=1

      updated_key = tuple(merged_pre_token)

      pre_token_count[updated_key] = pre_token_count[key_to_update]
      pre_token_count.pop(key_to_update, None)

      # Now i have to update the stats with the new pair and also update the ds
      for i in range(len(updated_key)-1):
         a = updated_key[i]
         b = updated_key[i+1]
         pair = (a,b)
         pair_count[pair]  = pair_count.get(pair, 0) + 1 * pre_token_count[updated_key]
         pair_to_key.setdefault(pair, set()).add(updated_key)


#---------

def train_bpe(
input_path: str | os.PathLike[str],
vocab_size: int,
special_tokens: list[str]
   
)-> tuple[dict[int,bytes], list[tuple[bytes,bytes]]]:

   # Regex pattern 
   PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

   pre_token_count = dict()

   vocab = {i: bytes([i]) for i in range(256)}

   merges = list()

   ## Usage
   with open(input_path, "rb") as f:
      num_processes = 8
      boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

   # The following is a serial implementation, but you can parallelize this
   # by sending each start/end pair to a set of processes.

   escaped_special_tokens = [re.escape(st) for st in special_tokens]
   segment_delimiter = "|".join(escaped_special_tokens)


   jobs = []
   for start, end in zip(boundaries[:-1], boundaries[1:]):
      # Launch process indipendently
      # then wait for them to finish and then merge into a unique vocab
      jobs.append((start,end,segment_delimiter,PAT,input_path))
      
   with Pool(num_processes) as pool:
      results = pool.starmap(process_chunk, jobs)

   # Now i have to merge the result into a unique pre_token_count

   pre_token_count = _merge_local_pre_token_counts(results)

   # pair_to_key should contain the mapping that goes from a given pair to the set of tuples that contains 
   # the byte of the pre-token affected by the pair
   pair_count , pair_to_key = _initialize_pair_statistics(pre_token_count)

   while len(vocab) < (vocab_size - len(special_tokens) ):

      
      # Now i have the pre-token dictionary, that takes into account of every chunk, now I can start to count the byte pairs and simply multiply
      # by the occurences of that precise pre-token

      # Now I can pick-up the most frequent pair, if we have a tie I must choose the lexicographically greater pair
      
      # To obtain the top-pair I can use the lambda that creates tuples, if we have a tie on the appearences then python compares lexicographically 
      # the byte pairs
      if len(pair_count) == 0:
         break

      top_pair = _return_top_pair(pair_count)
      

      # Now I can append the top sequences in the merges, and then I have to substitue in each occurence of the pair with the new byte
      merges.append(top_pair)

      # Now that I have the top pair I can start the merging process, I need to add to the merges the pair chosen for the process
      # When i find the pair, i substitute the occurence in the tuple with the new b'pair' and i can assign to them a new token id

      vocab[len(vocab)] = top_pair[0] + top_pair[1]

      # Now I have to substitute each occuren of the top pair with in the tuple with the fused byte sequence

      _update_token_count(pre_token_count, top_pair, pair_count, pair_to_key)


      # Now I have to recompute stats for the next merge round, 
      # and repeat this process until the vocab size length desired is not reached

   # At the end I can insert in the vocab the special tokens
   for st in special_tokens:
      vocab[len(vocab)] = st.encode("utf-8")  

   return (vocab, merges)




            


         
               




if __name__ == "__main__":
   vocab, merges = train_bpe(
        "data/TinyStoriesV2-GPT4-train.txt",
        500,
        ["<|endoftext|>"],
   )



