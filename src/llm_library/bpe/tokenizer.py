import pickle
import regex as re

class Tokenizer:
   def __init__(self, vocab, merges, special_tokens=None):
      """
      Construct a tokenizer from a given  vocabulary, list of merges, 
      and (optionally) a list of special tokens. 
      """

      self.vocab = vocab
      self.merges = merges
      self.special_tokens = special_tokens
      self.reversed_vocab =  {v : k for k, v in vocab.items()} #tuple of bytes - > index int
      self.merge_rank = {
         pair: rank
         for rank, pair in enumerate(self.merges)
      }

   @classmethod
   def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
      """ 
      Class method that constructs and returns a Tokenizer from a 
      serialized vocabulary and list of merges (in the same format that your BPE training code output) 
      and (optionally) a list of special tokens.

      """

      # I can assume that the vocab and merges were saved in a pickle file

      with open(vocab_filepath, "rb") as f:
         vocab = pickle.load(f)
      with open(merges_filepath, "rb") as f:
         merges = pickle.load(f)

      return cls(vocab, merges, special_tokens)

   def _encode_normal_text(self, text: str):
      """
      regex pre-tokenization -> BPE merges -> IDs.
      No special-token handling here.
      """

      PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

      merges = self.merges
      reversed_vocab = self.reversed_vocab
      text_merged_encoded = []

      for match in re.finditer(PAT, text): # Process every pre-token finded by the regex
         match_encoded = match.group(0).encode("utf-8")
         tuple_of_bytes = tuple([match_encoded[i:i+1] for i in range(len(match_encoded)) ])
         #Now given the tuple of bytes I have to apply the merges in order or appearance

         merge_rank = self.merge_rank
         
         while True:
            merged_pre_token = list()
            pairs = list()
            for i in range(len(tuple_of_bytes)-1):
               a = tuple_of_bytes[i]
               b = tuple_of_bytes[i+1]
               pair = (a,b)
               pairs.append(pair)
            valid_pairs = [pair for pair in pairs if pair in merge_rank]
            if valid_pairs:
               best_pair = min(valid_pairs, key=lambda pair: merge_rank[pair])
            else:
               best_pair = None
            if best_pair is None:
               break
            j = 0
            pair_to_find = best_pair
            # print(tuple_of_bytes)
            while j < len(tuple_of_bytes):
               if j < len(tuple_of_bytes) - 1 and pair_to_find[0] == tuple_of_bytes[j] and pair_to_find[1] == tuple_of_bytes[j + 1]:
                  merged_pre_token.append(pair_to_find[0] + pair_to_find[1])
                  j+=2
               else:
                  merged_pre_token.append(tuple_of_bytes[j])
                  j+=1
            tuple_of_bytes =tuple( merged_pre_token)
         for byte in tuple_of_bytes:
            text_merged_encoded.append(reversed_vocab[byte])
      return text_merged_encoded
         


   def encode(self, text: str):
      """
      Handle special-token boundaries and delegate ordinary text
      to _encode_normal_text().
      """

      special_tokens = self.special_tokens or []
      encoded = []

      if not special_tokens:
         
         return self._encode_normal_text(text)

      assert(self.special_tokens)
      special_tokens = sorted(
         self.special_tokens,
         key=len,
         reverse=True,
      )
      escaped_special_tokens = [re.escape(st) for st in special_tokens]
      segment_delimiter = "|".join(escaped_special_tokens)
      
      

      parts = re.split(f"({segment_delimiter})", text)

      for part in parts:
         if part in self.special_tokens:
            encoded.append(self.reversed_vocab[part.encode("utf-8")])
         else:
            encoded_list = self._encode_normal_text(part)
            encoded.extend(encoded_list)
            
      return encoded

   def encode_iterable(self, iterable) :
      """
      Given an iterable of
      strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is
      required for memory-efficient tokenization of large files that we cannot directly load into
      memory.
      """
      for text_chunk in iterable:
         assert isinstance(text_chunk, str)
         yield from self.encode(text_chunk) #difference from yield?



   def decode(self, ids: list[int]):

      """
      Decode a sequence of token IDs into text.
      """
      vocab = self.vocab
      result = b"".join(vocab[id] for id in ids)
      return result.decode("utf-8", errors="replace")
      




   


   
      


      

