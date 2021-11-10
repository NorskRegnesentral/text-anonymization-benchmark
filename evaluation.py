from __future__ import annotations

import json, re, sys, abc, argparse, math
import numpy as np
from typing import Any, Dict, List, Tuple
from dataclasses import dataclass

import intervaltree

# Tokens that can be ignored from the recall scores (because they
# do not carry much semantic content, and there are discrepancies on
# whether to include them in the annotated spans or no)
TOKENS_TO_IGNORE_IN_COUNTS = ["Mr", "Mrs", "Ms", "and", "or", "from", "until", "to", 
                              "about", "after", "as", "at", "before", "between", "by",
                              "during", "for", "in", "into", "of", "on", "over", "since",
                              "with", "the", "a", "an"]
patterns_to_skip = ["\\b" + re.escape(tok) + "\\b" for tok in TOKENS_TO_IGNORE_IN_COUNTS]
patterns_to_skip += [re.escape(punct) for punct in list(",.-;:/&()[]–'\" ")]
regex_to_skip = re.compile("(?:" + "|".join(patterns_to_skip) + ")", re.IGNORECASE)


class GoldCorpus:
    """Representation of a gold standard corpus for text anonymisation, extracted from a
    JSON file. See annotation guidelines for the TAB corpus for details. """
    
    def __init__(self, gold_standard_json_file:str):
        
        # documents indexed by identifier
        self.documents = {}
        
        # Train/dev/test splits
        self.splits = {}
        
        fd = open(gold_standard_json_file)
        annotated_docs = json.load(fd)
        fd.close()
        print("Reading annotated corpus with %i documents"%len(annotated_docs))
        
        if type(annotated_docs)!=list:
            raise RuntimeError("JSON file should be a list of annotated documents")

        for ann_doc in annotated_docs:
            
            for key in ["doc_id", "text", "annotations", "dataset_type"]:
                if key not in ann_doc:
                    raise RuntimeError("Annotated document is not well formed: missing variable %s"%key)
            
            # Creating the actual document (identifier, text and annotations)          
            new_doc = GoldDocument(ann_doc["doc_id"], ann_doc["text"], ann_doc["annotations"])
            self.documents[ann_doc["doc_id"]] = new_doc
            
            # Adding it to the list for the specified split (train, dev or test)
            data_split = ann_doc["dataset_type"]
            self.splits[data_split] = self.splits.get(data_split, []) + [ann_doc["doc_id"]]
          
            
    def get_entity_recall(self, masked_docs:List[MaskedDocument], include_direct=True, 
                       include_quasi=True):
        """Returns the entity-level recall of the masked spans when compared to the gold 
        standard annotations. Arguments:
        - masked_docs: documents together with spans masked by the system
        - include_direct: whether to include direct identifiers in the metric
        - include_quasi: whether to include quasi identifiers in the metric
        
        The recall is computed at the level of entities and not mentions, and we consider
        an entity to be masked only if all of its mentions are masked. 
        
        If annotations from several annotators are available for a given document, the recall 
        corresponds to a micro-average over the annotators. """

        nb_masked_entities = 0
        nb_entities = 0

   #     print("Computing entity-level recall on %i documents"%len(masked_docs),
   #           ("(include direct identifiers: %s, include quasi identifiers: %s)"
   #           %(include_direct, include_quasi)))

        for doc in masked_docs:
            
            gold_doc = self.documents[doc.doc_id]
            masked_entities = gold_doc.get_masked_entities(doc, include_direct, include_quasi)
            nb_masked_entities += len(masked_entities)
            nb_entities +=  len(gold_doc.get_entities_to_mask(include_direct, include_quasi))
                    
        return nb_masked_entities / nb_entities
    
      
            
    def get_mention_recall(self, masked_docs:List[MaskedDocument], include_direct=True, 
                       include_quasi=True):
        """Returns the mention-level recall of the masked spans when compared to the gold 
        standard annotations. Arguments:
        - masked_docs: documents together with spans masked by the system
        - include_direct: whether to include direct identifiers in the metric
        - include_quasi: whether to include quasi identifiers in the metric
                
        If annotations from several annotators are available for a given document, the recall 
        corresponds to a micro-average over the annotators. """

        nb_masked_mentions = 0
        nb_mentions = 0

   #     print("Computing mention-level recall on %i documents"%len(masked_docs),
   #           ("(include direct identifiers: %s, include quasi identifiers: %s)"
   #           %(include_direct, include_quasi)))

        for doc in masked_docs:
            
            gold_doc = self.documents[doc.doc_id]
            for entity in gold_doc.get_entities_to_mask(include_direct, include_quasi):
                
                masked_mentions = gold_doc.get_masked_mentions(doc, entity)
                s1 = [gold_doc.text[start:end] for start, end in entity.mentions]
                s2 = [gold_doc.text[start:end] for start, end in masked_mentions]
                if s1!= s2:
                    print(s1, s2)
                nb_masked_mentions += len(masked_mentions)
                nb_mentions +=  sum(entity.mention_level_masking)
                    
        return nb_masked_mentions / nb_mentions
    
        
    def get_precision(self, masked_docs:List[MaskedDocument], token_weighting:TokenWeighting):
        """Returns the weighted, token-level precision of the masked spans when compared 
        to the gold standard annotations. Arguments:
        - masked_docs: documents together with spans masked by the system
        - token_weighting: mechanism for weighting the information content of each token
        
        The precision is computed at the level of tokens, and is weighted by its information
        content. If annotations from several annotators are available for a given document, 
        the precision corresponds to a micro-average over the annotators."""      
        
        weighted_true_positives = 0.0
        weighted_masked_tokens = 0.0
        
    #    print("Computing weighted, token-level precision on %i documents"%len(masked_docs))
        
        for doc in masked_docs:
            gold_doc = self.documents[doc.doc_id]
            
            # We extract the list of token-level spans
            masked_tokens = []
            for start, end in doc.masked_spans:
                masked_tokens += list(gold_doc.split_by_tokens(start, end))
            
            # We compute the weights (information content) of each token
            weights = token_weighting.get_weights(gold_doc.text, masked_tokens)
            
            # We store the number of annotators in the gold standard document
            nb_annotators = len(set(entity.annotator for entity in gold_doc.entities.values()))
            
            for (start, end), weight in zip(masked_tokens, weights):
                
                # We extract the annotators that have also masked this token
                annotators = gold_doc.get_annotators_for_span(start, end)
                
                # And update the (weighted) counts
                weighted_true_positives += (len(annotators) * weight)
                weighted_masked_tokens += (nb_annotators * weight)
        
        return weighted_true_positives / weighted_masked_tokens
                         
            


class GoldDocument:
    """Representation of an annotated document"""
    
    def __init__(self, doc_id:str, text:str, annotations:Dict[str,List]):
        """Creates a new annotated document with an identifier, a text content, and 
        a set of annotations (see guidelines)"""
        
        # The (unique) document identifier and its text
        self.doc_id = doc_id
        self.text = text
        
        # Annotated entities (indexed by id)
        self.entities = {}        
        
        for annotator, ann_by_person in annotations.items():
            
            if "entity_mentions" not in ann_by_person:
                raise RuntimeError("Annotations must include entity_mentions")
            
            for entity in self._get_entities_from_mentions(ann_by_person["entity_mentions"]):
                
                # We require each entity_id to be specific for each annotator
                if entity.entity_id in self.entities:
                    raise RuntimeError("Entity ID %s already used by another annotator"%entity.entity_id)
                    
                entity.annotator = annotator
                entity.doc_id = doc_id
                self.entities[entity.entity_id] = entity
            
                    
    def _get_entities_from_mentions(self, entity_mentions):
        """Returns a set of entities based on the annotated mentions"""
        
        entities = {}
        
        for mention in entity_mentions:
                
            for key in ["entity_id", "identifier_type", "start_offset", "end_offset"]:
                if key not in mention:
                    raise RuntimeError("Unspecified key in entity mention: " + key)
                                   
            entity_id = mention["entity_id"]
            start = mention["start_offset"]
            end = mention["end_offset"]
                
            if start < 0 or end > len(self.text) or start >= end:
                raise RuntimeError("Invalid character offsets: [%i-%i]"%(start, end))
                
            if mention["identifier_type"] not in ["DIRECT", "QUASI", "NO_MASK"]:
                raise RuntimeError("Unspecified or invalid identifier type: %s"%(mention["identifier_type"]))

            need_masking = mention["identifier_type"] in ["DIRECT", "QUASI"]
            is_direct = mention["identifier_type"]=="DIRECT"
                
            # We check whether the entity is already defined
            if entity_id in entities:
                    
                # If yes, we simply add a new mention
                current_entity = entities[entity_id]
                current_entity.mentions.append((start, end))
                current_entity.mention_level_masking.append(need_masking)
                    
            # Otherwise, we create a new entity with one single mention
            else:
                new_entity = AnnotatedEntity(entity_id, [(start, end)], need_masking, is_direct, [need_masking])
                entities[entity_id] = new_entity
                
        for entity in entities.values():
            if set(entity.mention_level_masking) != {entity.need_masking}:
                entity.need_masking = True
    #            print("Warning: inconsistent masking of entity %s: %s"
    #                  %(entity.entity_id, str(entity.mention_level_masking)))
                
        return list(entities.values())
    
    def get_masked_entities(self, masked_doc:MaskedDocument, include_direct=True, include_quasi=True):
        """Given a document with a set of masked text spans, determines which entities
        are fully masked (which means that all their mentions are masked in the document),
        and returns their entity identifiers.
        
        Note that the entities are always specific to a given annotator"""
        
        covered_entities = set()
        
        for entity in self.get_entities_to_mask(include_direct, include_quasi):
            
            masked_mentions = self.get_masked_mentions(masked_doc, entity)
            
            for incr, mention in enumerate(entity.mentions):
                
                if mention in masked_mentions:
                    continue
            
                # The masking is sometimes inconsistent for the same entity, 
                # so we verify that the mention does need masking
                elif entity.mention_level_masking[incr]:
                    break
            else:
                covered_entities.add(entity.entity_id)
                                  
        return covered_entities
    
  
    def get_masked_mentions(self, masked_doc:MaskedDocument, entity: AnnotatedEntity):
        """Given a document with a set of masked text spans and a particular entity ID, 
        determines which mentions of that entity are fully masked. """
        
        covered_mentions = set()
            
        for mention_start, mention_end in entity.mentions:
                
            mention_to_mask = self.text[mention_start:mention_end].lower()
                
            # Computes the character offsets that must be masked
            offsets_to_mask = set(range(mention_start, mention_end))

            # Removes offsets that may be skipped

            # We build the set of character offsets that are not covered
            non_covered_offsets = offsets_to_mask - masked_doc.get_masked_offsets()
            
            # If we have not covered everything, we also make sure punctuations
            # spaces, titles are ignored
            if len(non_covered_offsets) > 0:
                for to_skip in regex_to_skip.finditer(mention_to_mask):
                    non_covered_offsets -= set(range(mention_start+to_skip.start(0), 
                                                     mention_start+to_skip.end(0)))

            # If that set is empty, we consider the mention as properly masked
            if len(non_covered_offsets) == 0:
                covered_mentions.add((mention_start, mention_end))
                
        return covered_mentions
    
        
    def get_entities_to_mask(self,  include_direct=True, include_quasi=True):
        """Return entities that should be masked, and satisfy the constraints 
        specified as arguments"""
        
        to_mask = []
        for entity in self.entities.values():     
             
            # We only consider entities that need masking and are the right type
            if not entity.need_masking:
                continue
            elif entity.is_direct and not include_direct:
                continue
            elif not entity.is_direct and not include_quasi:
                continue  
            to_mask.append(entity)
                
        return to_mask      
    
    
    def get_annotators_for_span(self, start_token: int, end_token: int):
        """Given a text span (typically for a token), determines which annotators 
        have also decided to mask it. Concretely, the method returns a (possibly
        empty) set of annotators names that have masked that span."""
        
        
        # We compute an interval tree for fast retrieval
        if not hasattr(self, "masked_spans"):
            self.masked_spans = intervaltree.IntervalTree()
            for entity in self.entities.values():
                if entity.need_masking:
                    for i, (start, end) in enumerate(entity.mentions):
                        if entity.mention_level_masking[i]:
                            self.masked_spans[start:end] = entity.annotator
        
        annotators = set()      
        for mention_start, mention_end, annotator in self.masked_spans[start_token:end_token]:
            
            # We require that the span is fully covered by the annotator
            if mention_start <=start_token and mention_end >= end_token:
                annotators.add(annotator)
                    
        return annotators
                                                
    
    def split_by_tokens(self, start: int, end: int):
        """Generates the (start, end) boundaries of each token included in this span"""
        
        for match in re.finditer("\w+", self.text[start:end]):
            start_token = start + match.start(0)
            end_token = start + match.end(0)
            yield start_token, end_token
                       

  
@dataclass          
class MaskedDocument:
    """Represents a document in which some text spans are masked, each span
    being expressed by their (start, end) character boundaries"""
    
    doc_id: str
    masked_spans : List[Tuple[int, int]]
    
    def get_masked_offsets(self):
        """Returns the character offsets that are masked"""
        if not hasattr(self, "masked_offsets"):
            self.masked_offsets = {i for start, end in self.masked_spans 
                                   for i in range(start, end)}
        return self.masked_offsets
        
        
    

@dataclass
class AnnotatedEntity:
    """Represents an entity annotated in a document, with a unique identifier, 
    a list of mentions (character-level spans in the document), whether it 
    needs to be masked, and whether it corresponds to a direct identifier"""
    
    entity_id: str
    mentions: List[Tuple[int, int]]
    need_masking: bool
    is_direct: bool
    mention_level_masking: List[bool]
    
    def __post_init__(self):
        if self.is_direct and not self.need_masking:
            raise RuntimeError("Direct identifiers must always be masked")  


class TokenWeighting:
    """Abstract class for token weighting schemes (used to compute the precision)"""
    
    @abc.abstractmethod
    def get_weights(self, text:str, text_spans:List[Tuple[int,int]]):
        """Given a text and a list of text spans, returns a list of numeric weights
        (of same length as the list of spans) representing the information content 
        conveyed by each span. 
        
        A weight close to 0 represents a span with low information content (i.e. which
        can be easily predicted from the remaining context), while a weight close to 1
        represents a high information content (which is difficult to predict from the
        context)"""
        
        return
    
class UniformTokenWeighting(TokenWeighting):
    """Uniform weighting (all tokens assigned to a weight of 1.0)"""
    def get_weights(self, text:str, text_spans:List[Tuple[int,int]]):
        return [1.0] * len(text_spans)
    
    
class BertTokenWeighting(TokenWeighting):
    """Token weighting based on a BERT language model. The weighting mechanism
    runs the BERT model on a text in which the provided spans are masked. The
    weight of each token is then defined as 1-(probability of the actual token value).
    
    In other words, a token that is difficult to predict will have a high
    information content, and therefore a high weight, whereas a token which can
    be predicted from its content will received a low weight. """
    
    def __init__(self, max_segment_size = 100):
        """Initialises the BERT tokenizers and masked language model"""
        
        from transformers import BertTokenizerFast, BertForMaskedLM
        self.tokeniser = BertTokenizerFast.from_pretrained('bert-base-uncased')

        import torch
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        self.model = self.model.to(self.device)
        
        self.max_segment_size = max_segment_size
        
        
    def get_weights(self, text:str, text_spans:List[Tuple[int,int]]):
        """Returns a list of numeric weights between 0 and 1, where each value
        corresponds to 1 - (probability of predicting the value of the text span
        according to the BERT model). 
        
        If the span corresponds to several BERT tokens, the probability is the 
        product of the probabilities for each token."""
        
        import torch
        
        # STEP 1: we tokenise the text
        bert_tokens = self.tokeniser(text, return_offsets_mapping=True)
        input_ids = bert_tokens["input_ids"]
        
        # STEP 2: we record the mapping between spans and BERT tokens
        bert_token_spans = bert_tokens["offset_mapping"]
        tokens_by_span = self._get_tokens_by_span(bert_token_spans, text_spans)

        # STEP 3: we mask the tokens that we wish to predict
        attention_mask = bert_tokens["attention_mask"]
        for token_indices in tokens_by_span.values():
            for token_idx in token_indices:
                attention_mask[token_idx] = 0
                input_ids[token_idx] = self.tokeniser.mask_token_id
          
        # STEP 4: we run the masked language model     
        logits = self._get_model_predictions(input_ids, attention_mask)
        
        # We are only interested in the scores for the actual token values
        scores = logits[torch.arange(len(input_ids)), input_ids]
        scores = scores.detach().cpu().numpy()
        
        # STEP 5: we compute the weights from those predictions
        weights = []
        for (span_start, span_end) in text_spans:
            
            # We normalise the prediction scores for each BERT token
            # to a probability. If the span comprises several BERT tokens, 
            # we then take the mininum probability
            probs = [1/(1+np.exp(-scores[token_idx])) 
                     for token_idx in tokens_by_span[(span_start, span_end)]]
            prob = np.min(probs)
            
            # We finally define the weight as -log(p)
            weights.append(-np.log(prob))
        
        return weights
        

    def _get_tokens_by_span(self, bert_token_spans, text_spans):
        """Given two lists of spans (one with the spans of the BERT tokens, and one with
        the text spans to weight), returns a dictionary where each text span is associated
        with the indices of the BERT tokens it corresponds to."""            
        
        # We create an interval tree to facilitate the mapping
        text_spans_tree = intervaltree.IntervalTree()
        for start, end in text_spans:
            text_spans_tree[start:end] = True
        
        # We create the actual mapping between spans and tokens
        tokens_by_span = {span:[] for span in text_spans}
        for token_idx, (start, end) in enumerate(bert_token_spans):
            for span_start, span_end, _ in text_spans_tree[start:end]:
                tokens_by_span[(span_start, span_end)].append(token_idx) 
        
        # And control that everything is correct
        for span in text_spans:
            if len(tokens_by_span[span])==0 :
                raise RuntimeError("Span (%i,%i) without any token"%(span_start, span_end))
        return tokens_by_span        
    
    
    def _get_model_predictions(self, input_ids, attention_mask):
        """Given tokenised input identifiers and an associated attention mask (where the
        tokens to predict have a mask value set to 0), runs the BERT language and returns
        the (unnormalised) prediction scores for each token.
        
        If the input length is longer than max_segment size, we split the document in
        small segments, and then concatenate the model predictions for each segment."""
        
        import torch
        nb_tokens = len(input_ids)
        
        input_ids = torch.tensor(input_ids)[None,:].to(self.device)
        attention_mask = torch.tensor(attention_mask)[None,:].to(self.device)
        
        # If the number of tokens is too large, we split in segments
        if nb_tokens > self.max_segment_size:
            nb_segments = math.ceil(nb_tokens/self.max_segment_size)
            
            # Split the input_ids (and add padding if necessary)
            input_ids_splits = torch.tensor_split(input_ids[0], nb_segments)
            input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_splits, batch_first=True)
            
            # Split the attention masks
            attention_mask_splits = torch.tensor_split(attention_mask[0], nb_segments)
            attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask_splits, batch_first=True)
                   
        # Run the model on the tokenised inputs + attention mask
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        # And get the resulting prediction scores
        scores = outputs.logits
        
        # If the batch contains several segments, concatenate the result
        if len(scores) > 1:
            scores = torch.vstack([scores[i] for i in range(len(scores))])
            scores = scores[:nb_tokens]
        else:
            scores = scores[0]
        
        return scores
        
        
        


def get_masked_docs_from_file(masked_output_file:str):
    """Given a file path for a JSON file containing the spans to be masked for
    each document, returns a list of MaskedDocument objects"""
    
    fd = open(masked_output_file)
    masked_output_docs = json.load(fd)
    fd.close()
    
    if type(masked_output_docs)!= dict:
        raise RuntimeError("%s must contain a mapping between document identifiers"%masked_output_file
                           + " and lists of masked spans in this document")
    
    masked_docs = []
    for doc_id, masked_spans in masked_output_docs.items():
        doc = MaskedDocument(doc_id, [])
        if type(masked_spans)!=list:
            raise RuntimeError("Masked spans for the document must be a list of (start, end) tuples")
        
        for start, end in masked_spans:
            doc.masked_spans.append((start, end))
        masked_docs.append(doc)
        
    return masked_docs
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Computes evaluation metrics for text anonymisation')
    parser.add_argument('gold_standard_file', type=str, 
                        help='the path to the JSON file containing the gold standard annotations')
    parser.add_argument('masked_output_file', type=str, nargs="+",
                        help='the path to the JSON file containing the actual spans masked by the system')
    parser.add_argument('--use_bert', dest='token_weighting', action='store_const', const="bert", default="uniform", 
                        help='use BERT to compute the information content of each content (default: disable weighting)')
    args = parser.parse_args()

    gold_corpus = GoldCorpus(args.gold_standard_file)
    
    for masked_output_file in args.masked_output_file:
        print("=========")
        masked_docs = get_masked_docs_from_file(masked_output_file)
        
        for masked_doc in masked_docs:
            if masked_doc.doc_id not in gold_corpus.documents:
                raise RuntimeError("Document %s not present in gold corpus"%masked_doc.doc_id)
               
        if args.token_weighting == "uniform":
            weighting_scheme = UniformTokenWeighting()
        elif args.token_weighting == "bert":
            weighting_scheme = BertTokenWeighting()
        else:
            raise RuntimeError("Unrecognised weighting scheme:", args.token_weighting)
        
        print("Computing evaluation metrics for", masked_output_file, "(%i documents)"%len(masked_docs))
        print("Weighting scheme:", args.token_weighting)
        
        mention_recall = gold_corpus.get_mention_recall(masked_docs, True, True)
        recall_direct_entities = gold_corpus.get_entity_recall(masked_docs, True, False)
        recall_quasi_entities = gold_corpus.get_entity_recall(masked_docs, False, True)
        weighted_token_precision = gold_corpus.get_precision(masked_docs, weighting_scheme)

        print("==> Mention-level recall on all identifiers: %.3f"%mention_recall)
        print("==> Entity-level recall on direct identifiers: %.3f"%recall_direct_entities)
        print("==> Entity-level recall on quasi identifiers: %.3f"%recall_quasi_entities)
        print("==> Weighted, token-level precision on all identifiers: %.3f"%weighted_token_precision)
        