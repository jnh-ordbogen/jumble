import spacy
import DateFactory

class Tokenizer:

    def __init__(self) -> None:
        self.nlp = spacy.load("da_core_news_sm")
        self.nlp.add_pipe("date_factory", last=True)

    def tokenize_sentence(self, sentence):
        doc = self.nlp(sentence)

        tokens = []
        current_chunk = []
        found_entities = ''
        in_named_entity = False
        
        for i, token in enumerate(doc):
            if not token.is_punct:
                if self._is_named_entity(token):
                    found_entities += token.text + ' '
                    if in_named_entity:
                        current_chunk.append(token.text)
                    else:
                        current_chunk = [token.text]
                        in_named_entity = True
                else:
                    if in_named_entity:
                        tokens.append(" ".join(current_chunk))
                        in_named_entity = False
                        current_chunk = []
                    tokens.append(token.text)

        if in_named_entity:
            tokens.append(" ".join(current_chunk))

        found_entities = found_entities.split()
        
        named_entity_indexes = []
        for i, token in enumerate(tokens):
            for e in found_entities:
                if e in token:
                    named_entity_indexes.append(i)
                    break
        return named_entity_indexes, tokens
    
    def _is_named_entity(self, token):
        out = token.ent_type_ != "" and len(token.text) > 2 or token._.date_token
        #if out:
        #    print(token)
        return out
