import spacy

class NEChecker:

    def __init__(self) -> None:
        self.nlp = spacy.load("en_core_web_sm")

    def is_named_entity(self, text: str):
        doc = self.nlp(text)
        
        for token in doc:
            if token.ent_type_ != "":
                return True
        
        return False
