from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TranslationPipeline
import torch

class InferenceEngine:

    def __init__(self, device='cuda') -> None:
        self.tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-da-en")
        self.device = device
        self.model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-da-en")
        self.model.to(self.device)
        self.pipe = TranslationPipeline(model=self.model, tokenizer=self.tokenizer, task="translation")

    def is_available(self):
        # Check if the model is available on the specified device
        return self.device != 'cpu'

    def translate(self, text):
        if not self.is_available():
            return "Model not available on the specified device."

        # Perform translation
        translation = self.pipe(text, max_length=100, num_return_sequences=1)

        if len(translation) > 0 and 'translation_text' in translation[0]:
            return translation[0]['translation_text']
        else:
            return "Translation failed."

# Example usage:
# engine = InferenceEngine()
# translation = engine.translate("Hello, how are you?")
# print(translation)
