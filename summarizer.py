import json
from transformers import BartTokenizer, BartForConditionalGeneration
from langchain.text_splitter import RecursiveCharacterTextSplitter

class TextSummarizer:
    def __init__(self, config_file):
        self.config = self.load_config(config_file)
        self.model, self.tokenizer = self.initialize_model(self.config)

    def load_config(self, config_file):
        with open(config_file, "r") as file:
            config = json.load(file)
        return config

    def initialize_model(self, config):
        tokenizer = BartTokenizer.from_pretrained(config["model_name"])
        model = BartForConditionalGeneration.from_pretrained(config["model_name"])
        return model, tokenizer

    def summarize_text(self, text, sum_max_length, sum_min_length):
        inputs = self.tokenizer(text, return_tensors="pt", max_length = self.config['chunk_size'], truncation=True)
        summary_ids = self.model.generate(
            inputs["input_ids"],
            max_length=sum_max_length,
            min_length=sum_min_length,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3
        )
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

    def process_text(self, input_text, summarization_tokens):
    # Use LangChain's RecursiveCharacterTextSplitter to split the text into chunks
        max_length = summarization_tokens["max_length"]
        min_length = summarization_tokens["min_length"]
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config["chunk_size"],
            chunk_overlap=0,
            separators=["\n", " ", ""]
        )
        chunks = text_splitter.split_text(input_text)

        summaries = []

        # Summarize each chunk and store the results
        for chunk in chunks:
            summary = self.summarize_text(chunk, sum_max_length=max_length, sum_min_length=min_length)
            # print(summarization_tokens)
            summaries.append(summary)
            print(summaries)

        # Combine the summaries into a final summary
        final_summary = " ".join(summaries)

        # Further summarize the combined summary to limit to 4-5 sentences
        final_summary = self.summarize_text(final_summary, sum_max_length=150, sum_min_length=50)
        print(final_summary)
        return final_summary
