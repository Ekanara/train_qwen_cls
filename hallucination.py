from datasets import load_dataset, concatenate_datasets
from sentence_transformers import SentenceTransformer

for index in range(19,18,-1):
    dataset_name = f"sigmaloop/sld-de-duplicated-split-{index}"

    dataset_corpus = load_dataset(
        dataset_name,
    )
    dataset_corpus_all = concatenate_datasets([dataset_corpus['train']])

    # Load the model
    model = SentenceTransformer(
        "Qwen/Qwen3-Embedding-4B",
        model_kwargs={
            "torch_dtype": "bfloat16",
            "attn_implementation": "sdpa",
            "device_map": "auto"
        },
        tokenizer_kwargs={"padding_side": "left"},
    )
    def embedding_text(sample):
    # query_embeddings = model.encode(sample['text'], prompt_name="query")
        document_embeddings = model.encode(sample['text'])
        sample['text_embedding'] = document_embeddings
        return sample

    dataset_corpus_all_embedded = dataset_corpus_all.map(
        embedding_text,
        batched=True,
        batch_size=1
    )

    dataset_corpus_all_embedded.push_to_hub(f"{dataset_name}-embedded")