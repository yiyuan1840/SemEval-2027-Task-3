"""
Compute an LLM score for every sentence, to use as a feature.

For each sentence we ask the LLM (in Swedish):
    "Does the word in [brackets] have this meaning? Answer Ja or Nej."
We don't let it write an answer. We look at the probabilities it gives
to the first answer token and save

    llm_score = log P("Ja") - log P("Nej")

Positive = the LLM leans "yes, target sense"; negative = leans "no".

No labels are used, so we score all 10 words (including the test word)
once and save the result. Model scripts then read the file.

Input:  data/interim/sv_master.csv
Output: data/processed/sv_llm_scores.csv   (word, sentence_id, llm_score)

Needs: torch and transformers (not installed by default).
How to run (from the project folder):
    python src/llm_scores.py
The first run downloads the model (about 6 GB).
"""

from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

PROJECT_DIR = Path(__file__).resolve().parent.parent
MASTER_FILE = PROJECT_DIR / "data" / "interim" / "sv_master.csv"
OUTPUT_FILE = PROJECT_DIR / "data" / "processed" / "sv_llm_scores.csv"


def make_prompt(row):
    """Build the question for one sentence, with the target word in brackets."""
    text = row["text"]
    start = row["start"]
    end = row["end"]
    marked = text[:start] + "[" + text[start:end] + "]" + text[end:]

    return (
        f"Mening: {marked}\n\n"
        f"Betydelse: {row['definition']}\n\n"
        f"Har ordet inom hakparentes [ ] denna betydelse i meningen? "
        f"Svara endast Ja eller Nej."
    )


def main():
    # Use the Mac GPU if available, otherwise the CPU
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print("Using device:", device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float16)
    model.to(device)
    model.eval()

    # The token ids for the two answers (first token of each word)
    ja_id = tokenizer.encode("Ja", add_special_tokens=False)[0]
    nej_id = tokenizer.encode("Nej", add_special_tokens=False)[0]
    print("Token ids: Ja =", ja_id, " Nej =", nej_id)

    master = pd.read_csv(MASTER_FILE, dtype={"gold_senses": str})

    scores = []
    for i in range(len(master)):
        row = master.iloc[i]

        # Put the question in the model's chat format, ending where the
        # assistant's answer would start
        messages = [{"role": "user", "content": make_prompt(row)}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # One forward pass; take the scores for the NEXT token
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1]
        log_probs = torch.log_softmax(logits.float(), dim=-1)

        score = (log_probs[ja_id] - log_probs[nej_id]).item()
        scores.append(score)

        if i % 100 == 0:
            print(f"  {i}/{len(master)}  {row['word']}  score = {score:.2f}")

    result = master[["word", "sentence_id"]].copy()
    result["llm_score"] = scores

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_FILE, index=False)
    print("Saved", OUTPUT_FILE)


if __name__ == "__main__":
    main()
