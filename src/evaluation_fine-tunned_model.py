import os
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Ortam ayarları
os.environ["TRANSFORMERS_CACHE"] = "/home/data/hf_cache"
os.environ["HF_HOME"] = "/home/data/hf_cache"

# -------------------------
# Fine-Tuned Model ve Tokenizer yükle
# -------------------------
model_path = "/home/data/mistral7b_meditron_peft_ckpts_1"
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto",
    local_files_only=True
)
model.eval()

# -------------------------
# Test verisini yükle
# -------------------------
with open("/home/data/test.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

correct, total = 0, 0
results = []
option_labels = ["A", "B", "C", "D"]

print("Evaluating...")

for sample in tqdm(test_data):
    q = sample["question"]
    options_str = sample["options"]
    correct_ans = sample["answer"].strip().upper().replace("(", "").replace(")", "")

    # Şıkları temizle
    option_lines = [line.strip() for line in options_str.strip().split("\n") if line.strip()]
    option_map = {}
    for line in option_lines:
        if line.startswith("(") and ")" in line:
            label = line[1]
            text = line.split(")", 1)[-1].strip()
            option_map[label] = text

    option_texts = [f"{label}) {option_map[label]}" for label in option_labels if label in option_map]

    # Prompt hazırla
    full_prompt = (
        "Aşağıda bir çoktan seçmeli tıbbi soru verilmiştir. "
        "Soruyu dikkatle okuyun ve en doğru cevabı seçin. "
        "Sadece A, B, C veya D harflerinden birini döndürün.\n\n"
        f"Soru: {q}\n" +
        "\n".join(option_texts) +
        "\nYanıt:"
    )

    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            temperature=0.0,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Tahmini harfi ayıkla
    predicted = ""
    for char in response[::-1]:
        if char.upper() in option_labels:
            predicted = char.upper()
            break

    is_correct = predicted == correct_ans
    correct += int(is_correct)
    total += 1

    results.append({
        "question": q,
        "options": option_texts,
        "correct_answer": correct_ans,
        "predicted_answer": predicted,
        "is_correct": is_correct,
        "raw_response": response
    })

# -------------------------
# Sonuçları yaz
# -------------------------
with open("/home/data/evaluation_mc_results_ft.jsonl", "w", encoding="utf-8") as f_out:
    for row in results:
        json.dump(row, f_out, ensure_ascii=False)
        f_out.write("\n")

# -------------------------
# Accuracy raporu
# -------------------------
accuracy = correct / total if total > 0 else 0.0
print("\n--- Evaluation Report (Fine-Tuned) ---")
print(f"Total Questions: {total}")
print(f"Correct:         {correct}")
print(f"Accuracy:        {accuracy:.4f}")
print("Saved results to /home/data/evaluation_mc_results_ft.jsonl")
