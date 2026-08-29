# Multiclass Bangla Smishing Detection — Project Handoff

## 1. Project Identity

**Project title:** Explainable Multiclass Bangla Smishing Detection Using BanglaBERT and URL Feature Fusion

**Main task:** 3-class Bangla/Banglish SMS classification

**Target classes:**
- `normal`
- `promo`
- `smish`

**Main NLP model:** `csebuetnlp/banglabert`

**Main research components:**
- BanglaBERT transfer learning
- URL/structured feature fusion
- Explainable AI (SHAP or LIME)

**Main evaluation focus:**
- Per-class precision / recall / F1
- Especially **Smishing recall**
- Especially **Smishing → Normal** errors

---

## 2. Dataset

**Dataset file:** `hf_bangla_smishing.csv`

**Size:** 1,253 SMS messages

**Original columns:**
- `label`
- `text`
- `source`

**Missing values:** None found

**Duplicate SMS issue:** No meaningful duplicate problem found

**Source:** `Bengali` only

Because `source` has one unique value, it is kept for documentation but **must not be used as a predictive feature**.

### Class distribution

| Label | Count | Approx. % |
|---|---:|---:|
| `smish` | 495 | 39.51% |
| `normal` | 451 | 35.99% |
| `promo` | 307 | 24.50% |

### Label mapping

```python
label2id = {
    "normal": 0,
    "promo": 1,
    "smish": 2
}

id2label = {
    0: "normal",
    1: "promo",
    2: "smish"
}
```

---

## 3. What Has Already Been Completed

### Dataset and text preprocessing

Completed:

- Dataset loading and inspection
- Missing-value checks
- Duplicate checks
- Label cleaning
- `label → label_id` encoding
- Source-column inspection
- Original text preservation
- Unicode NFC normalization
- Whitespace/newline normalization
- BanglaBERT-compatible normalization

### Important preprocessing decision

**Do not aggressively clean the SMS text.**

The following are intentionally preserved because they may contain useful smishing signals:

- Bangla
- Banglish / code-mixed text
- English words
- Numbers
- URLs
- Phone numbers
- Punctuation
- Currency symbols/terms
- Emojis

Stopwords are not removed.

Stemming and lemmatization are not used.

---

## 4. URL Handling

URLs are a core part of the research design.

The pipeline uses two branches:

```text
Original SMS
     |
     +-----------------------> URL extraction
     |                              |
     |                              v
     |                        URL features
     |
     v
text_for_model
     |
     v
actual URL → [URL]
     |
     v
BanglaBERT
```

### Example

Original:

```text
আপনার অ্যাকাউন্ট বন্ধ হবে। এখনই যাচাই করুন: bit.ly/AccountUpdateBD
```

Text sent to BanglaBERT:

```text
আপনার অ্যাকাউন্ট বন্ধ হবে। এখনই যাচাই করুন: [URL]
```

The original URL remains available for structured URL feature extraction.

This prevents the exact raw URL from being redundantly used in both branches while preserving its useful lexical information.

---

## 5. Engineered Features

### URL features

- `url_count`
- `url_total_length`
- `url_max_length`
- `url_digit_count`
- `url_dot_count`
- `url_hyphen_count`
- `url_slash_count`
- `url_has_https`
- `url_has_ip`
- `url_has_at`
- `url_is_shortened`

### Phone features

- `phone_count`
- `has_phone`

### Number features

- `digit_count`
- `number_count`
- `has_digits`

### Text-length features

- `char_count`
- `word_count`

### Punctuation features

- `exclamation_count`
- `question_count`
- `comma_count`
- `dot_count`
- `colon_count`
- `special_char_count`

### Currency features

- `currency_count`
- `has_currency`

### Security features

- `security_keyword_count`
- `has_security_keyword`

Examples of security-related terms include OTP, verify, login, password, PIN, account, and Bangla equivalents.

### Language/script features

- `bangla_char_count`
- `english_char_count`
- `bangla_ratio`
- `english_ratio`
- `has_bangla`
- `has_english`

---

## 6. Structured Feature Scaling

The structured numerical features were scaled with:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_structured = scaler.fit_transform(
    X_train_raw
)

X_val_structured = scaler.transform(
    X_val_raw
)

X_test_structured = scaler.transform(
    X_test_raw
)
```

### Critical rule

```text
TRAIN       → fit_transform()
VALIDATION  → transform()
TEST        → transform()
```

The scaler is **fit only on the training set** to avoid leakage.

### Important debugging note

During preprocessing, a duplicate `url_count` dataframe column temporarily caused a 35-vs-34 feature mismatch.

The actual intended feature list contains **34 unique structured features**.

The duplicate dataframe-column issue was identified and resolved before completing the preprocessing pipeline.

---

## 7. Train / Validation / Test Split

The data was split approximately:

- Train: 80%
- Validation: 10%
- Test: 10%

The split is stratified by `label`.

Current split sizes:

```text
Train:      1002
Validation: 125
Test:       126
```

Leakage checks were performed to ensure there is no exact SMS overlap between:

- Train ↔ Validation
- Train ↔ Test
- Validation ↔ Test

---

## 8. Class Weights

Class weights were calculated from the **training labels**.

They are available for use with the BanglaBERT classifier when appropriate.

Do not calculate class weights from validation/test data.

---

## 9. Token-Length Analysis

The BanglaBERT tokenizer was used to inspect token lengths.

The project uses a fixed maximum sequence length configured after token-length analysis.

Current setting:

```python
MAX_LENGTH = 128
```

Padding and truncation are applied consistently using this configuration.

---

## 10. BanglaBERT Tokenization

`text_for_model` is tokenized using:

```python
MODEL_NAME = "csebuetnlp/banglabert"
```

Each sample provides:

```text
input_ids
attention_mask
```

The tokenization, padding, and truncation pipeline has been completed for:

- Train
- Validation
- Test

---

## 11. PyTorch Dataset / DataLoader

The prepared PyTorch dataset contains:

```text
input_ids
attention_mask
structured_features
labels
```

DataLoaders were created for:

```text
train_loader
val_loader
test_loader
```

The pipeline has been tested for correct alignment between:

- tokenized text
- structured features
- labels

---

## 12. Important Objects Already Prepared

The notebook should already contain or have saved equivalents of:

```python
df
df_raw

train_df
val_df
test_df

train_encodings
val_encodings
test_encodings

X_train_structured
X_val_structured
X_test_structured

train_dataset
val_dataset
test_dataset

train_loader
val_loader
test_loader

tokenizer
scaler

label2id
id2label

STRUCTURED_FEATURE_COLUMNS
SCALED_FEATURE_COLUMNS
MAX_LENGTH
```

**Do not redo preprocessing unless there is a verified problem.**

---

## 13. Project Environment

The preprocessing environment was prepared in Google Colab with GPU support.

Recorded package versions include:

```text
Pandas       2.2.3
NumPy        2.2.6
Transformers 5.15.1
Datasets     5.0.1
Accelerate   1.14.0
Scikit-learn 1.9.0
```

SHAP and LIME were installed.

Bangla normalizer was installed/tested.

BanglaBERT tokenizer and model were successfully loaded/tested.

### Important environment warning

Do not blindly run a command that upgrades all major packages in Colab:

```python
!pip install -q -U transformers datasets accelerate scikit-learn pandas numpy matplotlib seaborn
```

This previously caused dependency conflicts.

---

# 14. Preprocessing Status

| Item | Status |
|---|---|
| Dataset loading/checking | ✅ |
| Missing values | ✅ |
| Duplicate checks | ✅ |
| Label cleaning | ✅ |
| Source check | ✅ |
| Unicode normalization | ✅ |
| Whitespace cleaning | ✅ |
| Bangla/Banglish/English preservation | ✅ |
| URL preservation/extraction | ✅ |
| Phone/number features | ✅ |
| Text-length features | ✅ |
| Punctuation features | ✅ |
| Currency features | ✅ |
| Security features | ✅ |
| Language/script features | ✅ |
| Label encoding | ✅ |
| Stratified split | ✅ |
| Structured feature scaling | ✅ |
| Class weights | ✅ |
| Token-length analysis | ✅ |
| BanglaBERT tokenization | ✅ |
| Padding/truncation | ✅ |
| PyTorch Dataset | ✅ |
| DataLoader | ✅ |
| Leakage verification | ✅ |
| Preprocessing artifacts | ✅ |

**Preprocessing is complete.**

---

# 15. What Has NOT Been Done Yet

The following work is the responsibility of the next stage:

- TF-IDF + Logistic Regression baseline
- BanglaBERT text-only fine-tuning
- BanglaBERT + URL feature fusion model
- Ablation/comparison experiments
- SHAP/LIME explainability
- Final evaluation
- Confusion matrices
- Error analysis
- Unseen-data/generalization testing if appropriate
- Final demo

---

# 16. EXACT NEXT STEPS

Do not restart preprocessing.

Start with:

## Step 1 — TF-IDF Baseline

```text
text
 ↓
TF-IDF
 ↓
Logistic Regression
 ↓
3-class prediction
```

This is the traditional baseline.

Use:

```text
Train → fit_transform
Validation/Test → transform
```

The baseline should report:

- Accuracy
- Precision
- Recall
- F1
- Per-class metrics
- Confusion matrix
- Smishing recall

---

## Step 2 — BanglaBERT Text-Only Model

Architecture:

```text
text_for_model
      ↓
BanglaBERT
      ↓
classification head
      ↓
normal / promo / smish
```

Use:

```text
normal = 0
promo  = 1
smish  = 2
```

Use class-weighted loss if justified by the training distribution.

Save the best model and training history.

---

## Step 3 — BanglaBERT + URL/Structured Feature Fusion

Main research architecture:

```text
                    SMS
                     |
             ┌───────┴────────┐
             v                v
       text_for_model     structured
             |              features
             v                |
         BanglaBERT           |
             |                |
             └───────┬────────┘
                     v
                Concatenate
                     |
                     v
                 Classifier
                     |
                     v
          normal / promo / smish
```

URL features are the **core feature-fusion contribution**.

The other structured features should be treated as auxiliary/experimental features rather than automatically assuming all of them improve the final model.

---

## Step 4 — Model Comparison / Ablation

At minimum compare:

| Model | Purpose |
|---|---|
| TF-IDF + Logistic Regression | Traditional baseline |
| BanglaBERT | Text-only transfer-learning model |
| BanglaBERT + URL features | Main proposed fusion model |
| BanglaBERT + additional structured features | Optional auxiliary experiment |

Suggested comparison table:

| Model | Accuracy | Macro F1 | Smish Precision | Smish Recall | Smish F1 |
|---|---:|---:|---:|---:|---:|
| TF-IDF + LR | | | | | |
| BanglaBERT | | | | | |
| BanglaBERT + URL | | | | | |
| BanglaBERT + all structured | | | | | |

---

# 17. XAI

After selecting the strongest model:

```text
Best model
    ↓
SHAP or LIME
    ↓
Explanation of individual predictions
```

Focus especially on:

- Why a message was classified as `smish`
- Which text signals contributed
- Which structured URL features contributed

Example interpretation:

```text
Prediction: SMISH

Important signals:
- shortened URL
- security keyword
- account/verification language
- suspicious URL structure
```

---

# 18. Final Evaluation

Report at least:

### Overall

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1
- Weighted F1

### Per class

For:

```text
normal
promo
smish
```

report:

- Precision
- Recall
- F1-score

### Especially report

**Smishing recall**

and

**Smishing → Normal errors**

These are key evaluation points for this project.

---

# 19. Error Analysis

Inspect misclassified SMS messages, especially:

```text
Actual smish → Predicted normal
Actual smish → Predicted promo
Actual promo  → Predicted smish
Actual normal → Predicted smish
```

Look for patterns such as:

- suspicious or unusual URLs
- Banglish/code-mixed text
- short SMS
- security language
- OTP/account language
- promotional language that resembles phishing
- normal messages containing security-related words
- messages with no URL
- unusual punctuation or numbers

Categorize the reasons for errors rather than only listing incorrect predictions.

---

# 20. Repository Structure

Recommended GitHub structure:

```text
Multiclass-Bangla-Smishing-Detection/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   └── Bangla_Smishing_Preprocessing.ipynb
│
├── data/
│   ├── raw/
│   │   └── hf_bangla_smishing.csv
│   │
│   └── processed/
│       ├── train_processed.csv
│       ├── val_processed.csv
│       └── test_processed.csv
│
├── preprocessing/
│   ├── preprocessing_config.json
│   └── class_weights.json
│
├── artifacts/
│   └── structured_feature_scaler.pkl
│
├── docs/
│   └── Bangla_Smishing_Project_Handoff.md
│
├── models/
│   ├── tfidf_baseline/
│   ├── banglabert/
│   └── banglabert_url_fusion/
│
├── outputs/
│   ├── figures/
│   ├── confusion_matrices/
│   └── predictions/
│
└── results/
    ├── metrics.csv
    └── experiment_results.csv
```

---

# 21. Important Rules for the Next Person

1. **Do not restart preprocessing.**

2. **Do not use `source` as a predictive feature.**

3. **Do not fit scalers or TF-IDF on validation/test data.**

4. **Keep `normal=0`, `promo=1`, `smish=2`.**

5. **Keep URLs masked as `[URL]` in the BanglaBERT text branch unless there is a documented research reason to change it.**

6. **Keep original URLs available for URL feature extraction.**

7. **Do not aggressively clean the SMS text.**

8. **Do not assume every structured feature belongs in the final model. Use ablation experiments.**

9. **Record every experiment in a results table.**

10. **Save the best models and their configurations.**

---

# 22. One-Paragraph Handoff

We are building **“Explainable Multiclass Bangla Smishing Detection Using BanglaBERT and URL Feature Fusion.”** The dataset is `hf_bangla_smishing.csv` with 1,253 SMS messages and three classes: `normal` (451), `promo` (307), and `smish` (495). Dataset inspection, label encoding, text normalization, URL extraction, URL/phone/number/text-length/punctuation/currency/security/language feature engineering, stratified train/validation/test splitting, leakage verification, structured-feature scaling, class-weight calculation, token-length analysis, BanglaBERT tokenization, padding/truncation, PyTorch Dataset/DataLoader preparation, and preprocessing artifact saving have been completed. The text branch uses `text_for_model`, where actual URLs are replaced with `[URL]`, while original URLs are retained for structured URL features. The main model is `csebuetnlp/banglabert`. **Do not redo preprocessing.** Continue from the modeling stage with TF-IDF + Logistic Regression baseline, then BanglaBERT text-only fine-tuning, then BanglaBERT + URL/structured feature fusion, followed by model comparison/ablation, SHAP/LIME explainability, final evaluation, and error analysis. The main evaluation focus is Smishing recall and Smishing → Normal errors.
