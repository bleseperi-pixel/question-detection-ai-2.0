# College Question Predictor

A responsive web application for college management. Staff submit the questions they
want to ask. The app finds the questions that repeat most often, ranks them, and builds
a question paper that follows the pattern you choose (for example only 2-mark questions,
only 5-mark questions, or a mix).

## Features

- Paste staff questions, one per line. Numbering and marks such as `(5 marks)` or
  `[2 marks]` are detected automatically.
- Detects near-duplicate questions even when the wording differs.
- Ranks predicted repeated questions by frequency and consistency.
- Generates a question paper with Part A / Part B sections, mark totals and warnings
  when there are not enough questions.
- Print or save the paper as PDF from the browser.
- Responsive layout for phones, tablets and desktops.

## How the AI model works

| Step | Method |
|------|--------|
| Parsing | Regular expressions remove numbering and extract marks |
| Text representation | TF-IDF vectors with word unigrams and bigrams, English stop words removed, sublinear term frequency |
| Similarity | Cosine similarity between question vectors |
| Repetition detection | Greedy clustering: questions with similarity at or above the threshold (default 0.6) form one group |
| Prediction score | `frequency x (0.5 + 0.5 x cohesion)`, where cohesion is the average similarity inside the group |
| Paper building | Top-ranked groups matching each requested mark type; each question is used once |

The model is unsupervised, so it needs no training data. Its accuracy depends on how
many staff questions you provide. The more historical questions you add, the better the
ranking. Treat the output as a strong suggestion and review it before use.

## Project structure

```
college_question_predictor/
├── app.py                  # Flask server and JSON API
├── predictor.py            # Parsing, similarity model, ranking, paper builder
├── requirements.txt
├── README.md
├── templates/
│   └── index.html          # Responsive single-page interface
├── static/
│   ├── css/style.css       # Styles, responsive layout and print rules
│   └── js/app.js           # Form handling and rendering (no innerHTML)
├── sample_data/
│   └── staff_questions.txt # Example input to try
└── tests/
    └── test_predictor.py   # Unit tests for the model and paper builder
```

## Setup and run

Requires Python 3.10 or newer.

```bash
cd college_question_predictor
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000 in your browser.

To try the sample, open `sample_data/staff_questions.txt`, copy its contents into the
input box, set 2-mark and 5-mark counts, and click **Analyze and generate paper**.

## Running the tests

```bash
pytest -q
```

## API

`POST /api/analyze` with a JSON body:

```json
{
  "questions": "1. Explain normalization (5 marks)\n2. Define primary key (2 marks)",
  "title": "Internal Examination",
  "subject": "DBMS",
  "threshold": 0.6,
  "top_n": 10,
  "sections": [{ "marks": 2, "count": 5 }, { "marks": 5, "count": 3 }]
}
```

Response (abbreviated):

```json
{
  "stats": { "questions_read": 2, "unique_questions": 2, "repeated_questions": 0 },
  "predictions": [{ "question": "Explain normalization", "frequency": 1, "marks": 5, "...": "..." }],
  "paper": { "title": "Internal Examination", "sections": [], "total_marks": 0, "warnings": [] }
}
```

Omit `sections` (or send an empty list) to get predictions only.

## Limitations and next steps

- The TF-IDF model matches shared words, so two questions that use completely different
  words for the same idea may not be grouped. A sentence-embedding model (for example
  `sentence-transformers`) can be swapped into `_similarity_matrix` for better semantic
  matching when internet access and extra disk space are available.
- The data is kept in memory only. A database table for subjects, staff and past papers
  is a natural next step.
- Add authentication before deploying for a real college, so only authorised staff can
  submit questions.
