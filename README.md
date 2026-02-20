# Suicide and Depression Detection using Machine Learning

## Overview
This tutorial demonstrates building a machine learning model to detect suicidal and depression-related content from social media posts (Facebook, Instagram, Twitter) using Natural Language Processing (NLP) and Long Short-Term Memory (LSTM) neural networks with GloVe embeddings.

## Problem Statement
**Mental health crisis statistics:**
- Over 48,000 suicide-related deaths in USA last year
- Approximately 132 people commit suicide daily
- Anxiety, stress, overthinking, family pressure, and depression are main factors

**Goal:** Build an AI system to automatically detect suicide-related content in social media posts.

---

## TECHNOLOGY STACK

### Libraries and Frameworks

**Data Processing:**
- **NumPy** - Multi-dimensional array calculations
- **Pandas** - Data loading and manipulation
- **pickle** - Model serialization

**Machine Learning:**
- **scikit-learn** - Train-test split, metrics
- **TensorFlow 2.3.0** - Deep learning backend
- **Keras 2.4.3** - Neural network API
- **LabelEncoder** - Label encoding (0 to 1)

**NLP Processing:**
- **NEATTEXT** - Text cleaning (remove special characters, stop words)
- **Keras Tokenizer** - Text tokenization
- **pad_sequences** - Sequence padding

**Visualization:**
- **Plotly Express** - Interactive charts
- **Matplotlib** - Data visualization

**Web Application:**
- **Streamlit** - Web app framework
- **tqdm** - Progress bar display

---

## DATASET

### Source
- **Platform:** Kaggle
- **Format:** CSV file (suicide_detection.csv)
- **Structure:** 2 columns

### Columns
1. **Text** - Social media post content
2. **Class** - Label (Suicide / Non-Suicide)

### Dataset Characteristics
- **Total records:** ~180,000+ posts
- **Suicide posts:** ~92,831
- **Non-Suicide posts:** ~92,831
- **Balance:** Well-balanced dataset (no imbalance handling needed)
- **Train/Test split:** 80/20

---

## STEP-BY-STEP PROCESS

### Part 1: Import Libraries

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from neattext.functions import clean_text
import plotly.express as px
from sklearn.metrics import classification_report
import keras
from keras.layers import Embedding, Dense, LSTM, GlobalMaxPooling1D, Input
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.models import Sequential
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
import pickle
```

---

### Part 2: Load and Explore Data

```python
# Load dataset
df = pd.read_csv('suicide_detection.csv')

# View first 5 rows
df.head()

# Check data balance
df['class'].value_counts()

# View class distribution
# Output: Suicide: 92,831, Non-Suicide: 92,831
```

**Observation:** Dataset is perfectly balanced - no need for resampling techniques.

---

### Part 3: Train-Test Split

```python
# Split data 80-20
X = df['text']
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.20, 
    random_state=42
)

# Verify split maintains class balance
y_train.value_counts()
```

**Result:**
- Training: 80% of data
- Testing: 20% of data
- Both splits maintain 50-50 suicide/non-suicide ratio

---

### Part 4: Visualize Data Distribution

```python
import plotly.express as px

# Plot training data distribution
fig = px.bar(y_train.value_counts())
fig.show()
```

**Visualization shows:**
- Non-Suicide: ~92,800 posts
- Suicide: ~92,800 posts
- Balanced distribution

---

### Part 5: Data Cleaning

#### Why Clean Data?
- Remove special characters
- Remove stop words
- Convert to lowercase
- Prepare for tokenization
- Improve model performance

#### Cleaning Function

```python
from neattext.functions import clean_text

def clean_text_data(texts):
    text_lengths = []
    cleaned_texts = []
    
    for text in tqdm(texts):
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters using neattext
        text = clean_text(text)
        
        # Store text length
        text_lengths.append(len(text))
        
        # Store cleaned text
        cleaned_texts.append(text)
    
    return cleaned_texts, text_lengths
```

#### Apply Cleaning

```python
# Clean training data
clean_train_texts, train_text_lengths = clean_text_data(X_train)

# Clean testing data
clean_test_texts, test_text_lengths = clean_text_data(X_test)
```

**What gets cleaned:**
- "I'm SO HAPPY!!!" → "im happy"
- "East coast... what??" → "east coast what"
- "@user #hashtag" → "user hashtag"
- Special characters, punctuation removed

---

### Part 6: Tokenization

#### Purpose
Convert text to numerical sequences for LSTM processing.

#### Process

```python
# Initialize tokenizer
tokenizer = Tokenizer()

# Fit on training text
tokenizer.fit_on_texts(clean_train_texts)

# Convert text to sequences
train_sequences = tokenizer.texts_to_sequences(clean_train_texts)
test_sequences = tokenizer.texts_to_sequences(clean_test_texts)
```

**Example transformation:**
- **Original:** "i am sad"
- **Tokens:** ["i", "am", "sad"]
- **Sequence:** [45, 12, 789]

---

### Part 7: Sequence Padding

#### Why Pad?
LSTM requires fixed-length input sequences.

```python
# Define maximum sequence length
MAX_LENGTH = 50

# Pad training sequences
train_padded = pad_sequences(
    train_sequences,
    maxlen=MAX_LENGTH,
    padding='post'
)

# Pad testing sequences
test_padded = pad_sequences(
    test_sequences,
    maxlen=MAX_LENGTH,
    padding='post'
)
```

**Example:**
- **Sequence:** [45, 12, 789]
- **Padded (MAX_LENGTH=50):** [45, 12, 789, 0, 0, 0, ..., 0]

**Parameters:**
- `maxlen=50` - All sequences truncated/padded to 50 tokens
- `padding='post'` - Add zeros at end
- Adjustable based on GPU capacity (can use 5000, 10000)

---

### Part 8: GloVe Embeddings

#### What is GloVe?

**GloVe = Global Vectors for Word Representation**

**How it works:**
- Matrix factorization technique
- Works on word context matrix
- Pre-trained on massive corpus (Wikipedia, Twitter)
- Creates dense word vectors

**Why use pre-trained GloVe?**
- Training from scratch takes 5-10 days
- Requires 30+ million parameters
- Pre-trained models already optimized
- Saves computational resources

#### GloVe File Used

**Model:** `glove.840B.300d.pkl`
- **Size:** ~3GB
- **Format:** Pickle file (.pkl)
- **Dimensions:** 300-dimensional vectors
- **Vocabulary:** 840 billion tokens
- **Source:** Kaggle (Stanford NLP converted to pickle)

**Download location:**
- Kaggle: Pre-converted pickle format (recommended)
- Stanford NLP: Original text format (requires conversion)

---

### Part 9: Label Encoding

```python
from sklearn.preprocessing import LabelEncoder

# Initialize encoder
label_encoder = LabelEncoder()

# Encode labels (Suicide=1, Non-Suicide=0)
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)
```

**Result:**
- "Suicide" → 1
- "Non-Suicide" → 0

---

### Part 10: Load GloVe Embeddings

```python
import pickle

# Load GloVe pickle file
with open('glove.840B.300d.pkl', 'rb') as file:
    glove_embeddings = pickle.load(file)
```

**Note:** File must be in working directory.

---

### Part 11: Create Embedding Matrix

#### Purpose
Map tokenized words to GloVe vectors for the embedding layer.

```python
# Get vocabulary size
vocab_size = len(tokenizer.word_index) + 1

# Create empty embedding matrix
embedding_dim = 300  # GloVe dimension
embedding_matrix = np.zeros((vocab_size, embedding_dim))

# Fill matrix with GloVe vectors
for word, idx in tokenizer.word_index.items():
    embedding_vector = glove_embeddings.get(word)
    
    if embedding_vector is not None:
        # Replace zero with actual vector
        embedding_matrix[idx] = embedding_vector
```

**Process:**
1. Create zero matrix (vocab_size × 300)
2. For each word in vocabulary:
   - Look up word in GloVe
   - If found, replace zeros with GloVe vector
   - If not found, keep zeros

**Result:** Matrix where each row = 300-dimensional word vector

---

### Part 12: Setup Early Stopping

```python
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Early stopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Reduce learning rate on plateau
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=0.00001
)
```

**Why use early stopping?**
- Set epochs=100, but model may converge at epoch 20
- Automatically stops training when no improvement
- Saves time and prevents overfitting

---

### Part 13: Build LSTM Model

#### Model Architecture

```python
from keras.models import Sequential
from keras.layers import Embedding, LSTM, GlobalMaxPooling1D, Dense, Input

# Initialize sequential model
model = Sequential()

# Input layer (sequence length = 50)
model.add(Input(shape=(50,)))

# Embedding layer (with GloVe weights)
model.add(Embedding(
    input_dim=vocab_size + 1,
    output_dim=300,
    weights=[embedding_matrix],
    trainable=False  # Freeze GloVe weights
))

# LSTM layer
model.add(LSTM(
    units=128,
    return_sequences=True  # Required for next layer
))

# Global max pooling (down-sampling)
model.add(GlobalMaxPooling1D())

# Output layer (binary classification)
model.add(Dense(
    units=1,
    activation='sigmoid'
))
```

#### Layer Breakdown

**1. Input Layer:**
- Shape: (50,) - 50-token sequences

**2. Embedding Layer:**
- Input: vocab_size + 1
- Output: 300 dimensions (GloVe)
- Weights: Pre-trained GloVe matrix
- Trainable: False (frozen weights)

**3. LSTM Layer:**
- Units: 128 cells
- Return sequences: True
- Recurrent: Learns from past context

**4. GlobalMaxPooling1D:**
- Down-samples by taking maximum values
- Reduces dimensionality over time
- Prepares for dense layer

**5. Dense Output Layer:**
- Units: 1 (binary output)
- Activation: Sigmoid (0 to 1 probability)

---

### Part 14: Compile Model

```python
from keras.optimizers import SGD

# Compile model
model.compile(
    optimizer=SGD(
        learning_rate=0.01,
        momentum=0.9
    ),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

**Configuration:**
- **Optimizer:** SGD (Stochastic Gradient Descent)
- **Learning rate:** 0.01
- **Momentum:** 0.9
- **Loss:** Binary crossentropy (binary classification)
- **Metrics:** Accuracy

---

### Part 15: View Model Summary

```python
model.summary()
```

**Output shows:**
- Total parameters
- Trainable parameters
- Non-trainable parameters (frozen GloVe)
- Layer dimensions

---

### Part 16: Train Model

```python
# Train model
history = model.fit(
    train_padded,
    y_train_encoded,
    validation_data=(test_padded, y_test_encoded),
    epochs=20,
    batch_size=256,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)
```

**Parameters:**
- **Epochs:** 20 (may stop early)
- **Batch size:** 256 sequences per batch
- **Validation data:** Test set
- **Callbacks:** Early stopping + learning rate reduction

**Training time:** ~20 minutes (depending on hardware)

---

### Part 17: Training Results

**Final results (stopped at epoch 14):**
- **Training accuracy:** 94% (0.94)
- **Validation accuracy:** 93.14% (0.9314)
- **Loss:** Decreased significantly
- **Early stopping triggered:** No improvement after epoch 14

**Why stopped early?**
Model reached optimal performance, preventing overfitting.

---

### Part 18: Evaluate Model

```python
from sklearn.metrics import classification_report

# Predict on test data
# Note: Use predict_classes() for TensorFlow 2.3.0 / Keras 2.4.3
predictions = model.predict_classes(test_padded)

# Classification report
print(classification_report(y_test_encoded, predictions))
```

**Important version note:**
- **TensorFlow:** 2.3.0
- **Keras:** 2.4.3
- `predict_classes()` deprecated in newer versions
- Use `np.argmax(model.predict(X), axis=1)` for newer TF

#### Classification Report Results

**Testing Data:**
- **Precision:** 0.93 (93%)
- **Recall:** 0.93 (93%)
- **F1-Score:** 0.93 (93%)
- **Accuracy:** 93%

**Training Data:**
- **Accuracy:** 94%

**Conclusion:** Model generalizes well (minimal overfitting)

---

### Part 19: Test on Real Examples

#### Test Case 1: Suicidal Content

```python
test_text = "This past year thought of suicide fear and anxiety so close to my limit"

# Preprocess
sequence = tokenizer.texts_to_sequences([test_text])
padded = pad_sequences(sequence, maxlen=50)

# Predict
prediction = model.predict(padded)[0][0]

print(f"Probability: {prediction * 100:.2f}%")
# Output: 95% probability - Potential Suicide Post
```

#### Test Case 2: Explicit Threat

```python
test_text = "I will kill myself"

# Preprocess and predict
sequence = tokenizer.texts_to_sequences([test_text])
padded = pad_sequences(sequence, maxlen=50)
prediction = model.predict(padded)[0][0]

print(f"Probability: {prediction * 100:.2f}%")
# Output: 86% probability - Potential Suicide Post
```

#### Test Case 3: Positive Content

```python
test_text = "I am happy"

# Preprocess and predict
sequence = tokenizer.texts_to_sequences([test_text])
padded = pad_sequences(sequence, maxlen=50)
prediction = model.predict(padded)[0][0]

print(f"Probability: {prediction * 100:.2f}%")
# Output: 0.36% probability - Non-Suicide Post
```

**Interpretation:**
- Probability > 0.5 → Suicide risk
- Probability < 0.5 → Non-suicide

---

### Part 20: Save Model and Tokenizer

```python
import pickle
from keras.models import save_model

# Save tokenizer
with open('tokenizer.pkl', 'wb') as file:
    pickle.dump(tokenizer, file)

# Save model
model.save('suicide_detection_model.h5')
```

**Files created:**
1. `tokenizer.pkl` - Tokenizer with vocabulary
2. `suicide_detection_model.h5` - Trained LSTM model

---

### Part 21: Load Saved Model

```python
import pickle
from keras.models import load_model

# Load tokenizer
with open('tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

# Load model
model = load_model('suicide_detection_model.h5')

# Test loaded model
test_text = "I need help"
sequence = tokenizer.texts_to_sequences([test_text])
padded = pad_sequences(sequence, maxlen=50)
prediction = model.predict(padded)[0][0]

print(f"Prediction: {prediction}")
```

**Use case:** Deploy without retraining

---

## BUILDING WEB APPLICATION WITH STREAMLIT

### Part 22: Create app.py

```python
import streamlit as st
import pickle
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import plotly.express as px
import pandas as pd

# Load tokenizer and model
with open('tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

model = load_model('suicide_detection_model.h5')

# Main function
def main():
    # Title
    st.title("Suicide Post Detection App")
    
    # Subheader
    st.subheader("Input the post content below")
    
    # Text input
    sentence = st.text_input("Enter your post content here")
    
    # Predict button
    button = st.button("Predict")
    
    if button:
        # Display post
        st.write(f"Post: {sentence}")
        
        # Preprocess
        sequence = tokenizer.texts_to_sequences([sentence])
        padded = pad_sequences(sequence, maxlen=50)
        
        # Predict
        prediction = model.predict(padded)[0][0]
        
        # Interpret result
        if prediction > 0.5:
            st.error("🚨 Potential Suicide Post")
        else:
            st.success("✅ Non-Suicide Post")
        
        # Calculate probabilities
        suicide_prob = prediction * 100
        non_suicide_prob = (1 - prediction) * 100
        
        # Display probability
        st.write(f"**LSTM + GloVe Model:** There is a higher {suicide_prob:.2f}% probability that the post content is a potential suicide post compared to {non_suicide_prob:.2f}% probability of being a non-suicide post.")
        
        # Create probability chart
        prob_dict = {
            'Class': ['Potential Suicide', 'Non-Suicide'],
            'Probability': [suicide_prob, non_suicide_prob]
        }
        
        df = pd.DataFrame(prob_dict)
        
        # Plot bar chart
        fig = px.bar(
            df,
            x='Class',
            y='Probability',
            color='Class',
            title='Prediction Probability'
        )
        
        st.plotly_chart(fig)

if __name__ == '__main__':
    main()
```

---

### Part 23: Run Streamlit App

**Command:**
```bash
streamlit run app.py
```

**Access:**
- **Local URL:** http://localhost:8501
- Opens automatically in default browser

---

### Part 24: Using the Web App

#### Interface Features

**1. Title:** "Suicide Post Detection App"

**2. Input Field:** Text box for post content

**3. Predict Button:** Trigger prediction

**4. Results Display:**
- Classification (Suicide / Non-Suicide)
- Probability percentages
- Interactive bar chart (Plotly)

#### Example Interactions

**Test 1: High-risk content**
- **Input:** "I will kill myself"
- **Prediction:** Potential Suicide Post
- **Probability:** 86% suicide risk
- **Chart:** Visual comparison (86% vs 14%)

**Test 2: Cry for help**
- **Input:** "I need help just help me I'm crying so hard"
- **Prediction:** Potential Suicide Post
- **Probability:** 96% suicide risk
- **Chart:** Visual comparison (96% vs 4%)

**Test 3: Hopelessness**
- **Input:** "I have nothing to live for my life is so bleak"
- **Prediction:** Potential Suicide Post
- **Probability:** 85% suicide risk

**Test 4: Positive content**
- **Input:** "I am so happy today"
- **Prediction:** Non-Suicide Post
- **Probability:** 1% suicide risk, 98% non-suicide
- **Chart:** Visual comparison (1% vs 98%)

---

## KEY CONCEPTS EXPLAINED

### LSTM (Long Short-Term Memory)

**What is LSTM?**
- Type of Recurrent Neural Network (RNN)
- Learns from sequential data
- Has memory cells
- Can remember long-term dependencies

**Why use LSTM for this task?**
- Text is sequential (word order matters)
- Context from previous words important
- Handles variable-length sequences
- Better than simple RNN for long text

**How LSTM works:**
1. Input: Word sequence
2. Memory: Stores context from previous words
3. Gates: Control information flow
4. Output: Classification prediction

### GloVe Embeddings

**What are embeddings?**
- Convert words to dense numerical vectors
- Similar words have similar vectors
- Captures semantic meaning

**GloVe vs Other Methods:**

**GloVe (Global Vectors):**
- Uses global word co-occurrence statistics
- Matrix factorization
- Pre-trained on massive corpus

**Word2Vec:**
- Predicts words from context
- Skip-gram or CBOW architecture

**TF-IDF:**
- Term frequency × Inverse document frequency
- Sparse vectors
- No semantic understanding

**Why GloVe chosen:**
- 300 dimensions (rich representation)
- Pre-trained (saves time)
- Proven performance on NLP tasks

### Binary Classification

**What is binary classification?**
- Two possible outcomes (Suicide / Non-Suicide)
- Output: Probability between 0 and 1
- Threshold: 0.5 (adjustable)

**Sigmoid activation:**
- Squashes output to 0-1 range
- Probability interpretation

### Tokenization

**Purpose:** Convert text to numbers

**Process:**
1. **Split:** "I am sad" → ["I", "am", "sad"]
2. **Assign IDs:** {"I": 45, "am": 12, "sad": 789}
3. **Create sequence:** [45, 12, 789]

**Vocabulary:**
- Built from training data
- Each unique word gets ID
- Unknown words mapped to special token

### Padding

**Why pad sequences?**
- Neural networks need fixed input size
- Sentences have variable lengths

**Methods:**
- **Post-padding:** Add zeros at end (used here)
- **Pre-padding:** Add zeros at start

**Example:**
- **Original:** [45, 12, 789] (length 3)
- **Padded:** [45, 12, 789, 0, 0, ..., 0] (length 50)

---

## MODEL EVALUATION METRICS

### Accuracy
**Formula:** (TP + TN) / (TP + TN + FP + FN)
**Result:** 93%
**Meaning:** 93% of predictions correct

### Precision
**Formula:** TP / (TP + FP)
**Result:** 93%
**Meaning:** Of posts flagged as suicide, 93% actually are

### Recall
**Formula:** TP / (TP + FN)
**Result:** 93%
**Meaning:** 93% of actual suicide posts detected

### F1-Score
**Formula:** 2 × (Precision × Recall) / (Precision + Recall)
**Result:** 93%
**Meaning:** Balanced measure of precision and recall

**Where:**
- TP = True Positives
- TN = True Negatives
- FP = False Positives
- FN = False Negatives

---

## DEPLOYMENT CONSIDERATIONS

### Real-World Application

**Use Cases:**
1. **Social Media Monitoring:**
   - Automatic flagging of concerning posts
   - Alert moderators for review
   - Trigger support resources

2. **Crisis Hotlines:**
   - Prioritize high-risk messages
   - Quick response to emergencies

3. **Mental Health Apps:**
   - Check-in content analysis
   - Early warning system
   - Connect users to resources

### Limitations

**Model Limitations:**
1. **Context understanding:**
   - May miss sarcasm
   - Cultural nuances
   - Metaphorical language

2. **False positives:**
   - Song lyrics about death
   - Academic discussions
   - Creative writing

3. **False negatives:**
   - Subtle cries for help
   - Coded language
   - Masked distress

**Ethical Considerations:**
1. Privacy concerns
2. Need for human review
3. Support resource integration
4. Consent and transparency

### Improvements

**Potential enhancements:**
1. **Multilingual support:**
   - Train on multiple languages
   - Cross-cultural validation

2. **Contextual features:**
   - User history
   - Time patterns
   - Social network analysis

3. **Multi-modal analysis:**
   - Combine with images
   - Tone analysis (if audio)
   - Emoji interpretation

4. **Real-time monitoring:**
   - Stream processing
   - Immediate alerts
   - Integration with crisis services

---

## TROUBLESHOOTING

### Common Issues

**Issue 1: predict_classes() not found**
- **Cause:** Using newer TensorFlow version
- **Solution:** Use `np.argmax(model.predict(X), axis=1)`
- **Or:** Downgrade to TF 2.3.0, Keras 2.4.3

**Issue 2: GloVe file too large**
- **Cause:** 3GB file, memory constraints
- **Solution:** Use smaller GloVe (50d, 100d)
- **Or:** Increase RAM allocation

**Issue 3: Tokenizer errors**
- **Cause:** Mismatch between saved and current
- **Solution:** Retrain with consistent maxlen

**Issue 4: Low accuracy**
- **Cause:** Data quality, hyperparameters
- **Solution:** Tune learning rate, batch size, epochs

---

## BEST PRACTICES

### Data Preparation
1. Balance dataset
2. Clean text thoroughly
3. Remove duplicates
4. Validate labels

### Model Training
1. Use early stopping
2. Monitor validation metrics
3. Save best weights
4. Regular checkpoints

### Production Deployment
1. Version control models
2. API rate limiting
3. Human review workflow
4. Privacy compliance

---

## FILES IN PROJECT

**Working directory contains:**
1. `suicide_detection.csv` - Dataset
2. `glove.840B.300d.pkl` - GloVe embeddings (3GB)
3. `tokenizer.pkl` - Saved tokenizer
4. `suicide_detection_model.h5` - Trained model
5. `app.py` - Streamlit application
6. `notebook.ipynb` - Jupyter notebook (training code)

---

## CONCLUSION

**What was accomplished:**
1. Built LSTM model with 93% accuracy
2. Integrated GloVe word embeddings
3. Created web interface with Streamlit
4. Deployed functional prototype

**Key takeaways:**
- NLP can help mental health crisis detection
- Pre-trained embeddings save time
- LSTM effective for sequence data
- User-friendly interfaces crucial

**Important reminder:**
This is a support tool, not a replacement for professional mental health services. Always combine AI detection with human expertise and crisis response protocols.

---

*This project demonstrates the application of deep learning for social good, specifically mental health crisis detection. The combination of LSTM neural networks with GloVe embeddings creates a powerful system for identifying concerning content in social media posts, potentially saving lives through early intervention.*
