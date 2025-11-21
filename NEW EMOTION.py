import streamlit as st
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, Conv1D, GlobalMaxPooling1D, GRU, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import Constant
from tensorflow.keras.regularizers import l2 # NEW: Import L2 Regularizer
import os
import sys
from collections import Counter
import re

# Suppress TensorFlow logging messages and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

# --- Configuration (MAXIMAL Stable Capacity for 94%+ Accuracy) ---
MAX_WORDS = 20000       
MAX_LEN = 150           
EMBEDDING_DIM = 128     
RNN_UNITS = 300         # Maximal stable unit count
DENSE_UNITS = 768       # Maximal stable unit count
NUM_CLASSES = 6
EPOCHS = 15             
NUM_REVIEWS = 10        
TRAINABLE_EMBEDDING = True 
REGULARIZATION_RATE = 1e-4 # NEW: L2 Regularization rate for stability

# Define the emotion labels for mapping
emotion_labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
label_to_id = {label: i for i, label in enumerate(emotion_labels)}
id_to_label = {i: label for i, label in enumerate(emotion_labels)}

# Custom Samples designed for clear classification tests
SAMPLE_REVIEWS = {
    "sadness": "I am not happy with this purchase. It makes me feel miserable and disappointed.", 
    "joy": "This product is amazing and fills me with joy! I am absolutely ecstatic.",
    "love": "I absolutely adore the design and quality, I'm completely in love.",
    "anger": "It broke immediately and this makes me so furious and upset. I hate it.",
    "fear": "I am afraid to use this device after the smoke I saw, it is worrying.",
    "surprise": "Wow! I truly did not expect it to be this good. What a pleasant surprise." 
}

# --- Preprocessing Function (Negation Handling) ---

def handle_negation(texts):
    """
    Handles negation by modifying phrases and appending a global negation flag.
    """
    negation_words = ['not', 'no', 'never', "don't", "isn't", "wasn't", "wouldn't", 
                      "couldn't", "won't", "can't", 'do not', "did not", "will not",
                      'hardly', 'scarcely']
    
    processed_texts = []
    
    for text in texts:
        did_negate = False
        words = text.split()
        
        for i, word in enumerate(words):
            
            # Check for standard contractions or single-word negators
            if word.lower() in negation_words and i + 1 < len(words):
                 # Create a specialized token (e.g., 'not_happy')
                 if word.lower() == 'not':
                    words[i+1] = f"not_{words[i+1]}"
                 else:
                    words[i+1] = f"{word}_{words[i+1]}"
                    
                 words[i] = '' 
                 did_negate = True
            
            # 2. Look for prefixes (un, in, etc.)
            if re.match(r'^(un|in|im|ir)\w+', word.lower()):
                 did_negate = True
        
        new_text = " ".join(filter(None, words))
        
        if did_negate:
            new_text = f"{new_text} __NEGATED__"
            
        processed_texts.append(new_text)
        
    return processed_texts


# --- Ensemble Model Building Functions ---

def create_embedding_layer(num_words, embedding_matrix=None):
    """Creates the Embedding layer, initialized with pre-trained vectors if provided."""
    return Embedding(
        input_dim=num_words,
        output_dim=EMBEDDING_DIM,
        weights=[embedding_matrix] if embedding_matrix is not None else None,
        input_length=MAX_LEN,
        trainable=TRAINABLE_EMBEDDING
    )


def build_cnn_model(num_words, embedding_matrix):
    """Builds the deep CNN model with L2 regularization for stability."""
    model = Sequential([
        create_embedding_layer(num_words, embedding_matrix),
        Conv1D(filters=RNN_UNITS, kernel_size=5, activation='relu', padding='same',
               kernel_regularizer=l2(REGULARIZATION_RATE)),
        Conv1D(filters=RNN_UNITS // 2, kernel_size=3, activation='relu', padding='same', 
               kernel_regularizer=l2(REGULARIZATION_RATE)), 
        GlobalMaxPooling1D(),
        Dense(DENSE_UNITS, activation='relu', kernel_regularizer=l2(REGULARIZATION_RATE)), 
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_bilstm_model(num_words, embedding_matrix):
    """
    Builds the two-layer BiLSTM model with L2 regularization for stability.
    """
    model = Sequential([
        create_embedding_layer(num_words, embedding_matrix),
        Dropout(0.3),
        Bidirectional(LSTM(RNN_UNITS, return_sequences=True, dropout=0.1, 
                           kernel_regularizer=l2(REGULARIZATION_RATE))), # L2 added
        Bidirectional(LSTM(RNN_UNITS, kernel_regularizer=l2(REGULARIZATION_RATE))), # L2 added
        Dense(DENSE_UNITS, activation='relu', kernel_regularizer=l2(REGULARIZATION_RATE)), 
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_gru_model(num_words, embedding_matrix):
    """Builds the Bidirectional GRU model with L2 regularization for stability."""
    model = Sequential([
        create_embedding_layer(num_words, embedding_matrix),
        Dropout(0.3),
        Bidirectional(GRU(RNN_UNITS, dropout=0.1, kernel_regularizer=l2(REGULARIZATION_RATE))), # L2 added
        Dense(DENSE_UNITS, activation='relu', kernel_regularizer=l2(REGULARIZATION_RATE)), 
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# --- Caching function to load data and train the model once ---

@st.cache_resource(show_spinner=True)
def load_and_train_model():
    """Loads data, trains the Ensemble of three models, and evaluates them."""
    st.info("Loading and pre-processing dataset...")
    
    # 1. Load Data
    data = load_dataset("dair-ai/emotion", "split")
    
    train_texts = list(data['train']['text']) + list(data['validation']['text'])
    train_labels = list(data['train']['label']) + list(data['validation']['label'])
    test_texts = list(data['test']['text'])
    test_labels = list(data['test']['label'])
    
    st.info("Applying custom negation preprocessing...")
    train_texts = handle_negation(train_texts)
    test_texts = handle_negation(test_texts)
    
    all_texts = train_texts + test_texts

    # 2. Tokenization and Sequencing
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<unk>")
    tokenizer.fit_on_texts(all_texts + ['__NEGATED__']) 
    
    num_words = min(MAX_WORDS, len(tokenizer.word_index) + 1)

    train_sequences = tokenizer.texts_to_sequences(train_texts)
    test_sequences = tokenizer.texts_to_sequences(test_texts)
    
    train_padded = pad_sequences(train_sequences, maxlen=MAX_LEN, padding='post', truncating='post')
    test_padded = pad_sequences(test_sequences, maxlen=MAX_LEN, padding='post', truncating='post')

    # Convert labels to one-hot encoding
    train_labels_one_hot = tf.keras.utils.to_categorical(train_labels, num_classes=NUM_CLASSES)
    
    # 3. Class Weighting (Anti-Bias Fix)
    st.info("Calculating class weights to counteract dataset bias...")
    total_samples = len(train_labels)
    class_counts = np.bincount(train_labels)
    class_weights = {}
    for i in range(NUM_CLASSES):
        class_weights[i] = total_samples / (NUM_CLASSES * class_counts[i])
        
    # 4. CRITICAL: Anti-Negation Embedding Initialization
    st.info("Initializing embedding matrix with anti-negation mirror semantics...")
    embedding_matrix = np.random.uniform(-0.05, 0.05, (num_words, EMBEDDING_DIM))
    
    # Perform semantic initialization (requires original words and negated forms to be indexed)
    for word, index in tokenizer.word_index.items():
        if index >= num_words:
            continue
            
        # Check for our manually negated tokens (e.g., 'not_happy', 'never_sad')
        if word.startswith('not_') or word.startswith('never_'):
            original_word = word.split('_', 1)[1] 
            original_index = tokenizer.word_index.get(original_word)
            
            if original_index is not None and original_index < num_words:
                # Initialize the negated token's embedding as the inverse of the original word's embedding
                embedding_matrix[index] = -embedding_matrix[original_index]
            
    # Set unique initialization for the global negation flag
    negated_index = tokenizer.word_index.get('__negated__', 0)
    if negated_index < num_words:
        embedding_matrix[negated_index] = np.random.uniform(-0.1, 0.1, EMBEDDING_DIM)


    # 5. Build and Train Ensemble Models
    st.info(f"Building and training ensemble of 3 deep learning models for up to {EPOCHS} epochs...")
    
    models = [
        build_cnn_model(num_words, embedding_matrix),
        build_bilstm_model(num_words, embedding_matrix),
        build_gru_model(num_words, embedding_matrix)
    ]

    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=5, 
        restore_best_weights=True, 
        verbose=0 
    )

    # Train all models silently
    for i, model in enumerate(models):
        st.info(f"Training Model {i+1}...")
        try:
            model.fit(
                train_padded, 
                train_labels_one_hot,
                epochs=EPOCHS, 
                batch_size=128, 
                validation_split=0.1,
                class_weight=class_weights, 
                callbacks=[early_stopping],
                verbose=0 
            )
        except Exception as e:
            st.error(f"Error during training Model {i+1}: {e}")
            continue
            
    # 6. Ensemble Prediction and Evaluation
    
    pred_probs_list = [model.predict(test_padded, verbose=0) for model in models if model is not None]
    
    if not pred_probs_list:
        st.error("All models failed to train. Cannot proceed with evaluation.")
        return models, tokenizer, {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1_score': 0}
        
    # Soft Voting: Average the probabilities across all models
    ensemble_probs = np.mean(pred_probs_list, axis=0)
    
    # Final prediction based on ensemble average
    y_pred = np.argmax(ensemble_probs, axis=1)
    
    # Calculate Metrics based on ensemble prediction
    accuracy = accuracy_score(test_labels, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, y_pred, average='macro', zero_division=0
    )
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

    st.success(f"Model Training Complete! Ensemble Accuracy: {accuracy:.4f}")
    return models, tokenizer, metrics


# --- Prediction Function ---
def predict_emotion(ensemble_models, tokenizer, text):
    """Predicts the emotion of a given review text using the ensemble models (Soft Voting)."""
    # 1. Apply the same negation preprocessing as training
    preprocessed_text = handle_negation([text])[0]
    
    # 2. Tokenize and pad
    sequence = tokenizer.texts_to_sequences([preprocessed_text])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')
    
    # 3. Predict using ensemble
    pred_probs_list = [model.predict(padded_sequence, verbose=0) for model in ensemble_models if model is not None]
    
    if not pred_probs_list:
        return "ERROR", None
        
    ensemble_prediction = np.mean(pred_probs_list, axis=0)[0]
    
    predicted_id = np.argmax(ensemble_prediction)
    predicted_label = id_to_label[predicted_id].capitalize()
    
    return predicted_label


# --- Core Analysis Logic ---
def get_recommendation_and_comment(all_emotions):
    """
    Determines the overall recommended emotion and the buy/no-buy comment.
    """
    if not all_emotions:
        return "N/A", "Please enter at least one review for analysis.", {}

    emotion_counts = Counter(all_emotions)
    
    max_count = max(emotion_counts.values())
    top_emotions = [emotion for emotion, count in emotion_counts.items() if count == max_count]

    dominant_emotion = top_emotions[0] 
    
    if len(top_emotions) > 1:
        last_review_emotion = all_emotions[-1]
        if last_review_emotion in top_emotions:
            dominant_emotion = last_review_emotion

    positive_buy_emotions = ['Joy', 'Love']
    
    if dominant_emotion in positive_buy_emotions:
        comment = (
            f"**Recommendation: BUY!** The dominant sentiment is strongly positive ({dominant_emotion}), "
            "indicating a highly satisfied customer base. This is a strong indicator of product quality."
        )
    elif dominant_emotion == 'Surprise':
        comment = (
            f"**Recommendation: CAUTIOUS BUY.** The dominant sentiment is 'Surprise'. While positive, "
            "it suggests a wide range of unexpected outcomes. Review customer feedback carefully."
        )
    else:
        # Sadness, Anger, Fear
        comment = (
            f"**Recommendation: DO NOT BUY.** The dominant sentiment is negative ({dominant_emotion}), "
            "suggesting significant customer dissatisfaction or potential product issues. Caution is advised."
        )

    return dominant_emotion, comment, emotion_counts

def create_simulated_word_cloud(emotion_counts, dominant_emotion):
    """
    Creates a simulated word cloud using HTML/CSS, where text size is based on count, 
    and the dominant emotion is highlighted and proportionally larger.
    """
    if not emotion_counts:
        return ""
        
    max_count = max(emotion_counts.values())
    min_count = min(emotion_counts.values())
    
    count_range = max_count - min_count
    if count_range == 0:
        count_range = 1 

    html_content = '<div style="display: flex; flex-wrap: wrap; justify-content: center; align-items: center; gap: 15px; margin-top: 30px; background: rgba(0, 0, 0, 0.4); padding: 20px; border-radius: 12px; min-height: 200px;">'
    
    color_map = {
        'Joy': '#4CAF50',       
        'Love': '#FF69B4',      
        'Surprise': '#FFD700',  
        'Sadness': '#2196F3',   
        'Anger': '#F44336',     
        'Fear': '#FF9800',      
        'N/A': '#9E9E9E'        
    }

    data_to_render = [(emotion.capitalize(), count) for emotion, count in emotion_counts.items()]
    
    for emotion, count in data_to_render:
        if count_range > 1:
            size_ratio = (count - min_count) / count_range
            font_size = 18 + (size_ratio * 30) 
        else:
            font_size = 36 

        
        if emotion == dominant_emotion:
            font_size *= 1.2
            emotion_style = f'font-size: {font_size:.0f}px; color: {color_map.get(emotion, "#FFFFFF")}; font-weight: 900; text-shadow: 0 0 10px #FFD700;'
            count_style = f'font-size: {font_size * 0.9:.0f}px; color: #FFD700; font-weight: 900; margin-left: -5px;'
        else:
            emotion_style = f'font-size: {font_size:.0f}px; color: {color_map.get(emotion, "#FFFFFF")}; opacity: 0.8; font-weight: 700;'
            count_style = f'font-size: {font_size * 0.8:.0f}px; color: #E0F7FA; font-weight: 700; margin-left: -5px;'
        
        html_content += f'<span style="display: inline-flex; align-items: baseline;">'
        html_content += f'<span style="{emotion_style}">{emotion}&nbsp;&nbsp;</span>'
        html_content += f'<span style="{count_style}">({count})</span>'
        html_content += f'</span>'


    html_content += '</div>'
    return html_content


# --- Main Streamlit App ---
def main():
    
    # Initialize session state for analysis results
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = []
    if 'product_name' not in st.session_state:
        st.session_state.product_name = "Affectlytics Product"
    if 'is_trained' not in st.session_state:
        st.session_state.is_trained = False
    if 'metrics' not in st.session_state:
         st.session_state.metrics = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1_score': 0}


    # --- CSS Injection ---
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;900&display=swap');
        
        /* Animated Gradient Background */
        .stApp {
            font-family: 'Poppins', sans-serif;
            color: #FFFFFF;
            background: linear-gradient(-45deg, #0f002a, #2b0846, #002a3a, #00404a); 
            background-size: 400% 400%;
            animation: gradientBG 25s ease infinite;
        }
        @keyframes gradientBG {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        h1, h2, h3, h4 { color: #FFFFFF !important; letter-spacing: 1.2px; }
        .header {
            color: #FFFFFF; text-align: center; padding: 15px; border-radius: 12px;
            background: rgba(0, 0, 0, 0.4); box-shadow: 0 4px 10px rgba(0, 0, 0, 0.7);
            margin-bottom: 25px;
        }
        .stButton>button {
            border-radius: 10px; transition: all 0.3s ease; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.4);
            color: white; background-color: #5b1076;
        }
        .stButton>button:hover {
            transform: translateY(-2px); box-shadow: 0 8px 16px rgba(0, 0, 0, 0.6); background-color: #7b1296;
        }
        .stTextInput input, .stTextArea textarea {
            background-color: rgba(0, 0, 0, 0.3) !important; color: white !important;
            border: 1px solid #7b1296;
        }
        
        /* CRITICAL: Style for Input Labels */
        div[data-testid="stTextInput"] label p, div[data-testid="stTextArea"] label p {
            color: #FFD700 !important; /* Gold text for input labels */
            font-weight: 700;
        }

        /* Specific styling for the recommendation box */
        .recommendation-box {
            background-color: #3d0a52; padding: 20px; border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.8); margin-top: 20px; border: 2px solid #FFD700;
        }
        .emotion-badge {
            display: inline-block; padding: 5px 10px; border-radius: 8px; font-weight: 600;
            margin-left: 5px; color: white; background-color: #7b1296;
        }
        /* Metric Box Styling (Enhanced) */
        .metric-box {
            background-color: #5b1076; 
            padding: 15px; 
            border-radius: 10px; 
            text-align: center;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.6);
            margin-bottom: 10px;
        }
        .metric-label {
            font-size: 14px; 
            color: #ccc; 
            font-weight: 400;
        }
        .metric-value {
            font-size: 24px; 
            color: #FFD700; 
            font-weight: 900;
        }
        
        /* --- HIGH-CONTRAST TABLE STYLING --- */
        .stTable thead th {
             color: #FFD700 !important; /* Gold header text */
             background-color: #5b1076 !important; /* Purple header background */
             font-weight: 800;
        }
        .stTable tbody tr th, .stTable tbody tr td {
            color: #E0F7FA !important; /* Lighter text color for contrast */
            background-color: rgba(61, 10, 82, 0.7) !important; /* Darker, contrasting purple background */
            border-bottom: 1px solid #FFD700;
            word-break: normal; white-space: normal;
            font-weight: 600; 
        }
        /* --- END HIGH-CONTRAST TABLE STYLING --- */
        </style>
        """,
        unsafe_allow_html=True
    )

    # UPDATED APP NAME
    st.markdown('<div class="header"><h1><span style="color: #FFD700;">🛍️</span> Affectlytics: Product Emotion Analyzer <span style="color: #FFD700;">📊</span></h1></div>', unsafe_allow_html=True)
    
    # Load and train the model (cached)
    try:
        models, tokenizer, metrics = load_and_train_model()
        st.session_state.is_trained = True
        st.session_state.metrics = metrics
    except Exception as e:
        # In case of initialization failure, log error and show a warning
        st.error(f"Failed to initialize model: {e}")
        st.session_state.is_trained = False
        return

    # --- Input Section ---
    st.markdown("---")
    st.markdown("<h3 style='color: white;'>Input Product Details and Reviews (10 Required):</h3>", unsafe_allow_html=True)
    
    # Product Name Input
    product_name_input = st.text_input(
        "Product Name", 
        value=st.session_state.product_name,
        key="product_name_key"
    )
    st.session_state.product_name = product_name_input
    
    # 10 Review Text Inputs
    review_inputs = []
    
    # Use columns for a cleaner 2-column layout for reviews
    cols = st.columns(2)

    for i in range(NUM_REVIEWS):
        col_index = i % 2
        with cols[col_index]:
            # Ensure the default review capitalizes the emotion label
            emotion_key = emotion_labels[i % 6]
            default_review = SAMPLE_REVIEWS.get(emotion_key, f"Review #{i+1} Example: This is fine, but nothing special.")
            
            # Use sticky input value if available
            sticky_value = st.session_state.analysis_results[i]['review'] if len(st.session_state.analysis_results) == NUM_REVIEWS else default_review
            
            review_text = st.text_area(
                f"Review #{i+1}", 
                value=sticky_value, 
                height=100, 
                key=f"review_{i}"
            )
            review_inputs.append(review_text)

    # --- Analysis Button ---
    if st.button("Analyze All 10 Reviews", use_container_width=True, type="primary"):
        if not st.session_state.is_trained:
            st.warning("Model is not fully initialized. Please wait a moment and try again.")
            return

        st.session_state.analysis_results = []
        all_emotions = []

        # Process each review
        for i, review in enumerate(review_inputs):
            if not review.strip():
                st.warning(f"Review #{i+1} is empty. Please fill all 10 inputs.")
                st.session_state.analysis_results = [] # Clear results if incomplete
                return

            predicted_emotion = predict_emotion(models, tokenizer, review)
            all_emotions.append(predicted_emotion)
            
            st.session_state.analysis_results.append({
                'product': st.session_state.product_name,
                'review_number': i + 1,
                'review': review,
                'emotion': predicted_emotion
            })

        # Generate final summary and recommendation
        dominant_emotion, recommendation_comment, emotion_counts = get_recommendation_and_comment(all_emotions)
        
        # Convert keys to capitalized for consistency with prediction results
        emotion_counts_capitalized = {k.capitalize(): v for k, v in emotion_counts.items()}
        
        st.session_state.dominant_emotion = dominant_emotion
        st.session_state.recommendation_comment = recommendation_comment
        st.session_state.emotion_counts = emotion_counts_capitalized


    # --- Results Display ---
    if st.session_state.analysis_results:
        st.markdown("<hr style='border: 1px solid #FFD700;'>", unsafe_allow_html=True)
        st.markdown(f"<h2>Analysis Results for: <span style='color:#FFD700;'>{st.session_state.product_name}</span></h2>", unsafe_allow_html=True)
        
        # 1. Final Recommendation and Buy/No-Buy Comment
        st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
        st.markdown(
            f"<h3>Final Recommendation Based on Sentiment:</h3>", 
            unsafe_allow_html=True
        )
        st.markdown(
            f"<h4>Dominant Emotion: <span class='emotion-badge'>{st.session_state.dominant_emotion}</span></h4>",
            unsafe_allow_html=True
        )
        st.markdown(st.session_state.recommendation_comment, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # 2. Display Word Cloud (Visualization Request)
        st.markdown("<h3 style='color: white; margin-top: 30px;'>Emotion Distribution Word Cloud:</h3>", unsafe_allow_html=True)
        
        word_cloud_html = create_simulated_word_cloud(
            st.session_state.emotion_counts, 
            st.session_state.dominant_emotion
        )
        st.markdown(word_cloud_html, unsafe_allow_html=True)

        # 3. Display Table with Detected Emotion
        st.markdown("<hr style='border: 1px solid #7b1296;'>", unsafe_allow_html=True)
        st.markdown("<h3 style='color: white;'>Detailed Review Analysis:</h3>", unsafe_allow_html=True)
        results_df = pd.DataFrame(st.session_state.analysis_results)
        results_df = results_df.rename(columns={
            'review_number': 'Review #',
            'review': 'Review Text',
            'emotion': 'Detected Emotion'
        })
        st.table(results_df[['Review #', 'Review Text', 'Detected Emotion']])


    # --- Evaluation Metrics Display (Modified to only show achieved accuracy) ---
    st.markdown("---")
    st.markdown("<h2 style='color: #FFD700; text-align: center;'>Model Evaluation Metrics (Background System)</h2>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    metrics = st.session_state.metrics

    def display_metric(col, label, value):
        col.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value:.4f}</div>
            </div>
        """, unsafe_allow_html=True)

    if st.session_state.is_trained:
        # Display all detailed metrics
        display_metric(col1, "Accuracy", metrics['accuracy'])
        display_metric(col2, "Macro Precision", metrics['precision'])
        display_metric(col3, "Macro Recall", metrics['recall'])
        display_metric(col4, "Macro F1-Score", metrics['f1_score'])
        
        # Display Achieved Accuracy at the bottom, as requested
        st.markdown(f"<p style='text-align: center; font-size: 1.5em; font-weight: 600; color: #4CAF50;'>Achieved Accuracy: {metrics['accuracy']:.4f}</p>", unsafe_allow_html=True)
        
    else:
        st.warning("Metrics unavailable: Model failed to initialize or train.")

if __name__ == "__main__":
    main()
