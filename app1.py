from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
import os
import pandas as pd

app = Flask(__name__)

# Ensure models directory exists
MODEL_DIR = "models"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Load AI text detection model and tokenizer
try:
    ai_text_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "ai_text_detection_model.h5"))
    ai_text_vectorizer = pickle.load(open(os.path.join(MODEL_DIR, "ai_text_vectorizer.pkl"), "rb"))
except Exception as e:
    print(f"Error loading AI text detection model: {e}")
    ai_text_model = None
    ai_text_vectorizer = None

# Load Personality Prediction Model and Tokenizer
try:
    bilstm_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "bilstm_model.h5"))
    label_encoder = pickle.load(open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb"))
    tokenizer = pickle.load(open(os.path.join(MODEL_DIR, "tokenizer.pkl"), "rb"))
except Exception as e:
    print(f"Error loading personality prediction model: {e}")
    bilstm_model = None
    label_encoder = None
    tokenizer = None

# Load MBTI Career Mapping Model and Encoders
try:
    rf_model = pickle.load(open(os.path.join(MODEL_DIR, "mbti_career_model.pkl"), "rb"))
    mbti_encoder = pickle.load(open(os.path.join(MODEL_DIR, "mbti_label_encoder.pkl"), "rb"))
    career_mlb = pickle.load(open(os.path.join(MODEL_DIR, "career_mlb.pkl"), "rb"))
except Exception as e:
    print(f"Error loading MBTI career mapping model: {e}")
    rf_model = None
    mbti_encoder = None
    career_mlb = None


def detect_ai_generated(text):
    if not ai_text_model or not ai_text_vectorizer:
        print("AI text detection model or tokenizer not loaded. Skipping detection.")
        return False  # Skip AI detection if models are not loaded

    try:
        # Tokenize the input text
        words = text.split()
        vectorized_text = [ai_text_vectorizer.get(w, 0) for w in words]  # Convert words to indices
        padded_sequence = pad_sequences([vectorized_text], maxlen=340)

        # Predict if the text is AI-generated
        prediction = ai_text_model.predict(padded_sequence)
        is_ai_generated = prediction[0][0] > 0.85  # Returns True if AI-generated

        print(f"AI text detection prediction: {prediction[0][0]} (Threshold: 0.8)")
        return is_ai_generated
    except Exception as e:
        print(f"Error during AI text detection: {e}")
        return False  # Skip detection if an error occurs
    
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("Received request data:", data)  # Debugging
        user_text = data.get("text", "").strip()

        # Validate input length
        if not user_text or len(user_text.split()) < 5:  # Minimum 5 words
            print("Error: Input text is too short or empty")  # Debugging
            return jsonify({"error": "Please enter a meaningful text with at least 5 words."}), 400

        # List of generic phrases to reject (only if the entire input matches)
        generic_phrases = ["hi", "hello", "bye", "excuse me", "thank you", "good morning", "good night"]
        if user_text.lower().strip() in generic_phrases:
            print("Error: Generic or short input detected")  # Debugging
            return jsonify({"error": "Please enter a meaningful text, not just greetings or short phrases."}), 400

        # Check if AI-generated
        if detect_ai_generated(user_text):
            print("Error: AI-generated text detected")  # Debugging
            return jsonify({"error": "AI-generated text detected. Please enter a human-written text."}), 400

        # Ensure models are loaded
        if not bilstm_model or not label_encoder or not tokenizer:
            return jsonify({"error": "Personality prediction model is not available."}), 500

        # Convert user text to sequence
        sequence = tokenizer.texts_to_sequences([user_text])
        padded_sequence = pad_sequences(sequence, maxlen=100)

        # Predict personality
        prediction = bilstm_model.predict(padded_sequence)
        predicted_class = np.argmax(prediction, axis=1)[0]
        personality_type = label_encoder.classes_[predicted_class]

        # Ensure MBTI type is valid
        if personality_type not in mbti_encoder.classes_:
            return jsonify({"error": "Invalid MBTI type provided."}), 400

        # Encode the MBTI type for the career prediction model
        mbti_encoded = mbti_encoder.transform([personality_type])

        # Fix: Convert MBTI encoded value into a proper DataFrame
        mbti_encoded_df = pd.DataFrame(mbti_encoded.reshape(-1, 1), columns=["mbti_encoded"])

        # Predict career recommendations
        career_predictions = rf_model.predict(mbti_encoded_df)
        recommended_careers = career_mlb.inverse_transform(career_predictions)[0]

        # Handle empty recommendations
        if not recommended_careers:
            recommended_careers = ["No career recommendations available."]

        # Personality descriptions
        personality_info = {
            "INTJ": "INTJ (Architect) - Strategic, logical, and independent thinkers. It is a personality type with the Introverted, Intuitive, Thinking, and Judging traits. These thoughtful tacticians love perfecting the details of life, applying creativity and rationality to everything they do. Their inner world is often a private, complex one.",
            "INTP": "INTP (Logician) - Analytical, curious, and inventive problem-solvers. It is a personality type with the Introverted, Intuitive, Thinking, and Prospecting traits. These flexible thinkers enjoy taking an unconventional approach to many aspects of life. They often seek out unlikely paths, mixing willingness to experiment with personal creativity.",
            "ENTJ": "ENTJ (Commander) - Bold, decisive, and natural leaders.It is a personality type with the Extraverted, Intuitive, Thinking, and Judging traits. They are decisive people who love momentum and accomplishment. They gather information to construct their creative visions but rarely hesitate for long before acting on them.",
            "ENTP": "ENTP (Debater) - Energetic, quick-witted, and challenge-driven individuals. It  is a personality type with the Extraverted, Intuitive, Thinking, and Prospecting traits. They tend to be bold and creative, deconstructing and rebuilding ideas with great mental agility. They pursue their goals vigorously despite any resistance they might encounter.",
            "INFJ": "INFJ (Advocate) - Visionary, deep-thinking, and compassionate. It is a personality type with the Introverted, Intuitive, Feeling, and Judging traits. They tend to approach life with deep thoughtfulness and imagination. Their inner vision, personal values, and a quiet, principled version of humanism guide them in all things.",
            "INFP": "INFP (Mediator) - Creative, introspective, and idealistic dreamers. It is a personality type with the Introverted, Intuitive, Feeling, and Prospecting traits. These rare personality types tend to be quiet, open-minded, and imaginative, and they apply a caring and creative approach to everything they do.",
            "ENFJ": "ENFJ (Protagonist) - Charismatic, inspiring, and people-focused leaders. It is a personality type with the Extraverted, Intuitive, Feeling, and Judging traits. These warm, forthright types love helping others, and they tend to have strong ideas and values. They back their perspective with the creative energy to achieve their goals.",
            "ENFP": "ENFP (Campaigner) - Enthusiastic, free-spirited, and open-minded adventurers. It  is a personality type with the Extraverted, Intuitive, Feeling, and Prospecting traits. These people tend to embrace big ideas and actions that reflect their sense of hope and goodwill toward others. Their vibrant energy can flow in many directions.",
            "ISTJ": "ISTJ (Logistician) - Organized, detail-oriented, and highly responsible. It  is a personality type with the Introverted, Observant, Thinking, and Judging traits. These people tend to be reserved yet willful, with a rational outlook on life. They compose their actions carefully and carry them out with methodical purpose.",
            "ISFJ": "ISFJ (Defender) - Warm-hearted, dedicated, and service-oriented. It  is a personality type with the Introverted, Observant, Feeling, and Judging traits. These people tend to be warm and unassuming in their own steady way. They’re efficient and responsible, giving careful attention to practical details in their daily lives.",
            "ESTJ": "ESTJ (Executive) - Efficient, hardworking, and practical decision-makers. It  is a personality type with the Extraverted, Observant, Thinking, and Judging traits. They possess great fortitude, emphatically following their own sensible judgment. They often serve as a stabilizing force among others, able to offer solid direction amid adversity.",
            "ESFJ": "ESFJ (Consul) - Supportive, loyal, and community-oriented individuals.It is a personality type with the Extraverted, Observant, Feeling, and Judging traits. They are attentive and people-focused, and they enjoy taking part in their social community. Their achievements are guided by decisive values, and they willingly offer guidance to others.", 
            "ISTP": "ISTP (Virtuoso) - Practical, hands-on, and adventurous problem-solvers.It is a personality type with the Introverted, Observant, Thinking, and Prospecting traits. They tend to have an individualistic mindset, pursuing goals without needing much external connection. They engage in life with inquisitiveness and personal skill, varying their approach as needed.",
            "ISFP": "ISFP (Adventurer) - Artistic, flexible, and in-the-moment explorers.It is a personality type with the Introverted, Observant, Feeling, and Prospecting traits. They tend to have open minds, approaching life, new experiences, and people with grounded warmth. Their ability to stay in the moment helps them uncover exciting potentials.",
            "ESTP": "ESTP (Entrepreneur) - Energetic, risk-taking, and action-oriented. It is a personality type with the Extraverted, Observant, Thinking, and Prospecting traits. They tend to be energetic and action-oriented, deftly navigating whatever is in front of them. They love uncovering life’s opportunities, whether socializing with others or in more personal pursuits.",
            "ESFP": "ESFP (Entertainer) - Fun-loving, expressive, and highly social individuals.It is a personality type with the Extraverted, Observant, Feeling, and Prospecting traits. These people love vibrant experiences, engaging in life eagerly and taking pleasure in discovering the unknown. They can be very social, often encouraging others into shared activities."
        }


        personality_desc = personality_info.get(personality_type, "No description available.")

        return jsonify({
            "personality": personality_type,
            "description": personality_desc,
            "recommended_jobs": list(recommended_careers)
        })

    except Exception as e:
        print("Error:", str(e))  # Debugging
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/result')
def result():
    personality_type = request.args.get('personality')
    description = request.args.get('description')
    recommended_jobs = request.args.get('recommended_jobs')
    return render_template('result.html', personality=personality_type, description=description, recommended_jobs=recommended_jobs)

if __name__ == '__main__':
    app.run(debug=True)